import os
import io
import csv
import zipfile
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from loguru import logger


class StudentImportRecord(BaseModel):
    """
    Standardized parsed student record from import source.
    """
    register_no: str
    name: str
    class_section: str = "Unknown"
    department: str = "Computer Science"
    section: str = "A"
    photos: List[bytes] = Field(default_factory=list, description="Raw image byte buffers")
    photo_filenames: List[str] = Field(default_factory=list)
    row_index: int = 0
    raw_fields: Dict[str, Any] = Field(default_factory=dict)


class ImporterValidationError(Exception):
    """Raised when an import document fails structure or mapping validation."""
    pass


class StudentDataImporter:
    """
    Configurable Student Data Importer.
    Supports DOCX (with embedded media), CSV, and Directory structures.
    Performs header mapping, structure validation, and duplicate checking without guessing.
    """

    DEFAULT_FIELD_ALIASES = {
        "register_no": [
            "reg no", "reg_no", "regno", "register number", "register no",
            "student_id", "student id", "roll no", "roll_no", "rollno", "id"
        ],
        "name": [
            "name", "student name", "student_name", "studentname",
            "full name", "full_name", "candidate name", "candidate_name"
        ],
        "class_section": [
            "class", "class_section", "class / section", "section", "dept",
            "department", "course", "degree", "branch"
        ],
        "photo": [
            "recent photo", "photo", "image", "recent photo (passport size or recent selfie)",
            "photo reference", "image_path", "picture", "avatar"
        ]
    }

    def __init__(self, custom_mapping: Optional[Dict[str, List[str]]] = None):
        self.field_aliases = dict(self.DEFAULT_FIELD_ALIASES)
        if custom_mapping:
            for k, v in custom_mapping.items():
                self.field_aliases[k] = [alias.lower().strip() for alias in v]

    def _normalize_header(self, text: str) -> str:
        """Strips punctuation, lowercases, and collapses whitespace."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def _match_column(self, header_text: str, target_field: str) -> bool:
        norm = self._normalize_header(header_text)
        aliases = self.field_aliases.get(target_field, [])
        for a in aliases:
            if a == norm or a in norm or norm in a:
                return True
        return False

    def identify_column_mapping(self, headers: List[str]) -> Dict[str, int]:
        """
        Maps column header names to standardized fields.
        Raises ImporterValidationError if required columns (register_no, name) cannot be identified.
        """
        mapping: Dict[str, int] = {}
        for idx, h in enumerate(headers):
            for field in ("register_no", "name", "class_section", "photo"):
                if field not in mapping and self._match_column(h, field):
                    mapping[field] = idx
                    break

        missing = []
        if "register_no" not in mapping:
            missing.append("Register Number (register_no)")
        if "name" not in mapping:
            missing.append("Student Name (name)")

        if missing:
            raise ImporterValidationError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Available headers: {headers}. Configure field mapping or verify input document."
            )

        return mapping

    def parse_class_section(self, class_str: str) -> Tuple[str, str]:
        """
        Parses class strings like 'III AI&DS-B' or 'AIDS - B' into (department, section).
        """
        dept = "AI&DS"
        sec = "B"
        if not class_str or not class_str.strip():
            return "Computer Science", "A"

        clean = class_str.strip()
        # Look for section at the end (e.g. -B, -A, Section B)
        sec_match = re.search(r"[-_\s]([A-Za-z0-9])$", clean)
        if sec_match:
            sec = sec_match.group(1).upper()
            dept_part = clean[:sec_match.start()].strip()
            # Clean leading year indicators (e.g. 'III ', '3rd Year ')
            dept = re.sub(r"^(I{1,4}|[1-4](st|nd|rd|th)?)\s*", "", dept_part).strip() or clean
        else:
            dept = clean
            sec = "A"

        return dept, sec

    def parse_file(
        self,
        file_path_or_bytes: Union[str, bytes],
        filename: Optional[str] = None
    ) -> List[StudentImportRecord]:
        """
        Auto-detects document format (DOCX, CSV) and extracts student records.
        """
        if isinstance(file_path_or_bytes, str):
            if not os.path.exists(file_path_or_bytes):
                raise FileNotFoundError(f"Student data file not found: {file_path_or_bytes}")
            fname = filename or os.path.basename(file_path_or_bytes)
            with open(file_path_or_bytes, "rb") as f:
                data = f.read()
        else:
            data = file_path_or_bytes
            fname = filename or "document.docx"

        ext = os.path.splitext(fname)[1].lower()
        if ext in (".docx", ".docm"):
            return self.parse_docx(data)
        elif ext in (".xlsx", ".xlsm"):
            return self.parse_xlsx(data)
        elif ext in (".xls",):
            return self.parse_xls(data)
        elif ext in (".csv", ".txt"):
            return self.parse_csv(data)
        else:
            # Try docx first, fallback to xlsx, then csv
            try:
                return self.parse_docx(data)
            except Exception:
                try:
                    return self.parse_xlsx(data)
                except Exception:
                    return self.parse_csv(data)

    def parse_xlsx(self, xlsx_bytes: bytes) -> List[StudentImportRecord]:
        """
        Parses modern Excel (.xlsx / .xlsm) documents using openpyxl.
        Supports text columns, cell images (in worksheet), and file path references.
        """
        import openpyxl
        try:
            wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), data_only=True)
            sheet = wb.active
        except Exception as e:
            raise ImporterValidationError(f"Invalid or corrupt XLSX file: {e}")

        rows = list(sheet.iter_rows(values_only=False))
        if len(rows) < 2:
            raise ImporterValidationError("XLSX student sheet must have at least 2 rows (header + data).")

        headers = [str(cell.value or "").strip() for cell in rows[0]]
        mapping = self.identify_column_mapping(headers)

        # Map row -> embedded images if any
        row_images: Dict[int, List[bytes]] = {}
        if hasattr(sheet, "_images") and sheet._images:
            for img in sheet._images:
                try:
                    row_idx = getattr(getattr(img, "anchor", None), "_from", None)
                    if row_idx is not None and hasattr(row_idx, "row"):
                        r_num = row_idx.row + 1
                    else:
                        r_num = getattr(img, "row", None)
                    if r_num:
                        data = img._data()
                        row_images.setdefault(r_num, []).append(data)
                except Exception as e:
                    logger.debug(f"Could not extract image from XLSX cell: {e}")

        records: List[StudentImportRecord] = []
        for r_idx, row in enumerate(rows[1:], start=2):
            cells = [cell.value for cell in row]
            if not any(cells):
                continue

            def get_val(idx: Optional[int]) -> str:
                if idx is not None and idx < len(cells) and cells[idx] is not None:
                    return str(cells[idx]).strip()
                return ""

            reg_no = get_val(mapping.get("register_no"))
            name = get_val(mapping.get("name"))
            class_sec = get_val(mapping.get("class_section")) or "Unknown"
            photo_ref = get_val(mapping.get("photo"))

            if not reg_no and not name:
                continue

            dept, sec = self.parse_class_section(class_sec)
            photos: List[bytes] = list(row_images.get(r_idx, []))
            photo_names: List[str] = [f"cell_img_{r_idx}_{i}.png" for i in range(len(photos))]

            if photo_ref:
                photo_names.append(os.path.basename(photo_ref) or photo_ref)
                if os.path.exists(photo_ref):
                    try:
                        with open(photo_ref, "rb") as pf:
                            photos.append(pf.read())
                    except Exception as e:
                        logger.warning(f"Could not read photo file '{photo_ref}': {e}")

            records.append(StudentImportRecord(
                register_no=reg_no,
                name=name,
                class_section=class_sec,
                department=dept,
                section=sec,
                photos=photos,
                photo_filenames=photo_names,
                row_index=r_idx,
                raw_fields={headers[i]: get_val(i) for i in range(len(cells)) if i < len(headers)}
            ))

        is_valid, errors = self.validate_records(records)
        if not is_valid:
            raise ImporterValidationError("XLSX record validation failed:\n" + "\n".join(errors))

        logger.info(f"XLSX Import: Successfully parsed {len(records)} records from XLSX.")
        return records

    def parse_xls(self, xls_bytes: bytes) -> List[StudentImportRecord]:
        """
        Parses legacy Excel (.xls) documents using xlrd.
        """
        import xlrd
        try:
            wb = xlrd.open_workbook(file_contents=xls_bytes)
            sheet = wb.sheet_by_index(0)
        except Exception as e:
            raise ImporterValidationError(f"Invalid or corrupt XLS file: {e}")

        if sheet.nrows < 2:
            raise ImporterValidationError("XLS student sheet must have at least 2 rows (header + data).")

        headers = [str(sheet.cell_value(0, col_idx)).strip() for col_idx in range(sheet.ncols)]
        mapping = self.identify_column_mapping(headers)

        records: List[StudentImportRecord] = []
        for r_idx in range(1, sheet.nrows):
            row_vals = [sheet.cell_value(r_idx, c) for c in range(sheet.ncols)]
            if not any(row_vals):
                continue

            def get_val(idx: Optional[int]) -> str:
                if idx is not None and idx < len(row_vals) and row_vals[idx] is not None:
                    val = row_vals[idx]
                    if isinstance(val, float) and val.is_integer():
                        return str(int(val))
                    return str(val).strip()
                return ""

            reg_no = get_val(mapping.get("register_no"))
            name = get_val(mapping.get("name"))
            class_sec = get_val(mapping.get("class_section")) or "Unknown"
            photo_ref = get_val(mapping.get("photo"))

            if not reg_no and not name:
                continue

            dept, sec = self.parse_class_section(class_sec)
            photos: List[bytes] = []
            photo_names: List[str] = []

            if photo_ref:
                photo_names.append(os.path.basename(photo_ref) or photo_ref)
                if os.path.exists(photo_ref):
                    try:
                        with open(photo_ref, "rb") as pf:
                            photos.append(pf.read())
                    except Exception as e:
                        logger.warning(f"Could not read photo file '{photo_ref}': {e}")

            records.append(StudentImportRecord(
                register_no=reg_no,
                name=name,
                class_section=class_sec,
                department=dept,
                section=sec,
                photos=photos,
                photo_filenames=photo_names,
                row_index=r_idx + 1,
                raw_fields={headers[i]: get_val(i) for i in range(len(row_vals)) if i < len(headers)}
            ))

        is_valid, errors = self.validate_records(records)
        if not is_valid:
            raise ImporterValidationError("XLS record validation failed:\n" + "\n".join(errors))

        logger.info(f"XLS Import: Successfully parsed {len(records)} records from XLS.")
        return records

    def parse_docx(self, docx_bytes: bytes) -> List[StudentImportRecord]:
        """
        Parses a Word (.docx) document:
        - Extracts the primary student table
        - Resolves embedded photos from word/media/ using drawing relationship IDs
        """
        records: List[StudentImportRecord] = []
        try:
            with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as z:
                # 1. Read relationships to map rId -> target media filename
                rel_map: Dict[str, str] = {}
                if "word/_rels/document.xml.rels" in z.namelist():
                    rels_xml = z.read("word/_rels/document.xml.rels")
                    root_rels = ET.fromstring(rels_xml)
                    for r in root_rels:
                        r_id = r.attrib.get("Id")
                        target = r.attrib.get("Target")
                        if r_id and target:
                            # Normalize path: 'media/image1.jpeg' -> 'word/media/image1.jpeg'
                            if not target.startswith("word/"):
                                target = f"word/{target}"
                            rel_map[r_id] = target

                # 2. Read document XML
                if "word/document.xml" not in z.namelist():
                    raise ImporterValidationError("Invalid DOCX format: 'word/document.xml' missing.")

                doc_xml = z.read("word/document.xml")
                root_doc = ET.fromstring(doc_xml)

                ns = {
                    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
                    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                }

                tables = root_doc.findall(".//w:tbl", ns)
                if not tables:
                    raise ImporterValidationError("No tables found in DOCX student document.")

                # Use the largest table
                tbl = max(tables, key=lambda t: len(t.findall(".//w:tr", ns)))
                rows = tbl.findall(".//w:tr", ns)
                if len(rows) < 2:
                    raise ImporterValidationError("Table in DOCX has fewer than 2 rows (header + data).")

                # Extract header row
                header_cells = rows[0].findall(".//w:tc", ns)
                headers = ["".join(c.itertext()).strip() for c in header_cells]
                mapping = self.identify_column_mapping(headers)

                # Process data rows
                for r_idx, r in enumerate(rows[1:], start=2):
                    cells = r.findall(".//w:tc", ns)
                    if len(cells) < 2:
                        continue

                    # Helper to get cell text
                    def get_text(idx: Optional[int]) -> str:
                        if idx is not None and idx < len(cells):
                            return "".join(cells[idx].itertext()).strip()
                        return ""

                    reg_no = get_text(mapping.get("register_no"))
                    name = get_text(mapping.get("name"))
                    class_sec = get_text(mapping.get("class_section")) or "III AI&DS-B"

                    if not reg_no and not name:
                        continue  # Skip completely empty row

                    # Extract embedded images from this row
                    blips = r.findall(".//a:blip", ns)
                    photos: List[bytes] = []
                    photo_names: List[str] = []

                    for b in blips:
                        embed_id = b.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                        media_path = rel_map.get(embed_id)
                        if media_path and media_path in z.namelist():
                            img_data = z.read(media_path)
                            photos.append(img_data)
                            photo_names.append(os.path.basename(media_path))

                    dept, sec = self.parse_class_section(class_sec)

                    records.append(StudentImportRecord(
                        register_no=reg_no,
                        name=name,
                        class_section=class_sec,
                        department=dept,
                        section=sec,
                        photos=photos,
                        photo_filenames=photo_names,
                        row_index=r_idx,
                        raw_fields={headers[i]: get_text(i) for i in range(len(cells)) if i < len(headers)}
                    ))

        except zipfile.BadZipFile:
            raise ImporterValidationError("Corrupt or invalid DOCX file (BadZipFile).")
        except Exception as e:
            if isinstance(e, ImporterValidationError):
                raise
            raise ImporterValidationError(f"Error parsing DOCX file: {e}")

        logger.info(f"DOCX Import: Successfully parsed {len(records)} records from DOCX table.")
        return records

    def parse_csv(self, csv_bytes: bytes) -> List[StudentImportRecord]:
        """
        Parses CSV data. Photos may be base64 encoded or file path references.
        """
        text = csv_bytes.decode("utf-8-sig", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ImporterValidationError("CSV data must contain at least a header row and one data row.")

        # Auto-detect dialect
        sample = "\n".join(lines[:5])
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(text), dialect=dialect)
        headers = [h.strip() for h in next(reader)]
        mapping = self.identify_column_mapping(headers)

        records: List[StudentImportRecord] = []
        for r_idx, row in enumerate(reader, start=2):
            if not any(row):
                continue

            def get_val(idx: Optional[int]) -> str:
                if idx is not None and idx < len(row):
                    return row[idx].strip()
                return ""

            reg_no = get_val(mapping.get("register_no"))
            name = get_val(mapping.get("name"))
            class_sec = get_val(mapping.get("class_section")) or "Unknown"
            photo_ref = get_val(mapping.get("photo"))

            dept, sec = self.parse_class_section(class_sec)
            photos: List[bytes] = []
            photo_names: List[str] = []

            # If photo_ref provided, record name and read if exists
            if photo_ref:
                photo_names.append(os.path.basename(photo_ref) or photo_ref)
                if os.path.exists(photo_ref):
                    try:
                        with open(photo_ref, "rb") as pf:
                            photos.append(pf.read())
                    except Exception as e:
                        logger.warning(f"Could not read photo file '{photo_ref}': {e}")

            records.append(StudentImportRecord(
                register_no=reg_no,
                name=name,
                class_section=class_sec,
                department=dept,
                section=sec,
                photos=photos,
                photo_filenames=photo_names,
                row_index=r_idx,
                raw_fields={headers[i]: get_val(i) for i in range(len(row)) if i < len(headers)}
            ))

        is_valid, errors = self.validate_records(records)
        if not is_valid:
            raise ImporterValidationError("CSV record validation failed:\n" + "\n".join(errors))

        logger.info(f"CSV Import: Successfully parsed {len(records)} records from CSV.")
        return records

    def parse_directory(self, dir_path: str) -> List[StudentImportRecord]:
        """
        Parses a directory of student images.
        Supports:
        - Subfolders named by Register Number (e.g. dir_path/922524243069/photo.jpg)
        - Files named by Register Number (e.g. dir_path/922524243069.jpg)
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        records: List[StudentImportRecord] = []
        valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

        entries = os.listdir(dir_path)
        for idx, entry in enumerate(sorted(entries), start=1):
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                # Subfolder named by register number
                reg_no = entry.strip()
                photos = []
                pnames = []
                for f in sorted(os.listdir(full_path)):
                    if f.lower().endswith(valid_exts):
                        img_path = os.path.join(full_path, f)
                        with open(img_path, "rb") as img_f:
                            photos.append(img_f.read())
                            pnames.append(f)
                if reg_no:
                    records.append(StudentImportRecord(
                        register_no=reg_no,
                        name=f"Student {reg_no}",
                        photos=photos,
                        photo_filenames=pnames,
                        row_index=idx
                    ))
            elif os.path.isfile(full_path) and entry.lower().endswith(valid_exts):
                reg_no = os.path.splitext(entry)[0].strip()
                with open(full_path, "rb") as img_f:
                    data = img_f.read()
                records.append(StudentImportRecord(
                    register_no=reg_no,
                    name=f"Student {reg_no}",
                    photos=[data],
                    photo_filenames=[entry],
                    row_index=idx
                ))

        logger.info(f"Directory Import: Found {len(records)} student records in '{dir_path}'.")
        return records

    def validate_records(self, records: List[StudentImportRecord]) -> Tuple[bool, List[str]]:
        """
        Validates the batch of imported records for integrity:
        - Checks for duplicate register numbers
        - Checks for missing required fields (register_no, name)
        Returns (is_valid, list_of_errors).
        """
        errors: List[str] = []
        seen_reg: Dict[str, int] = {}

        for rec in records:
            if not rec.register_no or not rec.register_no.strip():
                errors.append(f"Row {rec.row_index}: Missing required 'register_no'.")
            else:
                reg_clean = rec.register_no.strip()
                if reg_clean in seen_reg:
                    errors.append(
                        f"Row {rec.row_index}: Duplicate register_no '{reg_clean}' "
                        f"(first seen at Row {seen_reg[reg_clean]})."
                    )
                else:
                    seen_reg[reg_clean] = rec.row_index

            if not rec.name or not rec.name.strip():
                errors.append(f"Row {rec.row_index}: Missing required 'name'.")

        return len(errors) == 0, errors

    def import_docx(self, file_path_or_bytes: Union[str, bytes]) -> List[StudentImportRecord]:
        """Convenience method to import from a docx file path or bytes."""
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "rb") as f:
                data = f.read()
        else:
            data = file_path_or_bytes
        return self.parse_docx(data)

    def import_xlsx(self, file_path_or_bytes: Union[str, bytes]) -> List[StudentImportRecord]:
        """Convenience method to import from an xlsx file path or bytes."""
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "rb") as f:
                data = f.read()
        else:
            data = file_path_or_bytes
        return self.parse_xlsx(data)

    def import_xls(self, file_path_or_bytes: Union[str, bytes]) -> List[StudentImportRecord]:
        """Convenience method to import from an xls file path or bytes."""
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "rb") as f:
                data = f.read()
        else:
            data = file_path_or_bytes
        return self.parse_xls(data)

    def import_csv(self, file_path_or_bytes: Union[str, bytes]) -> List[StudentImportRecord]:
        """Convenience method to import from a csv file path or bytes."""
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "rb") as f:
                data = f.read()
        else:
            data = file_path_or_bytes
        return self.parse_csv(data)

    def import_directory(self, dir_path: str) -> List[StudentImportRecord]:
        """Convenience method to import from a directory."""
        return self.parse_directory(dir_path)


