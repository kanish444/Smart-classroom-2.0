import urllib.request
import os

candidates = [
    "https://vsbec.edu.in/wp-content/uploads/2026/06/VSB-Karur-1.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/06/cropped-VSB-Karur-1.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/06/cropped-VSB-Karur-1-512x512.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/02/OBJECT.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/02/Layer_1-1.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/04/Screenshot-2026-04-04-124353.png",
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for url in candidates:
    fname = os.path.basename(url)
    out_path = os.path.join("dashboard/static/images", fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"SUCCESS: {fname} ({len(data)} bytes)")
    except Exception as e:
        print(f"FAILED {fname}: {e}")
