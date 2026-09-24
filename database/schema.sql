-- Phase 2 Foundation Schema
-- SmartClass Vision AI

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    student_name TEXT NOT NULL,
    department TEXT NOT NULL,
    section TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    vector_blob BLOB NOT NULL,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Note: No real student data should be inserted in Phase 2.
