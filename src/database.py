"""
src/database.py
SQLite database manager for Biometric Blood Group Identification.
Stores prediction audit history without sensitive PII.
"""

import os
import sqlite3
import datetime
from src.config import DATABASE_PATH

def get_connection(db_path=DATABASE_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DATABASE_PATH):
    """Initializes the SQLite database and predictions table.
    Migration-safe: adds finger_used column if it does not already exist.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_type TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_group TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            notes TEXT,
            finger_used TEXT
        )
    """)
    # Migration: add finger_used column to pre-existing databases that lack it
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(predictions)").fetchall()]
    if "finger_used" not in existing_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN finger_used TEXT")
    conn.commit()
    conn.close()

def insert_prediction(
    input_type, image_path, predicted_group, confidence,
    status="NORMAL", notes="", finger_used=None, db_path=DATABASE_PATH
):
    """
    Inserts a new prediction record into the database.
    finger_used: optional acquisition finger identifier (e.g. 'LEFT_THUMB' for real MFS100 captures).
                 None for uploaded images — the scanner does not verify finger identity.
    Returns the integer id of the inserted record.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO predictions (timestamp, input_type, image_path, predicted_group, confidence, status, notes, finger_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, input_type, image_path, predicted_group, float(confidence), status, notes, finger_used))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id

def get_prediction_by_id(record_id, db_path=DATABASE_PATH):
    """Fetches a single prediction by its ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_predictions(limit=100, db_path=DATABASE_PATH):
    """Fetches recent predictions ordered by latest first."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
