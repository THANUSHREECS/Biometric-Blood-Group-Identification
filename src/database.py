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

    # Research Protocol Tables (Safe additions)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_sessions (
            session_id TEXT PRIMARY KEY,
            participant_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            consent_given INTEGER NOT NULL,
            self_reported_group TEXT,
            label_source TEXT,
            verification_status TEXT,
            notes TEXT,
            status TEXT NOT NULL,
            predicted_group TEXT,
            confidence REAL,
            canonical_probabilities TEXT,
            model_name TEXT,
            model_version TEXT,
            inference_completed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_captures (
            capture_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            impression_number INTEGER NOT NULL,
            finger_designation TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            quality_score INTEGER,
            nfiq_score INTEGER,
            fg_coverage REAL,
            ridge_coherence REAL,
            ridge_contrast REAL,
            unusable_proportion REAL,
            is_acceptable INTEGER NOT NULL,
            rejection_reason TEXT,
            FOREIGN KEY(session_id) REFERENCES research_sessions(session_id)
        )
    """)

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

# =========================================================================
# Research Session & Multi-Capture Management
# =========================================================================

def create_research_session(
    session_id, participant_id, consent_given=True,
    self_reported_group=None, label_source="self_reported",
    verification_status="unverified", notes="", db_path=DATABASE_PATH
):
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO research_sessions (
            session_id, participant_id, created_at, consent_given,
            self_reported_group, label_source, verification_status,
            notes, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, participant_id, ts, 1 if consent_given else 0,
        self_reported_group, label_source, verification_status,
        notes, "CREATED"
    ))
    conn.commit()
    conn.close()
    return session_id

def insert_research_capture(
    capture_id, session_id, impression_number, finger_designation,
    image_filename, width, height, quality_score, nfiq_score,
    fg_coverage=None, ridge_coherence=None, ridge_contrast=None,
    unusable_proportion=None, is_acceptable=1, rejection_reason=None,
    db_path=DATABASE_PATH
):
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO research_captures (
            capture_id, session_id, impression_number, finger_designation,
            image_filename, captured_at, width, height, quality_score,
            nfiq_score, fg_coverage, ridge_coherence, ridge_contrast,
            unusable_proportion, is_acceptable, rejection_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        capture_id, session_id, impression_number, finger_designation,
        image_filename, ts, width, height, quality_score,
        nfiq_score, fg_coverage, ridge_coherence, ridge_contrast,
        unusable_proportion, 1 if is_acceptable else 0, rejection_reason
    ))
    cursor.execute("UPDATE research_sessions SET status = 'CAPTURING' WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return capture_id

def update_session_inference(
    session_id, predicted_group, confidence,
    canonical_probabilities, model_name, model_version="1.0",
    db_path=DATABASE_PATH
):
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    import json
    probs_json = json.dumps(canonical_probabilities) if isinstance(canonical_probabilities, dict) else str(canonical_probabilities)
    cursor.execute("""
        UPDATE research_sessions
        SET status = 'COMPLETED',
            predicted_group = ?,
            confidence = ?,
            canonical_probabilities = ?,
            model_name = ?,
            model_version = ?,
            inference_completed_at = ?
        WHERE session_id = ?
    """, (predicted_group, float(confidence), probs_json, model_name, model_version, ts, session_id))
    conn.commit()
    conn.close()

def get_research_session(session_id, db_path=DATABASE_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM research_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    session_dict = dict(row)
    cursor.execute("""
        SELECT * FROM research_captures
        WHERE session_id = ?
        ORDER BY impression_number ASC
    """, (session_id,))
    captures = [dict(c) for c in cursor.fetchall()]
    session_dict["captures"] = captures
    conn.close()
    return session_dict

def get_all_research_sessions(limit=50, db_path=DATABASE_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM research_sessions ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    for r in rows:
        cursor.execute("SELECT COUNT(*) FROM research_captures WHERE session_id = ?", (r["session_id"],))
        r["capture_count"] = cursor.fetchone()[0]
    conn.close()
    return rows

def get_research_counts(db_path=DATABASE_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT participant_id) FROM research_sessions")
    unique_participants = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM research_sessions")
    total_sessions = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM research_captures WHERE is_acceptable = 1")
    valid_captures = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions")
    legacy_predictions = cursor.fetchone()[0]
    conn.close()
    return {
        "unique_participants": unique_participants,
        "total_sessions": total_sessions,
        "valid_captures": valid_captures,
        "legacy_predictions": legacy_predictions
    }

