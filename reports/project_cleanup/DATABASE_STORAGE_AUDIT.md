# Database Storage & Compaction Audit
**Project:** HemoPrint AI — Biometric Blood Group Detection Research System  
**Database File:** [`data/blood_group.db`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/data/blood_group.db)  
**Audit Mode:** Read-Only (`mode=ro`, 0 modifications)

---

## 1. Storage Baseline & Physical Geometry

| Metric | Measured Value | Analysis |
| :--- | :--- | :--- |
| **Total File Size** | **`208,896 bytes`** (`0.1992 MB`) | Extremely compact operational database |
| **Page Size** | `4096` bytes | Standard SQLite page size (4 KB) |
| **Total Pages** | `51` | Exactly 51 pages on disk |
| **Freelist (Unused) Pages** | **`0`** | **Zero free pages** |
| **Unused Free Space** | **`0 bytes`** (`0.0 MB`, **`0.0%`**) | **Zero internal storage fragmentation** |

---

## 2. Table Inventory & Record Counts

| Table Name | Row Count | Primary Purpose |
| :--- | :---: | :--- |
| `predictions` | **898** | Longitudinal inference audit logs, confidence distributions, and timestamped outcomes |
| `research_captures` | **26** | Linked biometric sensor acquisition records, quality metrics, and scan references |
| `research_sessions` | **18** | Participant research session tokens, study demographics, and protocol metadata |

---

## 3. Database Schema

The database defines three operational tables:

```sql
-- Table: predictions
CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_type TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_group TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            notes TEXT
        , finger_used TEXT);

-- Table: research_captures
CREATE TABLE research_captures (
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
        );

-- Table: research_sessions
CREATE TABLE research_sessions (
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
        );

```

---

## 4. Backups and Secondary Copies
* **Database Backup Search:** Scanned entire project tree for `*.db`, `*.sqlite`, `*.db-shm`, `*.db-wal`, `*blood_group*.db*`.
* **Identified Backups:** None found. `data/blood_group.db` is the single canonical source of truth for participant and prediction history.

---

## 5. Risk Assessment: Database Compaction & VACUUM

| Evaluation Criteria | Assessment |
| :--- | :--- |
| **Potential Space Savings** | **`0.00 MB` (0 bytes)**. With 0 free pages in the SQLite freelist, a `VACUUM` operation will yield zero storage reduction. |
| **Concurrency & Locking Risk** | Running `VACUUM` requires an exclusive write lock. If the live Flask backend or scanner process is running, lock conflicts or schema errors can occur. |
| **Filesystem & OneDrive Hazard** | `data/` is synced by Microsoft OneDrive. Triggering file replacement via `VACUUM` on an active cloud-synced folder can trigger synchronization conflicts, transient locks (`WinError 5`), or cloud-reversion bugs. |
| **Scientific Integrity Hazard** | Modifying or compacting the database risks invalidating rowids or transaction audit logs referenced by experimental scripts. |
| **Final Recommendation** | **DO NOT VACUUM OR MODIFY.** The database is already optimal (0% waste, 0.20 MB total) and poses zero storage burden. |
