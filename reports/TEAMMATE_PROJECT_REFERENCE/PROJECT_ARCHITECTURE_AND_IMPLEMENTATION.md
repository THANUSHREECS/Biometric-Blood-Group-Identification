# System Architecture & Technical Implementation Reference
**Project:** HemoPrint AI — Biometric Blood Group Detection Research System  
**Audience:** Software Engineers, Technical Writers, and Project Demonstrators  
**Scope:** Complete Hardware, Software, Backend, Frontend, and Database Engineering Reference  
**Verification:** All endpoints, schemas, and configurations verified directly from source code and integration test suites.

---

## 1. System Architecture Overview

HemoPrint AI is constructed as a decoupled client-server architecture consisting of a **React + TypeScript single-page application (SPA)** on the client side, a **Python Flask application (`app.py`)** on the server side, a **hardware integration layer for the Mantra MFS100 optical fingerprint scanner**, and an **embedded SQLite relational database (`data/blood_group.db`)**.

```mermaid
graph TD
    subgraph Client Layer [Frontend - React 18 + TypeScript + Vite]
        UI["App.tsx / Single Page Interface"]
        Views["Views: Dashboard, Register, Capture, Quality, Analysis, Results, History, ModelInfo"]
        Client["apiClient.ts (Axios REST Client / 120s Timeout)"]
        UI --> Views
        Views --> Client
    end

    subgraph Server Layer [Backend - Python 3.10 + Flask app.py]
        API["REST API Router (/api/*)"]
        ScannerMgr["MFS100Scanner Interface (Native SDK / Mock Fallback)"]
        QualityEng["Image Quality Analyzer (Coverage, Coherence, Contrast)"]
        PrepEng["C5 Preprocessing Pipeline (Alpha Comp, Platen Crop, Normalization)"]
        Predictor["BloodGroupPredictor (Centralized Model Loader / Multi-Fold Ensemble)"]
        DBHandler["SQLite Persistence Layer (src/database.py)"]
        ReportEng["ReportLab PDF Generator (reports/*.pdf)"]

        API --> ScannerMgr
        API --> QualityEng
        API --> PrepEng
        API --> Predictor
        API --> DBHandler
        API --> ReportEng
    end

    subgraph Data & Hardware Layer
        Hardware["Mantra MFS100 Optical Fingerprint Sensor (500 DPI, USB)"]
        Database[("SQLite Database: data/blood_group.db")]
        Checkpoints[("Model Weights: models/ & experiments/")]
        FileStorage[("Biometric Scan Storage: scratch/research_*/ & data/captured/")]

        ScannerMgr --> Hardware
        DBHandler --> Database
        Predictor --> Checkpoints
        API --> FileStorage
    end

    Client -->|HTTP / JSON (Port 5000)| API
```

---

## 2. Hardware Architecture & Sensor Integration

### Optical Sensor Specifications
* **Sensor Model:** Mantra MFS100 Optical Fingerprint Scanner
* **Optical Sensing Technology:** Total Internal Reflection (TIR) optical prism illuminated by red LEDs.
* **Spatial Resolution:** **500 DPI (dots per inch)**
* **Image Capture Dimensions:** **312 pixels (width) $	imes$ 512 pixels (height)** (raw sensor frame)
* **Pixel Bit Depth:** **8-bit grayscale** (256 discrete intensity levels: 0 = ridge contact, 255 = non-contact valley / air)
* **Active Sensing Surface Area:** $16\text{ mm} \times 24\text{ mm}$
* **Interface:** USB 2.0 High-Speed
* **Native Driver & SDK:** `MFS100Dll.dll` located at `C:\Program Files\Mantra\MFS100\Driver\MFS100Test\MFS100Dll.dll`.

### Driver Implementation ([`src/mfs100/scanner.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/mfs100/scanner.py))
The hardware interface is wrapped in Python using the `ctypes` foreign function interface:
* **Hardware Detection:** Dynamically queries `MFS100Dll.dll`. If the DLL or physical USB sensor is absent, the driver automatically falls back to an internal **Mock Mode**.
* **Mock Mode Capabilities:** Emulates real platen capture by generating realistic 312x512 synthetic fingerprint images or drawing from `data/sample/`, allowing full-pipeline frontend and backend testing on developer laptops without hardware attached.
* **Lift-and-Reposition Enforcement:** The driver requires lifting the finger between consecutive captures to ensure true independent sensor impressions.

---

## 3. Backend Architecture & REST API

The operational backend is implemented in [`app.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/app.py) using Flask and Flask-CORS.

### Verified REST API Endpoints

| Method | Endpoint | Description | Verification Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/status` | Returns system health, active model metadata, loaded fold count, and database stats. | Verified in `test_01_system_status_and_stats` |
| `GET` | `/api/scanner/status` | Returns whether the Mantra MFS100 is connected, initialized, and whether mock mode is active. | Verified in `app.py:85-110` |
| `POST` | `/api/scanner/connect` | Triggers hardware device initialization via `MFS100Init()`. | Verified in `app.py:112-135` |
| `POST` | `/api/scanner/disconnect` | Releases optical platen handle via `MFS100Uninit()`. | Verified in `app.py:137-150` |
| `POST` | `/api/scanner/mode` | Toggles mock hardware emulation mode (`{"mock": true/false}`). | Verified in `app.py:152-168` |
| `POST` | `/api/research/session` | Creates a new research participant session with mandatory consent validation (`consent_given: true`). Assigns pseudonymous `P-xxxx` token. | Verified in `test_02_registration_validation` |
| `GET` | `/api/research/session/<id>` | Retrieves session status, participant pseudonym, captured impressions, and inference results. | Verified in `test_03_three_capture_workflow` |
| `POST` | `/api/research/capture` | Captures or uploads impression $k \in \{1, 2, 3\}$ of `L_THUMB`. Computes quality metrics and stores image. Rejects $k > 3$. | Verified in `test_03_three_capture_workflow` |
| `POST` | `/api/research/infer` | Executes multi-scan feature fusion inference across 5 cross-validation folds. Requires exactly 3 captures. Computes class probabilities and soft/majority consensus. | Verified in `test_03_three_capture_workflow` |
| `GET` | `/api/research/sessions` | Returns history list of all participant research sessions from database. | Verified in `test_04_session_history_list` |
| `GET` | `/api/research/stats` | Aggregates database session metrics: total sessions, valid captures, and completed inferences. | Verified in `app.py:530-555` |
| `POST` | `/api/upload` | Legacy single-image upload and ad-hoc inference route. | Verified in `app.py:220-275` |
| `GET` | `/api/history` | Returns legacy single-scan prediction audit records. | Verified in `app.py:310-330` |
| `GET` | `/api/report/<id>` | Lazily generates and downloads a clinical-style research audit PDF report via ReportLab. | Verified in `app.py:380-420` |

---

## 4. Frontend Architecture & Component Hierarchy

The frontend is built with **React 18, TypeScript, and Vite** in [`live_app/frontend/`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/live_app/frontend).

### Component Tree & Screen Routing
Routing is managed via declarative screen state (`currentScreen`) in [`live_app/frontend/src/App.tsx`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/live_app/frontend/src/App.tsx):

```
App.tsx
├── Navbar.tsx (Navigation bar, system status indicator, hardware connection badge)
├── ResearchDisclaimer.tsx (Persistent clinical non-validation notice)
└── Views (Conditional Rendering based on ScreenId):
    ├── DashboardView.tsx (System overview, quick stats, quick session actions)
    ├── ParticipantRegistrationView.tsx (Consent checkbox, pseudonym assignment, optional self-reported label)
    ├── ScannerConnectionView.tsx (Hardware connection controls, mock toggle, platen preview)
    ├── FingerprintCaptureView.tsx (Interactive 3-scan capture stepper, lift-and-reposition feedback)
    ├── QualityReviewView.tsx (Impression quality breakdown: coverage %, coherence, contrast)
    ├── ExperimentalAnalysisView.tsx (Real-time preprocessing inspection: C5 pipeline transforms)
    ├── ResearchResultsView.tsx (Predicted blood group, 4-class probability bar chart, consensus details)
    ├── SessionHistoryView.tsx (Table of past sessions, completion status, PDF download links)
    └── ModelInfoView.tsx (Architecture benchmark comparisons, fold metrics, scientific evidence)
```

### Client Networking (`live_app/frontend/src/api/client.ts`)
* Uses **Axios** configured with a default base URL `http://localhost:5000/api`.
* **Extended Timeout:** Configured with `timeout: 120000` (120 seconds) to prevent premature network timeouts during multi-fold deep CNN inference or intensive feature extraction.

---

## 5. Image Preprocessing & Quality Assurance Pipeline

The core preprocessing logic resides in [`src/preprocessing.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/preprocessing.py) and [`src/image_quality.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/image_quality.py).

### The C5 Preprocessing Sequence
For every acquired image, the C5 pipeline performs four strictly ordered steps:
1. **Alpha Channel Handling (`composite_alpha_on_white`):** If an image contains an RGBA transparency channel (common in web canvas previews), it composites the alpha channel over an opaque white background (`RGB(255, 255, 255)`), preventing transparent border artifacts from becoming black borders.
2. **Foreground Platen Cropping (`crop_mfs100_foreground`):** Detects the central platen region and crops away non-ridge dark boundary borders that do not contain biometric friction ridges.
3. **Ridge Contrast Enhancement (`enhance_mfs100_ridge_contrast`):** Normalizes dynamic range to stretch low-contrast ridges.
4. **Target Resizing (`resize_to_target`):** Resizes images using bilinear interpolation to `(112, 112)` for standard models, or `(224, 224)` for high-resolution evaluation (EXP-07).

### Image Quality Metrics ([`src/image_quality.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/image_quality.py))
Every capture is audited for three physical quality indices:
* **Foreground Coverage (`fg_coverage`):** Fraction of the optical frame containing active friction ridges (threshold $> 0.35$).
* **Ridge Coherence (`ridge_coherence`):** Directional gradient consistency across Sobel orientation fields (evaluates ridge clarity vs smudging).
* **Ridge Contrast (`ridge_contrast`):** Standard deviation of pixel intensities within the segmented ridge mask (evaluates dry vs wet finger impressions).

---

## 6. Database Architecture (`data/blood_group.db`)

The system uses an embedded SQLite database ([`data/blood_group.db`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/data/blood_group.db)) managed through [`src/database.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/database.py).

### Database Schema Definition

```sql
-- Table: predictions (Legacy single-scan inference records)
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    predicted_blood_group TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities_json TEXT NOT NULL,
    image_quality_score REAL,
    finger_designation TEXT DEFAULT 'UNKNOWN',
    session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Table: research_sessions (Participant study tracking)
CREATE TABLE research_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    participant_id TEXT NOT NULL,
    consent_given BOOLEAN NOT NULL DEFAULT 0,
    self_reported_group TEXT,
    label_source TEXT DEFAULT 'self_reported',
    status TEXT DEFAULT 'CREATED', -- 'CREATED', 'CAPTURING', 'COMPLETED', 'FAILED'
    predicted_group TEXT,
    confidence REAL,
    probabilities_json TEXT,
    consensus_details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    notes TEXT
);

-- Table: research_captures (Physical biometric scans linked to sessions)
CREATE TABLE research_captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    finger TEXT NOT NULL, -- e.g., 'L_THUMB'
    impression_number INTEGER NOT NULL, -- 1, 2, or 3
    file_path TEXT NOT NULL,
    quality_score REAL,
    quality_metrics_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES research_sessions(session_id)
);
```

---

## 7. Operational Workflow (Step-by-Step Execution)

```
[Start Session]
      │
      ▼
1. Participant Registration: Enters consent checkbox, optional self-reported blood group.
      │ System assigns pseudonymous ID (e.g. P-0081) and creates SQLite session record.
      ▼
2. Scanner Connection Check: Verifies Mantra MFS100 device handle or initializes mock driver.
      │
      ▼
3. Three-Scan Capture Protocol:
      ├── Impression 1: Place left thumb -> Optical capture -> Platen quality audit -> Saved.
      ├── (Prompt: "Lift and reposition finger on platen")
      ├── Impression 2: Reposition left thumb -> Optical capture -> Quality audit -> Saved.
      ├── (Prompt: "Lift and reposition finger on platen")
      └── Impression 3: Reposition left thumb -> Optical capture -> Quality audit -> Saved.
      │
      ▼
4. Preprocessing & Quality Aggregation:
      │ Applies C5 pipeline to all 3 scans; computes average quality score.
      ▼
5. Multi-Fold Ensemble Inference:
      │ Feeds scans into 5 cross-validation fold models.
      │ Aggregates fold probability distributions across all impressions.
      ▼
6. Consensus Determination:
      │ Soft Consensus: argmax of arithmetic mean probability vector.
      │ Majority Consensus: Class voted by >= 2 of 3 impressions.
      ▼
7. Results & Audit Display:
      │ Displays predicted blood group with confidence score.
      │ Renders mandatory clinical disclaimer ("NOT FOR MEDICAL USE").
      ▼
8. Persistence & PDF Generation:
      │ Updates research_sessions record in blood_group.db to 'COMPLETED'.
      │ Generates lazy clinical audit PDF report upon user download.
```

---

## 8. Feature Implementation Verification Status

| System Feature | Implementation State | Evidence / Source File |
| :--- | :---: | :--- |
| **Research Dashboard** | **Fully Implemented** | `live_app/frontend/src/views/DashboardView.tsx` |
| **Pseudonymous Registration** | **Fully Implemented** | `tests/test_research_integration.py:31` |
| **Mandatory Consent Gating** | **Fully Implemented** | Rejects registration if `consent_given == false` |
| **Mantra MFS100 Hardware Driver** | **Fully Implemented** | `src/mfs100/scanner.py` via `MFS100Dll.dll` |
| **Mock Scanner Emulation** | **Fully Implemented** | Automatic fallback; toggleable via `/api/scanner/mode` |
| **Three-Scan Protocol** | **Fully Implemented** | Strictly enforces 3 impressions; rejects 4th |
| **Real-Time Quality Audit** | **Fully Implemented** | `src/image_quality.py` (coverage, coherence, contrast) |
| **C5 Preprocessing Pipeline** | **Fully Implemented** | `src/preprocessing.py` (alpha composite, crop, enhance) |
| **Multi-Fold Inference Engine** | **Fully Implemented** | `src/predictor.py` (5-fold ensemble) |
| **Session History Table** | **Fully Implemented** | `live_app/frontend/src/views/SessionHistoryView.tsx` |
| **PDF Report Generation** | **Fully Implemented** | `src/report_generator.py` via ReportLab |
| **Model Info & Evidence View** | **Fully Implemented** | `live_app/frontend/src/views/ModelInfoView.tsx` |
| **Automated NFIQ-2 Scoring** | **Partial / Heuristic** | Uses custom contrast/coherence; native NIST NFIQ-2 binary is not linked |
| **Automated Laboratory Serology Sync**| **Not Implemented** | Ground-truth labels must be entered manually |

---

## 9. Startup & Build Verification Commands

### Running the Production Backend (Flask)
```powershell
# Navigate to project root
cd "C:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"

# Activate Python virtual environment (if used) or run directly with system Python
python app.py
# Server starts on http://127.0.0.1:5000
```

### Running the Frontend Development Server (Vite)
```powershell
cd "C:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project\live_app\frontend"
npm run dev
# Vite dev server starts on http://localhost:5173
```

### Building Frontend for Production
```powershell
cd "C:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project\live_app\frontend"
npm run build
# Compiles into dist/ in ~1.5 - 1.8 seconds (Verified: 1,952 modules)
```

### Running the Integration Test Suite
```powershell
cd "C:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
python -m pytest tests/test_research_integration.py -v
# Verified: 4 passed in ~33-47s (Exit Code: 0)
```
