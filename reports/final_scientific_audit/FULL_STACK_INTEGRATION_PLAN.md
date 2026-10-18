# Implementation Plan: HemoPrint AI Professional Frontend & Full-Stack Integration

**Date:** 2026-10-18  
**Scope:** Frontend (React + TypeScript), Flask Backend (`app.py`), MFS100 Scanner Bridge (`src/mfs100/`), Research Inference Pipeline (`src/predictor.py`), and Database Storage (`src/database.py`).

---

## 1. Architectural Findings & Component Inventory

1. **Frontend Architecture:**
   - Location: `live_app/frontend` (Vite, React 19, TypeScript, Axios, Lucide-React).
   - Theme & Aesthetic: Medical-research dashboard, deep navy navigation (`#0f172a`), clean slate/neutral surfaces (`#f8fafc`), teal/cyan primary accents (`#0d9488`), crisp typography, accessible contrast, no deceptive green "verified" medical indicators.
   - 9 Required Screens:
     1. Dashboard (`DashboardView`)
     2. Participant Registration (`ParticipantRegistrationView`)
     3. Scanner Connection (`ScannerConnectionView`)
     4. Fingerprint Capture (`FingerprintCaptureView` - 3-step Left Thumb protocol)
     5. Image Quality Review (`QualityReviewView`)
     6. Experimental Analysis (`ExperimentalAnalysisView`)
     7. Research Results (`ResearchResultsView`)
     8. Participant Session History (`SessionHistoryView`)
     9. Model and System Information (`ModelInfoView`)

2. **Backend API (`app.py`):**
   - Framework: Flask (port 5000), CORS enabled.
   - Integrates existing scanner adapter `MFS100Scanner` (`src/mfs100/scanner.py`) which communicates with the 32-bit bridge at `http://127.0.0.1:5100`.
   - Endpoints provided / enhanced:
     - `/api/status`: System, scanner, model health and stats.
     - `/api/scanner/status`, `/api/scanner/connect`, `/api/scanner/disconnect`, `/api/scanner/mode`: Hardware & mock controls.
     - `/api/research/session/create`: Register participant session with pseudonymous ID, consent, optional self-reported blood group with explicit verification status flag.
     - `/api/research/session/<session_id>`: Get current session details, captures, and quality metrics.
     - `/api/research/capture`: Capture a single left thumb impression (1, 2, or 3) for the active session, compute MFS100 quality + NFIQ + EXP-10 ridge metrics, and save raw BMP.
     - `/api/research/infer`: Perform 3-scan multi-scan inference using the verified EXP-06C Config A linear projection model.
     - `/api/research/sessions`: Query past research sessions and participant records.
     - `/api/image/<filename>`: Safely serve captured images without filesystem leakage.

3. **Database Layer (`src/database.py`):**
   - Safe SQLite migration on `data/blood_group.db` adding `research_sessions` and `research_captures` tables while leaving existing 898 historical `predictions` records 100% untouched.

4. **Model Architecture & Checkpoint Compatibility:**
   - Selected Model: **EXP-06C Config A (Linear Projection Reference)**.
   - Architecture: Frozen DenseNet-121 backbone, 3-scan feature concatenation (3072-d), `Dense(512, relu)` -> `Dropout(0.5)` -> `Dense(4, softmax, name="clf")` and `Dense(128, None, name="emb")`.
   - Verified Checkpoints: `experiments/exp06c/checkpoints/config_a/fold_[1-5]_best.weights.h5`.
   - Preprocessing: `transform_c5` (alpha on white -> foreground crop -> contrast stretch -> resize to 112x112).
   - Class Mapping: Native `[A+, B+, AB+, O+]` remapped to Canonical `[A+, AB+, B+, O+]` via indices `[0, 2, 1, 3]`.

5. **Scanner Capture & Quality Protocol:**
   - Strict 3-step sequential Left Thumb capture workflow: Prompt -> Place -> Capture -> Lift & Reposition -> Repeat.
   - Image quality audit: MFS100 SDK Quality (0-100), NFIQ (1-5), and EXP-10 metrics (foreground coverage, ridge coherence, local contrast, unusable region ratio).

---

## 2. Implementation Steps

1. **Database Safe Migration:** Update `src/database.py` to create `research_sessions` and `research_captures` tables if not exists.
2. **Inference Engine Update:** Update `src/predictor.py` to support EXP-06C Config A ensemble with verified canonical class remapping.
3. **Backend API Extensions:** Add `/api/research/...` endpoints in `app.py`.
4. **Frontend Architecture & Components:**
   - Create typed API client `live_app/frontend/src/api/client.ts`.
   - Build 9 modular view components under `live_app/frontend/src/components/`.
   - Implement navigation bar with real scanner status pill and persistent clinical disclaimer.
   - Assemble in `live_app/frontend/src/App.tsx`.
5. **Testing & Verification:**
   - Verify frontend production build (`npm run build`).
   - Run backend integration and unit tests.
   - Generate full verification report.
