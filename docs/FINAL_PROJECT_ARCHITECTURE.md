Ã¯Â»Â¿# FINAL PROJECT ARCHITECTURE
## Biometric Blood Group Identification System

**Document Type:** System Architecture Reference
**Project:** B.E. Final Year Major Project Ã¢â‚¬â€ Computer Science and Engineering (AI and ML)
**Date:** September 2026
**Model Status:** FROZEN Ã¢â‚¬â€ Champion Ensemble V2 (85.41% locked test accuracy)

---

## 1. System Overview

The Biometric Blood Group Identification System is an end-to-end AI-based research prototype that classifies human ABO/Rh blood groups from optical fingerprint impressions.

```
Physical Fingerprint (Mantra MFS100) OR Uploaded Image
        Ã¢â€ â€œ
  Image Acquisition Layer
        Ã¢â€ â€œ
  Centralized Preprocessing Pipeline (src/preprocessing.py)
        Ã¢â€ â€œ
  Champion Ensemble V2 Inference Engine (src/predictor.py)
        Ã¢â€ â€œ
  ABO/Rh Analytical Decomposition
        Ã¢â€ â€œ
  Confidence Threshold Evaluation (>= 60%)
        Ã¢â€ â€œ
  SQLite Audit Record (src/database.py)
        Ã¢â€ â€œ
  Clinical PDF Report (src/report_generator.py)
        Ã¢â€ â€œ
  Web Dashboard Response (Flask REST API)
```

---

## 2. Complete Component Architecture

```
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š                        FRONTEND LAYER (Browser)                              Ã¢â€â€š
Ã¢â€â€š  templates/index.html Ã¢â‚¬â€ Clinical Dashboard                                   Ã¢â€â€š
Ã¢â€â€š  templates/history.html Ã¢â‚¬â€ Audit History View                                 Ã¢â€â€š
Ã¢â€â€š  static/js/app.js Ã¢â‚¬â€ UI Logic & API calls                                    Ã¢â€â€š
Ã¢â€â€š  static/css/style.css Ã¢â‚¬â€ Healthcare-themed responsive design                  Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                                  Ã¢â€â€š HTTP REST (port 5000)
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š                        FLASK BACKEND (app.py)                                Ã¢â€â€š
Ã¢â€â€š  GET  /                 Ã¢â‚¬â€ Dashboard HTML                                     Ã¢â€â€š
Ã¢â€â€š  GET  /history          Ã¢â‚¬â€ History HTML                                       Ã¢â€â€š
Ã¢â€â€š  GET  /api/status       Ã¢â‚¬â€ System health (model + scanner status)             Ã¢â€â€š
Ã¢â€â€š  POST /api/upload       Ã¢â‚¬â€ Image file upload Ã¢â€ â€™ prediction                     Ã¢â€â€š
Ã¢â€â€š  POST /api/capture      Ã¢â‚¬â€ MFS100 single capture Ã¢â€ â€™ prediction                 Ã¢â€â€š
Ã¢â€â€š  POST /api/capture/consensus Ã¢â‚¬â€ 3-scan LEFT THUMB consensus                  Ã¢â€â€š
Ã¢â€â€š  GET  /api/history      Ã¢â‚¬â€ Prediction audit records (JSON)                    Ã¢â€â€š
Ã¢â€â€š  GET  /api/report/<id>  Ã¢â‚¬â€ Clinical PDF download                              Ã¢â€â€š
Ã¢â€â€š  POST /api/scanner/mode Ã¢â‚¬â€ Toggle hardware/mock mode                          Ã¢â€â€š
Ã¢â€â€š  GET  /api/scanner/status Ã¢â‚¬â€ Scanner + bridge connectivity                    Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
       Ã¢â€â€š                      Ã¢â€â€š                      Ã¢â€â€š                      Ã¢â€â€š
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š src/        Ã¢â€â€š  Ã¢â€â€š src/predictor.py   Ã¢â€â€š  Ã¢â€â€š src/database.py   Ã¢â€â€š  Ã¢â€â€š src/report_ Ã¢â€â€š
Ã¢â€â€š mfs100/     Ã¢â€â€š  Ã¢â€â€š BloodGroupPredictorÃ¢â€â€š  Ã¢â€â€š SQLite Audit DB   Ã¢â€â€š  Ã¢â€â€š generator.pyÃ¢â€â€š
Ã¢â€â€š scanner.py  Ã¢â€â€š  Ã¢â€â€š                    Ã¢â€â€š  Ã¢â€â€š                   Ã¢â€â€š  Ã¢â€â€š ReportLab   Ã¢â€â€š
Ã¢â€â€š MFS100      Ã¢â€â€š  Ã¢â€â€š Champion Ensemble  Ã¢â€â€š  Ã¢â€â€š predictions table Ã¢â€â€š  Ã¢â€â€š PDF engine  Ã¢â€â€š
Ã¢â€â€š Adapter     Ã¢â€â€š  Ã¢â€â€š V2 Inference       Ã¢â€â€š  Ã¢â€â€š CRUD operations   Ã¢â€â€š  Ã¢â€â€š             Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
       Ã¢â€â€š                      Ã¢â€â€š
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š src/mfs100_ Ã¢â€â€š  Ã¢â€â€š src/preprocessing.py                                    Ã¢â€â€š
Ã¢â€â€š bridge/     Ã¢â€â€š  Ã¢â€â€š Centralized Image Preprocessing Pipeline               Ã¢â€â€š
Ã¢â€â€š bridge.py   Ã¢â€â€š  Ã¢â€â€š (shared by training, upload, MFS100, mock)             Ã¢â€â€š
Ã¢â€â€š 32-bit HTTP Ã¢â€â€š  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
Ã¢â€â€š bridge      Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

---

## 3. MFS100 Hardware Architecture

### 3.1 The Cross-Architecture Problem

The Mantra MFS100 driver DLL (`MFS100Dll.dll`) is a 32-bit (x86) Windows native library.
Python-based AI/ML environments run as 64-bit (x64) processes.

```
Windows 64-bit Python (AI/ML environment)
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ TensorFlow 2.21.0 (64-bit)
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Flask (64-bit)
    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ CANNOT load 32-bit DLLs directly (WinError 193)
```

### 3.2 32-bit Bridge Solution

A dedicated lightweight 32-bit Python bridge process (`src/mfs100_bridge/bridge.py`) runs under a separate 32-bit Python installation and serves as an HTTP intermediary:

```
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€       Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š   64-bit Flask Backend          Ã¢â€â€š       Ã¢â€â€š   32-bit MFS100 Bridge           Ã¢â€â€š
Ã¢â€â€š   (Python 3.10.11 x64)         Ã¢â€â€š       Ã¢â€â€š   (C:\Python310-32\python.exe)   Ã¢â€â€š
Ã¢â€â€š                                 Ã¢â€â€š       Ã¢â€â€š                                   Ã¢â€â€š
Ã¢â€â€š   MFS100Scanner.capture() Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Âº Ã¢â€â€š HTTP  Ã¢â€â€š Ã¢â€”â€žÃ¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ ctypes.WinDLL(MFS100Dll)   Ã¢â€â€š
Ã¢â€â€š   (src/mfs100/scanner.py)      Ã¢â€â€š :5100 Ã¢â€â€š       Physical MFS100 USB SDK    Ã¢â€â€š
Ã¢â€â€š                                 Ã¢â€â€šÃ¢â€”â€žÃ¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Ã¢â€â€šÃ¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Âº  scanner.capture()           Ã¢â€â€š
Ã¢â€â€š   Returns: image bytes +        Ã¢â€â€š       Ã¢â€â€š      quality, nfiq, image bytes  Ã¢â€â€š
Ã¢â€â€š   quality_meta (nfiq, score)    Ã¢â€â€š       Ã¢â€â€š                                   Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                                                        Ã¢â€â€š
                                          Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
                                          Ã¢â€â€š   Mantra MFS100 USB Hardware    Ã¢â€â€š
                                          Ã¢â€â€š   Optical platen: 316x354 px    Ã¢â€â€š
                                          Ã¢â€â€š   500 DPI optical sensor        Ã¢â€â€š
                                          Ã¢â€â€š   SDK: MFS100Dll.dll (32-bit)   Ã¢â€â€š
                                          Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

### 3.3 Bridge Startup

```bash
# Terminal 1: Start 32-bit bridge
C:\Python310-32\python.exe -m src.mfs100_bridge.bridge

# Terminal 2: Start 64-bit Flask backend
python app.py
```

### 3.4 Mock Mode

When `MFS100_MOCK=true` (default) or physical hardware is unavailable:
- `MFS100Scanner` operates in mock mode.
- Cycles through authentic fingerprint samples from `data/sample/`.
- Uses identical preprocessing, inference, and reporting paths.
- `finger_used` is set to `None` (not LEFT_THUMB) in mock captures.
- DB label: `MFS100 (Mock)` instead of `MFS100`.

---

## 4. Dataset Architecture

```
dataset_blood_group/          Ã¢â€  Raw source fingerprint dataset
data/
  raw/                        Ã¢â€  Unmodified raw images (6,000 total)
  processed/                  Ã¢â€  5,985 cleaned images (15 duplicates quarantined)
  quarantine/
    duplicates/               Ã¢â€  15 quarantined SHA-256 duplicate images
    quarantine_manifest.json  Ã¢â€  Full quarantine audit map
  sample/                     Ã¢â€  8 reference images (one per class, for mock mode)
  captured/                   Ã¢â€  Runtime: MFS100 and upload captures stored here
  blood_group.db              Ã¢â€  SQLite prediction audit database
```

**Dataset splits (stratified, leakage-free):**

| Split | Images | Purpose |
|---|---|---|
| Training | 4,189 | Model training |
| Validation | 898 | Hyperparameter tuning and intermediate evaluation |
| Test (LOCKED) | 898 | Official one-time benchmark evaluation Ã¢â‚¬â€ NOT re-accessed |

---

## 5. Preprocessing Pipeline Architecture

### 5.1 Upload Mode (source="upload")

```
Input: File path / bytes / PIL
    Ã¢â€ â€œ load_image_to_pil()
    Ã¢â€ â€œ composite_alpha_on_white()    Ã¢â€  RGBA transparent borders Ã¢â€ â€™ white (255,255,255)
    Ã¢â€ â€œ resize_to_target(224Ãƒâ€”224)     Ã¢â€  Lanczos resampling
    Ã¢â€ â€œ Per-architecture preprocessing:
       ResNet50: ImageEnhance.Contrast(Ãƒâ€”1.2) + resnet50.preprocess_input
       VGG16:    vgg16.preprocess_input
       ENB0:     efficientnet.preprocess_input
    Ã¢â€ â€œ Batch tensors Ã¢â€ â€™ ensemble inference
```

### 5.2 MFS100 Hardware Mode (source="mfs100")

```
Input: Raw MFS100 capture (316Ãƒâ€”354, NFIQ quality check)
    Ã¢â€ â€œ load_image_to_pil()
    Ã¢â€ â€œ composite_alpha_on_white()
    Ã¢â€ â€œ crop_mfs100_foreground()      Ã¢â€  Hardware foreground crop (removes sensor margins)
       4 safeguards: blank guard, region check, size floor, margin presence check
    Ã¢â€ â€œ enhance_mfs100_ridge_contrast()
       CLAHE (clipLimit=2.5, 8Ãƒâ€”8) + percentile stretch + bilateral filter
    Ã¢â€ â€œ resize_to_target(224Ãƒâ€”224)
    Ã¢â€ â€œ Per-architecture preprocessing (same as upload above)
    Ã¢â€ â€œ Batch tensors Ã¢â€ â€™ ensemble inference
```

---

## 6. Champion Ensemble V2 Inference Architecture

```
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š                    Champion Ensemble V2                                  Ã¢â€â€š
Ã¢â€â€š              Locked Test Accuracy: 85.41% (767/898)                     Ã¢â€â€š
Ã¢â€â€š                                                                          Ã¢â€â€š
Ã¢â€â€š  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€  Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€š ResNet50 (Improved)  Ã¢â€â€š  Ã¢â€â€š      VGG16           Ã¢â€â€š  Ã¢â€â€š EfficientNetB0Ã¢â€â€š  Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€š 277 MB checkpoint    Ã¢â€â€š  Ã¢â€â€š 164 MB checkpoint    Ã¢â€â€š  Ã¢â€â€š 41 MB checkpt Ã¢â€â€š  Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€š Weight: 0.58 (58%)  Ã¢â€â€š  Ã¢â€â€š Weight: 0.38 (38%)  Ã¢â€â€š  Ã¢â€â€š Weight: 0.04  Ã¢â€â€š  Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€š 1.2x contrast        Ã¢â€â€š  Ã¢â€â€š Native VGG16         Ã¢â€â€š  Ã¢â€â€š Native ENB0   Ã¢â€â€š  Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€š preprocessing        Ã¢â€â€š  Ã¢â€â€š preprocessing        Ã¢â€â€š  Ã¢â€â€š preprocessing Ã¢â€â€š  Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€š
Ã¢â€â€š             Ã¢â€â€š                         Ã¢â€â€š                       Ã¢â€â€š          Ã¢â€â€š
Ã¢â€â€š  P_resnet[8] Ãƒâ€” 0.58   +  P_vgg[8] Ãƒâ€” 0.38   +  P_eff[8] Ãƒâ€” 0.04         Ã¢â€â€š
Ã¢â€â€š             Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â´Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ          Ã¢â€â€š
Ã¢â€â€š                                       Ã¢â€ â€œ                                  Ã¢â€â€š
Ã¢â€â€š              Ensemble_probs[8] = weighted soft-vote average              Ã¢â€â€š
Ã¢â€â€š                                       Ã¢â€ â€œ                                  Ã¢â€â€š
Ã¢â€â€š              predicted_class = argmax(Ensemble_probs)                   Ã¢â€â€š
Ã¢â€â€š              confidence = max(Ensemble_probs)                           Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

**Post-inference decision layers:**

```
confidence >= 0.60 Ã¢â€ â€™ status = NORMAL
confidence <  0.60 Ã¢â€ â€™ status = LOW_CONFIDENCE (advisory + recapture recommendation)

ABO/Rh Decomposition (analytics only Ã¢â‚¬â€ does NOT change prediction):
  P(A)  = P(A+) + P(A-)
  P(B)  = P(B+) + P(B-)
  P(AB) = P(AB+) + P(AB-)
  P(O)  = P(O+) + P(O-)
  top_abo = argmax of above

Rh Uncertainty Advisory (does NOT change prediction):
  rh_margin < 0.20  Ã¢â€ â€™ advisory flag
  confidence < 0.65 Ã¢â€ â€™ advisory flag
  predicted_Rh == "-" Ã¢â€ â€™ advisory flag (weak ridge-Rh correlation)
```

---

## 7. Database Architecture

**File:** `data/blood_group.db` (SQLite)

**Table: predictions**

| Column | Type | Description |
|---|---|---|
| id | INTEGER PRIMARY KEY | Auto-increment record ID |
| timestamp | TEXT | ISO 8601 datetime of prediction |
| input_type | TEXT | 'upload', 'MFS100', 'MFS100 Consensus - Left Thumb (3 Impressions)', 'MFS100 (Mock)' |
| image_path | TEXT | Filename of stored fingerprint image |
| predicted_group | TEXT | Predicted blood group (e.g., 'O+') |
| confidence | REAL | Float 0.0Ã¢â‚¬â€œ1.0 |
| status | TEXT | 'NORMAL' or 'LOW_CONFIDENCE' |
| notes | TEXT | Model message / advisory text |
| finger_used | TEXT | 'LEFT_THUMB' for hardware; NULL for upload/mock |

**Migration safety:** `finger_used` was added via non-destructive `ALTER TABLE` Ã¢â‚¬â€ existing records retain `NULL`.

---

## 8. PDF Report Architecture

**Generator:** `src/report_generator.py` (ReportLab)
**Output:** `reports/blood_group_report_<id>_<timestamp>.pdf`

**Report structure:**
1. Header banner Ã¢â‚¬â€ project title
2. Medical disclaimer (mandatory red alert box)
3. Examination & audit details table (Report ID, timestamp, input source, confidence, model name, LEFT THUMB row if hardware)
4. Biometric impression + predicted blood group card (large 32pt display)
5. ABO group, Rh factor, confidence metadata
6. Rh Uncertainty Advisory (amber banner, conditional)
7. 8-class probability distribution table
8. Footer

---

## 9. Three Operational Modes

| Mode | Trigger | Source | finger_used | Preprocessing | PDF Label |
|---|---|---|---|---|---|
| Real MFS100 | scanner.mock=False | mfs100 | LEFT_THUMB | Crop + enhance + multi-arch | MFS100 \| LEFT THUMB |
| Mock Mode | MFS100_MOCK=true | upload | NULL | Multi-arch (no crop) | MFS100 (Mock Mode) |
| File Upload | POST /api/upload | upload | NULL | Multi-arch (no crop) | Upload |

---

## 10. Left Thumb Acquisition Protocol

**Standard finger:** LEFT THUMB (standardized for all physical MFS100 demonstrations)

**Important constraints:**
- LEFT_THUMB is acquisition metadata stored in `finger_used` column only.
- The AI model does NOT receive `finger_used` as an input.
- The prediction is NOT modified based on which finger is used.
- The scanner does NOT automatically verify finger identity.
- This is a convention-based protocol for demo reproducibility only.

**Reference:** `docs/MFS100_ACQUISITION_PROTOCOL.md`

---

## 11. Project Directory Structure

```
major project/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ app.py                          Ã¢â€  Flask web server (main entry point)
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt                Ã¢â€  Python dependencies
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ README.md                       Ã¢â€  Project overview
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ models/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ resnet50_improved_blood_group.keras   Ã¢â€  ResNet50 (277 MB, FROZEN)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ vgg16_blood_group.keras               Ã¢â€  VGG16 (164 MB, FROZEN)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ efficientnetb0_blood_group.keras      Ã¢â€  EfficientNetB0 (41 MB, FROZEN)
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ class_names.json                      Ã¢â€  8-class label mapping
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ outputs/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ ensemble_weights.json        Ã¢â€  LOCKED: ResNet50=58%, VGG16=38%, ENB0=4%
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ FINAL_MODEL_EVALUATION.json  Ã¢â€  Official 85.41% benchmark record
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ dataset_manifest.csv         Ã¢â€  Full 5,985-image split manifest
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ ab_plus_diagnostics/         Ã¢â€  AB+ generalization analysis outputs
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ src/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config.py                    Ã¢â€  All paths and constants
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ predictor.py                 Ã¢â€  Inference engine (Champion Ensemble V2)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ preprocessing.py             Ã¢â€  Centralized preprocessing pipeline
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ database.py                  Ã¢â€  SQLite CRUD operations
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ report_generator.py          Ã¢â€  PDF clinical report generator
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ mfs100/scanner.py            Ã¢â€  MFS100 hardware + mock adapter
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ mfs100_bridge/bridge.py      Ã¢â€  32-bit HTTP bridge process
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ templates/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ index.html                   Ã¢â€  Clinical dashboard UI
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ history.html                 Ã¢â€  Audit history page
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ static/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ css/style.css                Ã¢â€  Healthcare dashboard styles
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ js/app.js                    Ã¢â€  Frontend logic
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tests/                           Ã¢â€  56 automated tests (all passing)
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ data/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ blood_group.db               Ã¢â€  SQLite prediction audit database
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ sample/                      Ã¢â€  8 mock mode reference images
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ captured/                    Ã¢â€  Runtime fingerprint captures
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ reports/                         Ã¢â€  Generated PDF clinical reports
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ docs/                            Ã¢â€  Full project documentation suite
```

---

*This document is part of the B.E. Final Year Major Project Ã¢â‚¬â€ Biometric Blood Group Identification System.*
*Architecture frozen as of 2026-09-05. For academic demonstration purposes only.*
