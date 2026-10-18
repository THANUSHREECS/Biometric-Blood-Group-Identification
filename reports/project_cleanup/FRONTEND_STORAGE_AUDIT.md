# Frontend Storage & Dependency Audit
**Project:** HemoPrint AI — Frontend Application Storage Profile  
**Directory:** [`live_app/frontend/`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/live_app/frontend)  
**Audit Mode:** Inspection Only (0 modifications)

---

## 1. Storage Breakdown

| Component | Path | Size (MB) | File Count | Rebuildable from Source? | Lockfile Available? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Node Modules** | `live_app/frontend/node_modules/` | **`127.47 MB`** | 5,401 | Yes (via `npm ci` / `npm install`) | Yes (`package-lock.json`) |
| **Production Build** | `live_app/frontend/dist/` | **`0.38 MB`** | 5 | Yes (via `npm run build` in 1.5s) | N/A |
| **Vite Dev Cache** | `live_app/frontend/node_modules/.vite/` | **`7.06 MB`** | — | Yes (auto-generated on dev start) | N/A |
| **Source Code** | `live_app/frontend/src/` | **`0.17 MB`** | 20 | Source of Truth (Preserved) | N/A |
| **Total Frontend** | `live_app/frontend/` | **`~128.11 MB`** | **~5,430** | — | — |

---

## 2. Potential Storage Reclaim Scenarios

### Scenario A: Prune Production Build (`dist/`)
* **Storage Reclaimed:** `0.38 MB` (398 KB)
* **Application Impact:** Dev server (`npm run dev`) does NOT use `dist/`; it compiles in-memory with Vite. Production deployment requires `npm run build`.
* **Regeneration Cost:** `1.52 seconds` via `npm run build`.
* **Recommendation:** Keep `dist/` in active working directory so backend can serve compiled static assets if configured; exclude from research publication zip.

### Scenario B: Prune Vite Compilation Cache (`node_modules/.vite/`)
* **Storage Reclaimed:** `7.06 MB`
* **Application Impact:** Zero impact. Vite automatically rebuilds cache on next browser reload.
* **Regeneration Cost:** Sub-second during next dev launch.
* **Recommendation:** Safe to purge if reclaiming small cache files; currently part of `node_modules`.

### Scenario C: Clean Dependency Tree (`node_modules/`)
* **Storage Reclaimed:** **`127.47 MB`**
* **Application Impact:** Immediate failure of frontend dev server (`npm run dev`) and build script (`npm run build`). The application cannot be launched until dependencies are reinstalled.
* **Regeneration Cost:** Re-running `npm ci` requires network access, Node.js environment, and approximately 15–30 seconds.
* **Lockfile Integrity:** [`package-lock.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/live_app/frontend/package-lock.json) is verified present and intact, ensuring 100% deterministic reinstallation.
* **Recommendation:** **Retain in active local working environment** to ensure immediate application usability. **Exclude completely from public publication package / Zenodo release archive**.

---

## 3. Publication Packaging Guidance
When packaging the research codebase for journal submission:
1. Include `package.json` and `package-lock.json`.
2. Include the entire `src/`, `public/`, `index.html`, `vite.config.ts`, `tsconfig.json`.
3. Exclude `node_modules/` (saves 127.47 MB and eliminates 5,401 third-party files).
4. Exclude `dist/` (saves 0.38 MB; reviewer builds with one command: `npm ci && npm run build`).
