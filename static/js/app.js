
// Globals
let CONFIG = { threshold: 70.0, image_size: 112, class_order: ["A+", "AB+", "B+", "O+"] };

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        if (res.ok) {
            CONFIG = await res.json();
            document.getElementById("ui-thresh-1").innerText = CONFIG.threshold.toFixed(1);
            document.getElementById("ui-thresh-2").innerText = CONFIG.threshold.toFixed(0);
            document.getElementById("ui-thresh-3").innerText = CONFIG.threshold.toFixed(1);
            document.getElementById("ui-res-1").innerText = CONFIG.image_size;
            document.getElementById("ui-res-2").innerText = CONFIG.image_size;
        }
    } catch (e) {
        console.warn("Could not fetch config, using defaults", e);
    }
}
fetchConfig();

/**
 * app.js
 * Frontend controller for Biometric Blood Group Identification System.
 * Uses native fetch() for communication with Flask backend endpoints.
 */

let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
    checkSystemStatus();
    initDropzone();
});

function switchTab(tabName) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

    if (tabName === "scanner") {
        document.getElementById("tab-scanner-btn").classList.add("active");
        document.getElementById("tab-scanner-content").classList.add("active");
    } else {
        document.getElementById("tab-upload-btn").classList.add("active");
        document.getElementById("tab-upload-content").classList.add("active");
    }
}

let currentScannerMode = "mock";

async function checkSystemStatus() {
    try {
        const res = await fetch("/api/scanner/status");
        const data = await res.json();
        const sc = data.device || {};
        currentScannerMode = data.mock_mode ? "mock" : "hardware";

        // Update Scanner Indicator
        const scannerDot = document.querySelector("#scanner-indicator .status-dot");
        const scannerText = document.getElementById("scanner-status-text");
        const badgePill = document.getElementById("scanner-badge-pill");
        const modeLabel = document.getElementById("scanner-mode-label");
        const toggleBtn = document.getElementById("btn-toggle-scanner-mode");
        const mockPicker = document.getElementById("mock-sample-group");
        const hwCard = document.getElementById("hardware-info-card");

        if (data.mock_mode) {
            scannerDot.classList.add("active");
            scannerText.innerText = "Scanner: Mock Mode (Ready)";
            if (badgePill) {
                badgePill.innerText = "MOCK MODE";
                badgePill.style.background = "#fef3c7";
                badgePill.style.color = "#92400e";
                badgePill.style.border = "1px solid #fcd34d";
            }
            if (modeLabel) modeLabel.innerText = "Mode: Mock Emulation (Authentic Samples)";
            if (toggleBtn) toggleBtn.innerText = "Switch to Physical Scanner";
            if (mockPicker) mockPicker.style.display = "block";
            if (hwCard) hwCard.style.display = "none";

            // Hide LEFT THUMB protocol panel in mock mode
            const ltPanel = document.getElementById("left-thumb-protocol-panel");
            if (ltPanel) ltPanel.style.display = "none";

            // Restore standard consensus label in mock mode
            const consensusLabel = document.getElementById("consensus-mode-label");
            if (consensusLabel) {
                consensusLabel.innerHTML = `🎯 3-Scan Consensus Mode <small style="font-weight: 400; color: #15803d;">(Recommended: averages 3 impressions to eliminate noise)</small>`;
            }
        } else {
            const bridgeReachable = data.bridge_health && data.bridge_health.reachable;
            if (bridgeReachable) {
                scannerDot.classList.add("active");
                scannerText.innerText = "Scanner: Physical MFS100 (Online)";
                if (badgePill) {
                    badgePill.innerText = "REAL MFS100";
                    badgePill.style.background = "#dcfce7";
                    badgePill.style.color = "#166534";
                    badgePill.style.border = "1px solid #86efac";
                }
                if (modeLabel) modeLabel.innerText = "Mode: Mantra MFS100 Physical Hardware Active";
                if (hwCard) {
                    hwCard.style.display = "block";
                    document.getElementById("hw-serial").innerText = sc.serial || "11215389";
                    document.getElementById("hw-model").innerText = sc.model || "MFS100";
                    document.getElementById("hw-make").innerText = sc.make || "MANTRA";
                    document.getElementById("hw-res").innerText = `${sc.width || 316}×${sc.height || 354}`;
                    document.getElementById("hw-status").innerText = sc.status || "Ready (Bridge Online)";
                }

                // Show LEFT THUMB protocol panel when hardware is active
                const ltPanel = document.getElementById("left-thumb-protocol-panel");
                if (ltPanel) ltPanel.style.display = "block";

                // Update consensus label to reflect Left Thumb Protocol
                const consensusLabel = document.getElementById("consensus-mode-label");
                if (consensusLabel) {
                    consensusLabel.innerHTML = `🖐 3-Scan Left Thumb Protocol <small style="font-weight: 400; color: #15803d;">(Recommended: 3 left-thumb impressions for maximum reliability)</small>`;
                }
            } else {
                scannerDot.classList.remove("active");
                scannerText.innerText = "Scanner: Bridge Offline";
                if (badgePill) {
                    badgePill.innerText = "BRIDGE OFFLINE";
                    badgePill.style.background = "#fee2e2";
                    badgePill.style.color = "#991b1b";
                    badgePill.style.border = "1px solid #fca5a5";
                }
                if (modeLabel) modeLabel.innerText = "Mode: 32-bit Bridge Offline (Run python -m src.mfs100_bridge.bridge)";

                // Hide LEFT THUMB panel if bridge is offline
                const ltPanel = document.getElementById("left-thumb-protocol-panel");
                if (ltPanel) ltPanel.style.display = "none";
            }
            if (toggleBtn) toggleBtn.innerText = "Switch to Mock Mode";
            if (mockPicker) mockPicker.style.display = "none";
        }

        // Update Model Tag
        const modelRes = await fetch("/api/status");
        const statusData = await modelRes.json();
        const modelTag = document.getElementById("model-tag");
        if (statusData.model) {
            modelTag.innerText = "E2 Champion Ensemble (" + statusData.model.accuracy + ")";
            modelTag.className = "badge badge-primary";
            modelTag.title = "E2 Champion Model - Research accuracy is 86.02% (random split, includes identity leakage) | ~0.00% (group-aware), not a guarantee of individual prediction accuracy";
        }

    } catch (err) {
        console.error("Status check failed:", err);
    }
}

async function toggleScannerMode() {
    try {
        const res = await fetch("/api/scanner/mode", { method: "POST" });
        const data = await res.json();
        await checkSystemStatus();
    } catch (err) {
        alert("Failed to toggle mode: " + err.message);
    }
}

async function checkScanner() {
    try {
        const res = await fetch("/api/scanner/status");
        const data = await res.json();
        const sc = data.device || {};
        const bh = data.bridge_health || {};
        alert(
            `Mantra Scanner Diagnostic:\n\n` +
            `Mode: ${data.mock_mode ? 'Mock Mode (Simulation)' : 'Physical MFS100'}\n` +
            `Connected: ${data.is_connected ? 'Yes' : 'No'}\n` +
            `Bridge Reachable: ${bh.reachable ? 'Yes' : 'No'}\n` +
            `Serial No: ${sc.serial || 'N/A'}\n` +
            `Model: ${sc.model || 'MFS100'}\n` +
            `Make: ${sc.make || 'Mantra'}\n` +
            `Resolution: ${sc.width || 316} x ${sc.height || 354} px\n` +
            `Device Status: ${sc.status || 'N/A'}`
        );
    } catch (err) {
        alert("Scanner diagnostic failed: " + err.message);
    }
}

// -------------------------------------------------------------------------
// Fingerprint Acquisition via Mantra / Mock Scanner
// -------------------------------------------------------------------------
async function triggerCapture() {
    const progressBox = document.getElementById("progress-box");
    const progressText = document.getElementById("progress-text");
    const captureBtn = document.getElementById("btn-capture");
    const sampleSelect = document.getElementById("sample-select");
    const consensusChk = document.getElementById("chk-consensus-mode");
    const isConsensus = consensusChk && consensusChk.checked;

    // Only send sample_class if explicitly in mock mode
    const sampleClass = (currentScannerMode === "mock" && sampleSelect) ? sampleSelect.value : null;

    try {
        captureBtn.disabled = true;
        progressBox.style.display = "flex";
        
        if (isConsensus) {
            if (currentScannerMode === "hardware") {
                progressText.innerText = "3-Scan Left Thumb Protocol: capturing impression 1 of 3... Keep LEFT THUMB steady!";
            } else {
                progressText.innerText = "Capturing 3 authentic impressions for consensus voting (Mock Mode)...";
            }
        } else {
            if (currentScannerMode === "hardware") {
                progressText.innerText = "Place your LEFT THUMB on the Mantra MFS100 scanner and hold steady...";
            } else {
                progressText.innerText = "Capturing authentic fingerprint sample (Mock Mode)...";
            }
        }

        const payload = (currentScannerMode === "mock" && sampleClass) ? { sample_class: sampleClass } : {};
        const endpoint = isConsensus ? "/api/capture/consensus" : "/api/capture";

        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (!res.ok || data.error) {
            throw new Error(data.error || "Capture failed");
        }

        progressText.innerText = "Enhancing ridges & evaluating via Champion Ensemble V2...";
        await new Promise(r => setTimeout(r, 400));
        progressBox.style.display = "none";

        renderPredictionResult(data);

    } catch (err) {
        progressBox.style.display = "none";
        alert("Capture error: " + err.message);
    } finally {
        captureBtn.disabled = false;
    }
}

// -------------------------------------------------------------------------
// File Upload & Drag & Drop Handling
// -------------------------------------------------------------------------
function initDropzone() {
    const dropzone = document.getElementById("dropzone");

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "#2563eb";
            dropzone.style.backgroundColor = "#e0f2fe";
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "#93c5fd";
            dropzone.style.backgroundColor = "#f0f7ff";
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelected(input) {
    if (input.files && input.files[0]) {
        handleFile(input.files[0]);
    }
}

function handleFile(file) {
    selectedFile = file;
    document.getElementById("file-name-preview").innerText = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    document.getElementById("btn-upload-predict").disabled = false;
}

async function submitUpload() {
    if (!selectedFile) return;

    const progressBox = document.getElementById("progress-box");
    const progressText = document.getElementById("progress-text");
    const uploadBtn = document.getElementById("btn-upload-predict");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        uploadBtn.disabled = true;
        progressBox.style.display = "flex";
        progressText.innerText = "Uploading fingerprint image...";

        await new Promise(r => setTimeout(r, 400));
        progressText.innerText = "Preprocessing & running ResNet50 classification...";

        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!res.ok || data.error) {
            throw new Error(data.error || "Upload failed");
        }

        progressText.innerText = "Prediction completed successfully!";
        await new Promise(r => setTimeout(r, 300));
        progressBox.style.display = "none";

        renderPredictionResult(data);

    } catch (err) {
        progressBox.style.display = "none";
        alert("Upload error: " + err.message);
    } finally {
        uploadBtn.disabled = false;
    }
}

// -------------------------------------------------------------------------
// Render Prediction Results in View
// -------------------------------------------------------------------------
function renderPredictionResult(data) {
    document.getElementById("result-placeholder").style.display = "none";
    const resultContent = document.getElementById("result-content");
    resultContent.style.display = "block";

    // Set preview image
    if (data.image_url) {
        document.getElementById("preview-image").src = data.image_url;
    }

    // Blood group value - hide if low confidence
    if (data.needs_confirmation) {
        document.getElementById("blood-group-val").innerText = "Not Reliable";
        document.getElementById("blood-group-val").style.fontSize = "1.5rem";
        document.getElementById("blood-group-val").style.color = "#b45309";
    } else {
        document.getElementById("blood-group-val").innerText = data.blood_group;
        document.getElementById("blood-group-val").style.fontSize = "3.5rem";
        document.getElementById("blood-group-val").style.color = "#dc2626";
    }

    // Confidence
    const confPct = (data.confidence * 100).toFixed(1);
    document.getElementById("confidence-val").innerText = `${confPct}%`;
    document.getElementById("confidence-bar").style.width = `${confPct}%`;

    // Research/Prototype Advisory Decision-Support Layer (tau = 0.70)
    const chip = document.getElementById("confidence-chip");
    const advAlert = document.getElementById("advisory-screening-alert");
    const advMsg = document.getElementById("advisory-screening-message");
    const advPill = document.getElementById("advisory-reason-pill");
    const advOkAlert = document.getElementById("advisory-ok-alert");
    const lowConfAlert = document.getElementById("low-confidence-alert");

    // Legacy 60% threshold indicator
    if (lowConfAlert) {
        lowConfAlert.style.display = data.is_low_confidence ? "block" : "none";
    }

    // Advisory Confirmation Screening (Validation Threshold tau = 0.70 + Rh factor check)
    if (data.needs_confirmation) {
        if (chip) {
            chip.innerText = "⚠️ NEEDS CONFIRMATION";
            chip.className = "status-chip chip-warning";
        }
        if (advAlert) {
            advAlert.style.display = "block";
            if (advMsg && data.advisory_message) {
                advMsg.innerText = data.advisory_message;
            }
            if (advPill) {
                advPill.innerText = (data.advisory_reason || "CONFIRMATION_RECOMMENDED").replace(/_/g, " ");
            }
        }
        if (advOkAlert) advOkAlert.style.display = "none";
    } else {
        if (chip) {
            chip.innerText = "✓ CONFIDENCE OK";
            chip.className = "status-chip";
        }
        if (advAlert) advAlert.style.display = "none";
        if (advOkAlert) advOkAlert.style.display = "block";
    }




    // Smart NFIQ Consensus Suggestion
    const smartConsensus = document.getElementById("smart-consensus-suggestion");
    const consensusChk = document.getElementById("chk-consensus-mode");
    const isAlreadyConsensus = consensusChk && consensusChk.checked;
    const qm = data.quality_metadata;

    if (smartConsensus) {
        if (!isAlreadyConsensus && qm && (qm.nfiq >= 2 || (qm.quality && qm.quality < 80))) {
            const promptText = `Suboptimal optical impression detected (NFIQ: ${qm.nfiq}, Quality: ${qm.quality}/100). For maximum stability and noise reduction, switch to 3-Scan Consensus mode.`;
            document.getElementById("smart-consensus-text").innerText = promptText;
            smartConsensus.style.display = "block";
        } else {
            smartConsensus.style.display = "none";
        }
    }

    // Render 8-Class Probability Grid
    const probGrid = document.getElementById("prob-grid");
    probGrid.innerHTML = "";

    if (data.probabilities) {
        for (const [bg, prob] of Object.entries(data.probabilities)) {
            const isTop = (bg === data.blood_group) && !(data.needs_confirmation);
            const item = document.createElement("div");
            item.className = `prob-item ${isTop ? 'active' : ''}`;
            item.innerHTML = `
                <div class="prob-label">${bg}</div>
                <div class="prob-val">${(prob * 100).toFixed(1)}%</div>
            `;
            probGrid.appendChild(item);
        }
    }

    // ---- MFS100 Crop Metadata Badge ----
    // Show a transparent info block only when real hardware foreground cropping was applied.
    let cropInfoEl = document.getElementById("crop-metadata-info");
    if (!cropInfoEl) {
        cropInfoEl = document.createElement("div");
        cropInfoEl.id = "crop-metadata-info";
        cropInfoEl.style.cssText = [
            "margin-top:12px",
            "padding:10px 14px",
            "border-radius:8px",
            "font-size:0.78rem",
            "font-family:monospace",
            "background:linear-gradient(135deg,#ecfdf5,#d1fae5)",
            "border:1px solid #6ee7b7",
            "color:#065f46",
            "display:none"
        ].join(";");
        // Insert after the probability grid
        probGrid.parentNode.insertBefore(cropInfoEl, probGrid.nextSibling);
    }

    const cm = data.crop_metadata;
    if (cm && cm.crop_applied) {
        const bbox = cm.bbox || [0,0,0,0];
        const src  = cm.orig_size || [0,0];
        cropInfoEl.innerHTML = [
            `<strong>&#x2702;&#xfe0f; MFS100 Foreground Crop Applied</strong>`,
            `<span style="margin-left:8px;padding:1px 6px;background:#10b981;color:#fff;border-radius:4px;font-size:0.72rem;">${data.scanner_mode === 'hardware' ? 'REAL MFS100' : 'MFS100'}</span>`,
            `<br><span>Original: ${src[0] || '?'}&times;${src[1] || '?'}px &rarr; `,
            `BBox: (${bbox[0]||0}, ${bbox[1]||0}) &rarr; (${bbox[2]||'?'}, ${bbox[3]||'?'}) &rarr; 112&times;112px</span>`,
            cm.guard_triggered ? `<br><span style="color:#d97706;">&#9888; Blank-platen guard triggered &mdash; full image used</span>` : ""
        ].join("");
        cropInfoEl.style.display = "block";
    } else {
        cropInfoEl.style.display = "none";
    }

    // ---- Source tag on scanner mode ----
    // If the result panel has a scanner-source-tag element, update it
    const srcTag = document.getElementById("result-source-tag");
    if (srcTag) {
        if (data.scanner_mode === "hardware") {
            srcTag.innerText = "Source: MFS100 Physical Hardware";
            srcTag.style.cssText = "display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.75rem;background:#dcfce7;color:#166534;border:1px solid #86efac;";
        } else if (data.scanner_mode === "mock") {
            srcTag.innerText = "Source: Mock Scanner";
            srcTag.style.cssText = "display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.75rem;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;";
        } else {
            srcTag.innerText = "Source: File Upload";
            srcTag.style.cssText = "display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.75rem;background:#e0f2fe;color:#075985;border:1px solid #7dd3fc;";
        }
    }

    // ---- Finger Used Badge (LEFT THUMB protocol indicator) ----
    const fingerTag = document.getElementById("result-finger-tag");
    if (fingerTag) {
        if (data.finger_used === "LEFT_THUMB") {
            fingerTag.innerText = "🖐 Acquired: LEFT THUMB";
            fingerTag.style.cssText = "display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:700;background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;margin-left:4px;";
        } else {
            fingerTag.style.display = "none";
        }
    }

    // ---- Quality & Placement Guide Widget Update ----
    const qGuide = document.getElementById("quality-placement-guide");
    if (qm && qGuide) {
        const qScore = qm.quality || 75;
        const qBadge = document.getElementById("quality-score-badge");
        const qBar = document.getElementById("quality-progress-bar");
        const qPress = document.getElementById("quality-pressure-val");
        const qNfiq = document.getElementById("quality-nfiq-val");
        const qAdvice = document.getElementById("quality-advice-val");

        if (qBadge) qBadge.innerText = `Quality: ${qScore}/100`;
        if (qBar) {
            qBar.style.width = `${qScore}%`;
            if (qScore >= 70) {
                qBar.style.background = "#22c55e";
                if (qBadge) { qBadge.style.background = "#dcfce7"; qBadge.style.color = "#15803d"; }
            } else if (qScore >= 50) {
                qBar.style.background = "#eab308";
                if (qBadge) { qBadge.style.background = "#fef9c3"; qBadge.style.color = "#854d0e"; }
            } else {
                qBar.style.background = "#ef4444";
                if (qBadge) { qBadge.style.background = "#fee2e2"; qBadge.style.color = "#991b1b"; }
            }
        }
        if (qPress) {
            qPress.innerText = qm.pressure_status || "OPTIMAL";
            qPress.style.color = (qm.pressure_status === "OPTIMAL") ? "#16a34a" : "#d97706";
        }
        if (qNfiq) {
            const nfiqNames = { 1: "1 (Excellent)", 2: "2 (Very Good)", 3: "3 (Good)", 4: "4 (Fair)", 5: "5 (Poor)" };
            qNfiq.innerText = nfiqNames[qm.nfiq] || `${qm.nfiq}`;
        }
        if (qAdvice) qAdvice.innerText = qm.placement_advice || "";
        qGuide.style.display = "block";
    }

    // ---- 3-Scan Consensus Breakdown ----
    const conCard = document.getElementById("consensus-breakdown-card");
    if (conCard) {
        const conMeta = data.consensus_metadata;
        if (conMeta && conMeta.individual_scans) {
            // Update card title to reflect Left Thumb protocol if applicable
            const conTitle = document.getElementById("consensus-card-title");
            if (conTitle) {
                conTitle.innerText = data.finger_used === "LEFT_THUMB"
                    ? "🖐 3-Scan Left Thumb Protocol - Consensus Breakdown"
                    : "🎯 3-Impression Consensus Breakdown";
            }
            const badge = document.getElementById("consensus-agreement-badge");
            if (badge) {
                badge.innerText = `${conMeta.agreement_count}/${conMeta.num_scans} (${conMeta.agreement_pct}% Agreement)`;
                badge.style.background = (conMeta.agreement_pct >= 66) ? "#22c55e" : "#eab308";
            }
            const listEl = document.getElementById("consensus-scans-list");
            if (listEl) {
                listEl.innerHTML = conMeta.individual_scans.map(s => `
                    <div style="background:#fff; border:1px solid #bbf7d0; border-radius:6px; padding:8px; text-align:center;">
                        <div style="font-weight:600; color:#475569; font-size:0.75rem;">Impression #${s.scan_index}</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#047857; margin:3px 0;">${s.blood_group}</div>
                        <div style="font-size:0.75rem; color:#64748b;">${(s.confidence * 100).toFixed(1)}% conf</div>
                    </div>
                `).join("");
            }
            conCard.style.display = "block";
        } else {
            conCard.style.display = "none";
        }
    }

    // Download PDF Link
    const pdfBtn = document.getElementById("btn-download-pdf");
    if (data.report_url) {
        pdfBtn.href = data.report_url;
        pdfBtn.style.display = "inline-flex";
    }

    // Smooth scroll to result
    resultContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function enableAndTriggerConsensus() {
    const chk = document.getElementById("chk-consensus-mode");
    if (chk) {
        chk.checked = true;
    }
    const suggestion = document.getElementById("smart-consensus-suggestion");
    if (suggestion) {
        suggestion.style.display = "none";
    }
    triggerCapture();
}

function resetView() {
    document.getElementById("result-placeholder").style.display = "block";
    document.getElementById("result-content").style.display = "none";
    document.getElementById("file-name-preview").innerText = "";
    document.getElementById("file-input").value = "";
    document.getElementById("btn-upload-predict").disabled = true;
    const smartConsensus = document.getElementById("smart-consensus-suggestion");
    if (smartConsensus) smartConsensus.style.display = "none";

    const fingerTag = document.getElementById("result-finger-tag");
    if (fingerTag) fingerTag.style.display = "none";
    selectedFile = null;
}
