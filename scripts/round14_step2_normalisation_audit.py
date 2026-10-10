"""
round14_step2_normalisation_audit.py
Round 14, Step 2: For each prior evaluation, state whether it used preprocess_input.

BACKGROUND:
- The E2 training script (scratch/experiment6_fusion_supcon.py) defines its own
  transform_c5() at lines 45-51 which calls src.preprocessing functions
  (composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast,
  resize_to_target) but then does: np.array(resized, dtype=np.float32) — NO preprocess_input.
- The deployed src/preprocessing.py transform_c5() adds densenet.preprocess_input at line 269.
- Therefore: any evaluation that went through src.preprocessing.transform_c5() used
  preprocess_input (MISMATCHED), while those using the local copy did NOT.

This script prints each evaluation's normalisation path and verdict.
"""

print("""
=== ROUND 14 STEP 2: WHICH EVALUATIONS USED preprocess_input? ===

KEY FACTS:
- Training used: transform_c5() LOCAL in scratch/experiment6_fusion_supcon.py
  = composite_alpha + crop + CLAHE + BILINEAR resize + raw float32 [0,255]. NO preprocess_input.
- Deployed path uses: transform_c5() from src/preprocessing.py
  = same steps + densenet.preprocess_input (shifts to ~[-2.1, 2.6]).
- This was introduced in the ONLY git commit to src/preprocessing.py (c1aac08),
  so it was present in the deployed path from initial commit, before final_evaluation_e2.py
  was run (which DID import from src.preprocessing).

EVALUATION-BY-EVALUATION AUDIT:
""")

evaluations = [
    {
        "round": "Round 13 reproduction (scripts/round13_reproduce_audit.py)",
        "description": "Reproduced 21 locked test (8/21=38.10%) and 80 dev folds",
        "import_path": "scratch/experiment6_fusion_supcon.py local transform_c5 (was the explicit aim)",
        "used_preprocess_input": "UNKNOWN – need to check script",
        "verdict": "SEE BELOW"
    },
    {
        "round": "scratch/final_evaluation_e2.py (wrote FINAL_TEST_REPORT.md / phase3_final_evaluation/)",
        "description": "21 locked test => 8/21 = 38.10%",
        "import_path": "Line 12: from src.preprocessing import ... resize_to_target; local transform_c5 at L30-36 calls resize_to_target then np.array(resized, dtype=np.float32) — NO preprocess_input",
        "used_preprocess_input": False,
        "verdict": "VALID (matches training)"
    },
    {
        "round": "scratch/experiment6_fusion_supcon.py (training script)",
        "description": "Trained 5 folds on 80 dev participants",
        "import_path": "Line 13: from src.preprocessing import ... resize_to_target; local transform_c5 at L45-51 calls resize_to_target then np.array(resized, dtype=np.float32) — NO preprocess_input",
        "used_preprocess_input": False,
        "verdict": "VALID (this IS training)"
    },
    {
        "round": "Round 9 sensitivity analysis (101 participants, 4 variants)",
        "description": "Tested 101 participants via scripts/live_sensitivity_analysis.py or similar",
        "import_path": "Used src.preprocessing.transform_c5() which has densenet.preprocess_input at line 269",
        "used_preprocess_input": True,
        "verdict": "INVALID: normalisation mismatch – preprocess_input applied; scores reflect mismatched pipeline"
    },
    {
        "round": "Round 12 probes (scripts/round12_live_101_eval.py)",
        "description": "101 participant eval (27.72% reported)",
        "import_path": "Used src.preprocessing.transform_c5() which has densenet.preprocess_input",
        "used_preprocess_input": True,
        "verdict": "INVALID: normalisation mismatch – preprocess_input applied; 27.72% is a collapsed-pipeline number"
    },
    {
        "round": "Round 12 class-order probes (scripts/round12_class_order_and_triplet_probe.py)",
        "description": "Class order / triplet exploration on 101 participants",
        "import_path": "Used src.preprocessing.transform_c5() — preprocess_input present",
        "used_preprocess_input": True,
        "verdict": "INVALID: normalisation mismatch"
    },
    {
        "round": "Round 11 reproduce check (scripts/round11_reproduce_check.py)",
        "description": "Attempted reproduction; reported accuracy figures",
        "import_path": "Used src.preprocessing.transform_c5() — preprocess_input present",
        "used_preprocess_input": True,
        "verdict": "INVALID: normalisation mismatch"
    },
    {
        "round": "Deployed web app /api/upload and /api/capture paths",
        "description": "All live predictions via src/predictor.py -> src/preprocessing.transform_c5()",
        "import_path": "src.preprocessing.transform_c5() -- preprocess_input present",
        "used_preprocess_input": True,
        "verdict": "INVALID: normalisation mismatch vs training"
    },
]

for e in evaluations:
    print(f"  Evaluation: {e['round']}")
    print(f"    Description: {e['description']}")
    print(f"    Normalisation: used preprocess_input = {e['used_preprocess_input']}")
    print(f"    Verdict: {e['verdict']}")
    print()

print("""
CONTRADICTION FROM ROUND 13 SUMMARY:
- Round 13 summary claimed experiment6 used raw [0,255] bilinear-only (NO CLAHE, NO crop).
- ACTUAL CODE: experiment6_fusion_supcon.py calls crop_mfs100_foreground and
  enhance_mfs100_ridge_contrast from src.preprocessing. It applies CLAHE and crop.
  The difference from deployed path is ONLY the final densenet.preprocess_input step.
- Therefore Round 13 Step 3 description was partially wrong: 
  the divergence is NOT "crop vs no-crop" or "CLAHE vs no-CLAHE".
  The divergence is ONLY the preprocess_input step.
- Also: final_evaluation_e2.py DID import from src.preprocessing (line 12), but defines
  its OWN local transform_c5 that calls those imported helpers WITHOUT preprocess_input.
  This local function shadows any transform_c5 that might be imported.
""")
