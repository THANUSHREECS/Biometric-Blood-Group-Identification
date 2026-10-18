import pandas as pd
from pathlib import Path

p = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project/reports/efficientnet_e2_ensemble_experiment/participant_disagreement_analysis.csv")
df = pd.read_csv(p)

for cohort in ["DEVELOPMENT_80", "LOCKED_TEST_21", "SEPARATE_26"]:
    sub = df[df["cohort"] == cohort]
    print(f"\nCohort: {cohort} (N={len(sub)})")
    print("  Models agree:", sub["models_agree"].sum())
    print("  Models disagree:", (~sub["models_agree"]).sum())
    print("  Ensemble corrects EffNet:", sub["ensemble_corrects_effnet"].sum())
    print("  Ensemble corrects E2:", sub["ensemble_corrects_e2"].sum())
    print("  Ensemble introduces error (either model correct, but ensemble wrong):", sub["ensemble_introduces_error"].sum())
    print("  EffNet correct:", sub["effnet_correct"].sum())
    print("  E2 correct:", sub["e2_correct"].sum())
    print("  Ensemble correct:", sub["ensemble_correct"].sum())
