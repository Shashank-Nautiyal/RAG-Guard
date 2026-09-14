import csv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "final_comparison.csv"
rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# 1. Baseline vs Improved Correctness
def is_correct(val):
    return "correct" in val.lower() and "incorrect" not in val.lower()

b_correct = sum(1 for r in rows if is_correct(r["baseline_result"]))
b_incorrect = sum(1 for r in rows if not is_correct(r["baseline_result"]))
imp_correct = sum(1 for r in rows if is_correct(r["improved_result"]))
imp_incorrect = sum(1 for r in rows if not is_correct(r["improved_result"]))

fig, ax = plt.subplots(figsize=(7, 5))
categories = ["Baseline", "Improved"]
x = range(len(categories))
width = 0.35

rects1 = ax.bar([i - width/2 for i in x], [b_correct, imp_correct], width, label="Correct", color="#2ecc71")
rects2 = ax.bar([i + width/2 for i in x], [b_incorrect, imp_incorrect], width, label="Incorrect", color="#e74c3c")

ax.set_ylabel("Number of Questions")
ax.set_title("Baseline vs Improved: Answer Correctness")
ax.set_xticks(list(x))
ax.set_xticklabels(categories)
ax.set_ylim(0, 26)
ax.legend()

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "baseline_vs_improved_correctness.png", dpi=300)
plt.close()

# 2. Faithfulness Violations
def is_violation(val):
    val_upper = val.upper()
    return "UNGROUNDED" in val_upper or "MIXED" in val_upper

b_violations = sum(1 for r in rows if is_violation(r["baseline_grounding"]))
imp_violations = sum(1 for r in rows if is_violation(r["improved_grounding"]))

fig, ax = plt.subplots(figsize=(6, 5))
models = ["Baseline", "Improved"]
bars = ax.bar(models, [b_violations, imp_violations], color=["#e67e22", "#3498db"], width=0.45)
ax.set_ylabel("Count of Violations")
ax.set_title("Faithfulness Violations (UNGROUNDED / MIXED Grounding)")
ax.set_ylim(0, max(b_violations, imp_violations) + 2)

for bar in bars:
    h = bar.get_height()
    ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "faithfulness_violations.png", dpi=300)
plt.close()
print("Plots regenerated successfully.")
