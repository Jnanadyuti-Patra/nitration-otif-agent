"""Score the agent against the four root causes hidden in the simulator.

python -m otifagent.eval   ->  results/eval.json
"""
import json
import pathlib
import re

from .agent import ask

HIDDEN = {
    "R-103 jacket fouling": r"R-103|foul",
    "summer cooling water": r"cooling water|summer|\bCW\b",
    "Supplier B moisture / rework": r"Supplier B",
    "night-shift QC delay": r"night",
}
QUESTIONS = [
    "Why did our export OTIF drop from April, and what is it costing us?",
    "Our customers say shipments from Dahej are late since summer. What are the root causes and what should we fix first?",
    "Where is lead time being lost in the plant, and how much money is at stake?",
]


def score(result: dict) -> dict:
    text = json.dumps(result["diagnosis"])
    return {k: bool(re.search(p, text, re.I)) for k, p in HIDDEN.items()}


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    runs = []
    for q in QUESTIONS:
        r = ask(q)
        r["found"] = score(r)
        runs.append(r)
        print(q, "->", sum(r["found"].values()), "/ 4 in", r["seconds"], "s", flush=True)
    recall = sum(sum(r["found"].values()) for r in runs) / (4 * len(runs))
    summary = {"questions": len(runs), "root_cause_recall": round(recall, 2),
               "grounded_runs": sum(not r["unsupported_numbers_left"] for r in runs),
               "mean_seconds": round(sum(r["seconds"] for r in runs) / len(runs), 1), "runs": runs}
    (out / "eval.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("recall", summary["root_cause_recall"], "grounded", summary["grounded_runs"], "/", len(runs))
