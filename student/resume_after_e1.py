import json
from student.common import (
    ROOT, SPLITS_DIR, RUNS_DIR, PRED_DIR,
    read_jsonl, evaluate_predictions,
)
from student.run_lab import require_gpu, run_sft_experiment, run_e4, run_e5
from student.reporting import generate_report

require_gpu()
train_rows = read_jsonl(SPLITS_DIR / "train.jsonl")
val_rows = read_jsonl(SPLITS_DIR / "val.jsonl")
test_rows = read_jsonl(SPLITS_DIR / "test.jsonl")

e1_meta = json.loads((RUNS_DIR / "e1" / "run.json").read_text(encoding="utf-8"))
e1 = {
    "experiment": "e1",
    "run_dir": str(RUNS_DIR / "e1"),
    "metadata": e1_meta,
    "val": evaluate_predictions(
        val_rows, read_jsonl(PRED_DIR / "e1_val.jsonl"),
        "E1", "val", meta=e1_meta,
    ),
    "test": evaluate_predictions(
        test_rows, read_jsonl(PRED_DIR / "e1_test.jsonl"),
        "E1", "test", meta=e1_meta,
    ),
}
results = [e1]

for exp in ("e2", "e3"):
    print(f"Tiếp tục {exp.upper()}", flush=True)
    results.append(run_sft_experiment(
        exp, train_rows, val_rows, test_rows,
        batch_size=2, grad_accum=4,
    ))

best = max(results, key=lambda r: r["val"]["micro"]["f1"])
summary = {
    "selected_by": "highest validation micro F1",
    "best_sft": best["experiment"],
    "sft_results": results,
}
summary_path = ROOT / "work" / "experiment_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Setup tốt nhất: {best['experiment'].upper()}", flush=True)

run_e4(best, train_rows, val_rows, test_rows, batch_size=2, grad_accum=4)
run_e5(best, train_rows, val_rows, test_rows, batch_size=2, grad_accum=4)
generate_report()
