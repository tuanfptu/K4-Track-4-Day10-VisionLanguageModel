import json

from student.common import ROOT, SPLITS_DIR, read_jsonl
from student.run_lab import require_gpu, run_e5
from student.reporting import generate_report

summary_path = ROOT / "work" / "experiment_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
best_name = summary["best_sft"]
best = next(
    result for result in summary["sft_results"]
    if result["experiment"] == best_name
)

train_rows = read_jsonl(SPLITS_DIR / "train.jsonl")
val_rows = read_jsonl(SPLITS_DIR / "val.jsonl")
test_rows = read_jsonl(SPLITS_DIR / "test.jsonl")

require_gpu()
print(f"Chạy E5 từ mô hình {best_name.upper()}", flush=True)
run_e5(best, train_rows, val_rows, test_rows, batch_size=2, grad_accum=4)
generate_report()
