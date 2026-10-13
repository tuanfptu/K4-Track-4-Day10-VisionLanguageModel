"""End-to-end runner for E0-E5. Run from the repository root."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from student.common import (
    DATA_DIR, EVIDENCE_DIR, MODEL_ID, PRED_DIR, ROOT, RUNS_DIR, SPLITS_DIR,
    evaluate_predictions, load_adapter, load_base_model, predict_rows, read_jsonl,
    save_evidence, save_projector, train_sft, write_jsonl,
)


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy CUDA GPU. Hãy chạy trong PyTorch Docker có gắn RTX 3090.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def _model_architecture(model):
    groups = {"vision_tower": 0, "projector": 0, "llm": 0, "other": 0}
    for name, param in model.named_parameters():
        if "visual.merger." in name or "visual.deepstack_merger_list." in name:
            groups["projector"] += param.numel()
        elif "visual." in name:
            groups["vision_tower"] += param.numel()
        elif "language_model." in name:
            groups["llm"] += param.numel()
        else:
            groups["other"] += param.numel()
    groups["total"] = sum(groups.values())
    path = ROOT / "work" / "architecture.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_id": MODEL_ID, "parameters": groups}, indent=2), encoding="utf-8")
    return groups


def run_e0(batch_size: int):
    require_gpu()
    train_rows = read_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = read_jsonl(SPLITS_DIR / "val.jsonl")
    test_rows = read_jsonl(SPLITS_DIR / "test.jsonl")
    model, processor = load_base_model(training=False)
    architecture = _model_architecture(model)
    print("Tham số gốc:", json.dumps(architecture, ensure_ascii=False))
    for split, rows in (("val", val_rows), ("test", test_rows)):
        predictions = predict_rows(model, processor, rows, batch_size=batch_size, split_name=f"E0/{split}")
        pred_path = PRED_DIR / f"e0_{split}.jsonl"
        write_jsonl(pred_path, predictions)
        metrics = evaluate_predictions(rows, predictions, "E0", split)
        if split == "test":
            save_evidence(metrics, EVIDENCE_DIR / "e0.json")
        print(f"E0 {split}: F1={metrics['micro']['f1']:.4f}, valid={metrics['parse_valid_rate']:.3f}")
    del model, processor
    torch.cuda.empty_cache()


def run_sft_experiment(experiment: str, rows_train, rows_val, rows_test,
                       batch_size: int, grad_accum: int, max_steps: int = 300,
                       train_fraction: float = 1.0):
    run_dir = RUNS_DIR / experiment
    actual_rows = list(rows_train)
    if train_fraction < 1.0:
        rng = random.Random(42)
        rng.shuffle(actual_rows)
        actual_rows = actual_rows[:max(1, int(round(len(actual_rows) * train_fraction)))]
        actual_rows.sort(key=lambda x: x["image_id"])
    metadata = train_sft(
        experiment, actual_rows, run_dir, max_steps=max_steps,
        batch_size=batch_size, grad_accum=grad_accum,
    )
    model, processor = load_adapter(experiment, run_dir, training=False)
    val_pred = predict_rows(model, processor, rows_val, batch_size=batch_size, split_name=f"{experiment}/val")
    test_pred = predict_rows(model, processor, rows_test, batch_size=batch_size, split_name=f"{experiment}/test")
    write_jsonl(PRED_DIR / f"{experiment}_val.jsonl", val_pred)
    write_jsonl(PRED_DIR / f"{experiment}_test.jsonl", test_pred)
    val_metrics = evaluate_predictions(rows_val, val_pred, experiment.upper(), "val", meta=metadata)
    test_metrics = evaluate_predictions(rows_test, test_pred, experiment.upper(), "test", meta=metadata)
    save_evidence(test_metrics, EVIDENCE_DIR / f"{experiment}.json")
    print(f"{experiment.upper()}: val F1={val_metrics['micro']['f1']:.4f}; test F1={test_metrics['micro']['f1']:.4f}")
    del model, processor
    torch.cuda.empty_cache()
    return {"experiment": experiment, "run_dir": str(run_dir), "metadata": metadata,
            "val": val_metrics, "test": test_metrics}


def run_e4(best: dict, rows_train, rows_val, rows_test, batch_size: int, grad_accum: int):
    """Compare full data to 10%, keeping the selected E1/E2/E3 setup fixed."""
    full_meta = best["metadata"]
    full_pred = read_jsonl(PRED_DIR / f"{best['experiment']}_test.jsonl")
    full_metrics = evaluate_predictions(
        rows_test, full_pred, f"E4 data_size=1.0 ({best['experiment'].upper()})", "test", meta=full_meta
    )
    save_evidence(full_metrics, EVIDENCE_DIR / "e4_data_size_1.0.json")
    ablation_dir = RUNS_DIR / "e4_data_size_0.1"
    subset = list(rows_train)
    random.Random(42).shuffle(subset)
    subset = sorted(subset[:max(1, round(len(subset) * 0.1))], key=lambda x: x["image_id"])
    meta = train_sft(
        best["experiment"], subset, ablation_dir, rank=16, max_steps=300,
        batch_size=batch_size, grad_accum=grad_accum, seed=42, learning_rate=2e-4,
    )
    model, processor = load_adapter(best["experiment"], ablation_dir, training=False)
    val_pred = predict_rows(model, processor, rows_val, batch_size=batch_size, split_name="E4/val")
    test_pred = predict_rows(model, processor, rows_test, batch_size=batch_size, split_name="E4/test")
    write_jsonl(PRED_DIR / "e4_data_size_0.1_val.jsonl", val_pred)
    write_jsonl(PRED_DIR / "e4_data_size_0.1_test.jsonl", test_pred)
    val_metrics = evaluate_predictions(rows_val, val_pred, "E4 data_size=0.1", "val", meta=meta)
    test_metrics = evaluate_predictions(rows_test, test_pred, "E4 data_size=0.1", "test", meta=meta)
    save_evidence(test_metrics, EVIDENCE_DIR / "e4_data_size_0.1.json")
    del model, processor
    torch.cuda.empty_cache()
    print(f"E4 data_size 1.0 F1={full_metrics['micro']['f1']:.4f}; 0.1 F1={test_metrics['micro']['f1']:.4f}")
    return {"full": full_metrics, "small": test_metrics, "small_val": val_metrics}


def run_e5(best: dict, rows_train, rows_val, rows_test,
           batch_size: int, grad_accum: int):
    """Run the required optional GRPO bonus with the best validation SFT adapter."""
    try:
        from datasets import Dataset, Image as DatasetImage
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("Thiếu datasets hoặc TRL; cài requirements-colab.txt (trl==1.9.2).") from exc
    from student.common import PROMPT, gt_boxes, image_path
    from student.reward import iou_reward

    require_gpu()
    model, processor = load_adapter(best["experiment"], Path(best["run_dir"]), training=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    grpo_rows = []
    for row in rows_train:
        prompt = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]}]
        grpo_rows.append({
            "prompt": prompt,
            "image": str(image_path(row)),
            "gt_boxes_json": json.dumps(gt_boxes(row), separators=(",", ":")),
        })
    dataset = Dataset.from_list(grpo_rows).cast_column("image", DatasetImage())
    run_dir = RUNS_DIR / "e5_grpo"
    run_dir.mkdir(parents=True, exist_ok=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    cfg = GRPOConfig(
        output_dir=str(run_dir / "trainer"), learning_rate=5e-6, beta=0.0,
        num_generations=4, max_completion_length=128, max_steps=60,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        fp16=False, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_steps=30, save_strategy="steps", report_to="none",
        remove_unused_columns=False, use_vllm=False, seed=42,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=iou_reward, args=cfg,
        train_dataset=dataset, processing_class=processor,
    )
    train_result = trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    extra = {name: p for name, p in trainer.model.named_parameters()
             if "visual.merger." in name or "visual.deepstack_merger_list." in name}
    save_projector(run_dir / "extra_trainable.pt", extra)
    elapsed = (time.time() - started) / 60
    metadata = {
        "experiment": "E5 GRPO", "initialized_from": best["experiment"].upper(),
        "trainable_params": trainable_params,
        "peak_vram_gb": max(torch.cuda.max_memory_allocated() / 1024**3, 1e-6),
        "train_minutes": elapsed, "max_steps": 60, "num_generations": 4,
        "learning_rate": 5e-6, "beta": 0.0, "effective_batch_size": batch_size * grad_accum,
        "train_metrics": {k: float(v) for k, v in train_result.metrics.items()
                          if isinstance(v, (int, float))},
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    del trainer, model, processor
    torch.cuda.empty_cache()
    model, processor = load_adapter("e5", run_dir, training=False)
    for split, rows in (("val", rows_val), ("test", rows_test)):
        preds = predict_rows(model, processor, rows, batch_size=batch_size, split_name=f"E5/{split}")
        write_jsonl(PRED_DIR / f"e5_{split}.jsonl", preds)
        metrics = evaluate_predictions(rows, preds, "E5 GRPO", split, meta=metadata)
        if split == "test":
            save_evidence(metrics, EVIDENCE_DIR / "e5.json")
        print(f"E5 {split}: F1={metrics['micro']['f1']:.4f}")
    del model, processor
    torch.cuda.empty_cache()
    return metadata


def run_all(args):
    if not (DATA_DIR / "splits" / "train.jsonl").exists():
        raise FileNotFoundError("Chưa thấy data/splits/train.jsonl. Chạy: python scripts/prepare_data.py --zip archive.zip")
    require_gpu()
    rows_train = read_jsonl(SPLITS_DIR / "train.jsonl")
    rows_val = read_jsonl(SPLITS_DIR / "val.jsonl")
    rows_test = read_jsonl(SPLITS_DIR / "test.jsonl")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    run_e0(args.eval_batch_size)
    results = []
    for exp in ("e1", "e2", "e3"):
        results.append(run_sft_experiment(
            exp, rows_train, rows_val, rows_test,
            args.train_batch_size, args.grad_accum,
        ))
    best = max(results, key=lambda r: r["val"]["micro"]["f1"])
    summary = {"selected_by": "highest validation micro F1", "best_sft": best["experiment"],
               "sft_results": results}
    (ROOT / "work" / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Setup được chọn bằng val: {best['experiment'].upper()} (F1={best['val']['micro']['f1']:.4f})")
    run_e4(best, rows_train, rows_val, rows_test, args.train_batch_size, args.grad_accum)
    if not args.skip_e5:
        run_e5(best, rows_train, rows_val, rows_test, args.train_batch_size, args.grad_accum)
    from student.reporting import generate_report
    generate_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["all", "e0", "report"], default="all")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--skip-e5", action="store_true", help="Bỏ GRPO E5; mặc định chạy đủ E0-E5.")
    args = parser.parse_args()
    if args.command == "report":
        from student.reporting import generate_report
        generate_report()
    elif args.command == "e0":
        run_e0(args.eval_batch_size)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
