"""Shared Qwen3-VL data, model, training, inference, and scoring helpers."""
from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RUNS_DIR = ROOT / "work" / "runs"
EVIDENCE_DIR = ROOT / "report" / "evidence"
PRED_DIR = ROOT / "work" / "predictions"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
CLASSES = ("pothole", "crack", "manhole")
PROMPT = (
    "Detect every pothole, crack, and manhole in this road image. "
    "Return only a JSON list. Each item must have exactly the keys "
    "label and bbox_2d; label must be pothole, crack, or manhole. "
    "bbox_2d is [x1,y1,x2,y2] on the normalized 0–1000 coordinate scale, "
    "where 0 and 1000 are the image edges. Use [] when none are present."
)
PROMPT_PIXEL = PROMPT.replace("0–1000 coordinate scale", "pixel coordinate scale")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_path(entry: dict[str, Any]) -> Path:
    path = Path(entry["image_path"])
    return path if path.is_absolute() else DATA_DIR / path


def load_image(entry: dict[str, Any]) -> Image.Image:
    with Image.open(image_path(entry)) as im:
        return im.convert("RGB")


def gt_boxes(entry: dict[str, Any], scale: str = "norm1000") -> list[dict[str, Any]]:
    w_img, h_img = entry["width"], entry["height"]
    scale_factor_x = 1000 if scale == "norm1000" else w_img
    scale_factor_y = 1000 if scale == "norm1000" else h_img
    out = []
    for obj in entry["objects"]:
        xc, yc, w, h = obj["yolo"]
        box = [
            round((xc - w / 2) * scale_factor_x),
            round((yc - h / 2) * scale_factor_y),
            round((xc + w / 2) * scale_factor_x),
            round((yc + h / 2) * scale_factor_y),
        ]
        lim_x, lim_y = scale_factor_x, scale_factor_y
        box = [max(0, min(int(lim_x if i % 2 == 0 else lim_y), v)) for i, v in enumerate(box)]
        out.append({"label": obj["label"], "bbox_2d": box})
    return sorted(out, key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))


def answer_json(entry: dict[str, Any], scale: str = "norm1000") -> str:
    return json.dumps(gt_boxes(entry, scale), separators=(",", ":"), ensure_ascii=False)


def chat_messages(image: Image.Image, prompt: str = PROMPT) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]


def load_base_model(training: bool = False):
    """Load 4-bit LLM while keeping visual modules and lm_head unquantized."""
    try:
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Cần transformers==5.19.0; cài requirements-colab.txt trước.") from exc

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(torch.bfloat16 if training else torch.float16),
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    compute_dtype = torch.bfloat16 if training else torch.float16
    kwargs = {"quantization_config": quant, "dtype": compute_dtype, "device_map": {"": 0}}
    token = None
    try:
        import os
        token = os.environ.get("HF_TOKEN")
    except Exception:
        pass
    if token:
        kwargs["token"] = token
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    processor.tokenizer.padding_side = "right" if training else "left"
    model.config.use_cache = not training
    return model, processor


def setup_lora(model, experiment: str, rank: int = 16):
    """Freeze the base, add LoRA to required projections, and optionally unfreeze projector."""
    from peft import LoraConfig, get_peft_model

    for param in model.parameters():
        param.requires_grad_(False)
    llm = r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
    vit = r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"
    target_pattern = llm + ("|" + vit if experiment == "e3" else "")
    matched = [name for name, _ in model.named_modules() if re.fullmatch(target_pattern, name)]
    expected = 348 if experiment == "e3" else 252
    if len(matched) != expected:
        raise RuntimeError(
            f"LoRA target count sai: tìm thấy {len(matched)}, cần {expected}. "
            "Kiểm tra cây module Qwen3-VL trong phiên bản transformers đã cài."
        )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=target_pattern, bias="none",
    )
    wrapped = get_peft_model(model, cfg)
    projector_state = {}
    if experiment in {"e2", "e3"}:
        for name, param in wrapped.named_parameters():
            if "visual.merger." in name or "visual.deepstack_merger_list." in name:
                param.data = param.data.float()
                param.requires_grad_(True)
                projector_state[name] = param
    return wrapped, matched, projector_state


def _encode(processor, entry: dict[str, Any], scale: str = "norm1000"):
    image = load_image(entry)
    prompt = PROMPT if scale == "norm1000" else PROMPT_PIXEL
    user = chat_messages(image, prompt)
    full = user + [{"role": "assistant", "content": answer_json(entry, scale)}]
    prompt_inputs = processor.apply_chat_template(
        user, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    full_inputs = processor.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    n_prompt = prompt_inputs["input_ids"].shape[1]
    prefix = full_inputs["input_ids"][:, :n_prompt]
    if prefix.shape != prompt_inputs["input_ids"].shape or not torch.equal(prefix, prompt_inputs["input_ids"]):
        raise RuntimeError(f"Prompt không phải tiền tố chuỗi đầy đủ: {entry['image_id']}")
    full_inputs["labels"] = full_inputs["input_ids"].clone()
    full_inputs["labels"][:, :n_prompt] = -100
    return {k: v.squeeze(0) if torch.is_tensor(v) and v.ndim >= 2
            and k not in {"pixel_values", "image_grid_thw"} else v
            for k, v in full_inputs.items()}


class SFTDataset(Dataset):
    def __init__(self, rows, processor, scale="norm1000"):
        self.rows, self.processor, self.scale = rows, processor, scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return _encode(self.processor, self.rows[idx], self.scale)


def collate_sft(features, pad_token_id: int):
    batch = {}
    max_len = max(x["input_ids"].shape[-1] for x in features)
    for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
        if not all(key in x for x in features):
            continue
        pad = -100 if key == "labels" else (pad_token_id if key == "input_ids" else 0)
        batch[key] = torch.nn.utils.rnn.pad_sequence([x[key] for x in features], batch_first=True, padding_value=pad)
    for key in ("pixel_values", "image_grid_thw"):
        vals = [x[key] for x in features if key in x]
        if vals:
            batch[key] = torch.cat(vals, dim=0)
    return batch


def save_projector(path: Path, projector_state: dict[str, torch.Tensor]) -> None:
    if projector_state:
        torch.save({k: v.detach().float().cpu() for k, v in projector_state.items()}, path)


def load_adapter(experiment: str, run_dir: Path, training: bool = False):
    from peft import PeftModel
    model, processor = load_base_model(training=training)
    model.config.use_cache = not training
    model = PeftModel.from_pretrained(model, str(run_dir / "adapter"), is_trainable=training)
    extra_path = run_dir / "extra_trainable.pt"
    if extra_path.exists():
        state = torch.load(extra_path, map_location="cpu", weights_only=True)
        params = dict(model.named_parameters())
        missing = set(state) - set(params)
        if missing:
            raise RuntimeError(f"Không tìm thấy projector khi nạp: {sorted(missing)[:3]}")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))
    return model, processor


@torch.inference_mode()
def predict_rows(model, processor, rows: list[dict[str, Any]], batch_size: int = 4,
                 max_new_tokens: int = 256, split_name: str = "predictions") -> list[dict[str, str]]:
    model.eval()
    processor.tokenizer.padding_side = "left"
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        images = [load_image(r) for r in chunk]
        messages = [chat_messages(im) for im in images]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        out.extend({"image_id": row["image_id"], "raw": text.strip()} for row, text in zip(chunk, texts))
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(rows):
            print(f"[{split_name}] {min(start + batch_size, len(rows))}/{len(rows)}")
    return out


def evaluate_predictions(rows, predictions, name: str, split: str, coord: str = "norm1000", meta=None):
    from scripts.score import SCORE_VERSION, evaluate
    mapping = {r["image_id"]: r["raw"] for r in predictions}
    metrics = evaluate(rows, mapping, coord=coord)
    meta_data = json.loads((SPLITS_DIR / "meta.json").read_text(encoding="utf-8"))
    metrics.update({
        "score_version": SCORE_VERSION, "name": name,
        "dataset_version": meta_data["dataset_version"], "split": split,
        "coord": coord, "meta": meta or {},
    })
    return metrics


def save_evidence(metrics, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def train_sft(experiment: str, rows: list[dict[str, Any]], run_dir: Path,
              rank: int = 16, max_steps: int = 300, batch_size: int = 2,
              grad_accum: int = 4, seed: int = 42, learning_rate: float = 2e-4,
              scale: str = "norm1000") -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_base_model(training=True)
    model, matched, projector_state = setup_lora(model, experiment, rank=rank)
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    if trainable_params <= 0:
        raise RuntimeError("Không có tham số nào được mở để train.")
    dataset = SFTDataset(rows, processor, scale=scale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        collate_fn=lambda batch: collate_sft(batch, processor.tokenizer.pad_token_id))
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    warmup = max(1, int(max_steps * 0.05))
    def lr_lambda(step):
        if step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # BF16 không cần GradScaler.
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    losses, log_path = [], run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    model.train()
    for step in range(1, max_steps + 1):
        aggregate_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss không hữu hạn ở bước {step}: {loss.item()}")
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            aggregate_loss += float(loss.detach()) / grad_accum
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        losses.append(aggregate_loss)
        if step % 10 == 0 or step == 1:
            record = {"step": step, "loss": aggregate_loss, "lr": scheduler.get_last_lr()[0],
                      "elapsed_minutes": (time.time() - started) / 60}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[{experiment}] step {step}/{max_steps} loss={aggregate_loss:.4f} "
                  f"lr={record['lr']:.2e} elapsed={record['elapsed_minutes']:.1f}m")
        if step % 100 == 0 or step == max_steps:
            ckpt = run_dir / f"checkpoint-{step}"
            (ckpt / "adapter").mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt / "adapter")
            save_projector(ckpt / "extra_trainable.pt", projector_state)
            (ckpt / "step.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    (run_dir / "adapter").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir / "adapter")
    save_projector(run_dir / "extra_trainable.pt", projector_state)
    elapsed = (time.time() - started) / 60
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    metadata = {
        "experiment": experiment, "model_id": MODEL_ID, "dataset_version": json.loads(
            (SPLITS_DIR / "meta.json").read_text(encoding="utf-8"))["dataset_version"],
        "trainable_params": trainable_params, "matched_lora_modules": len(matched),
        "peak_vram_gb": max(peak_gb, 1e-6), "train_minutes": elapsed,
        "max_steps": max_steps, "batch_size": batch_size, "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum, "rank": rank,
        "learning_rate": learning_rate, "seed": seed, "coord_scale": scale,
        "mean_last_20_loss": sum(losses[-20:]) / max(1, len(losses[-20:])),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    del model, processor, optimizer, loader
    torch.cuda.empty_cache()
    return metadata
