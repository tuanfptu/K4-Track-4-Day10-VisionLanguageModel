"""Create evidence-backed report answers and annotated validation examples."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.score import evaluate, iou_xyxy, parse_output
from student.common import (
    DATA_DIR, EVIDENCE_DIR, PRED_DIR, ROOT, SPLITS_DIR, gt_boxes, image_path,
    read_jsonl,
)


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(evidence, cls=None):
    if evidence is None:
        return None
    return evidence["micro"] if cls is None else evidence.get("per_class", {}).get(cls)


def _f1(e):
    return "—" if e is None else f"{e['micro']['f1']:.3f}"


def _ci(e):
    return "—" if e is None else f"[{e['f1_ci95'][0]:.3f}, {e['f1_ci95'][1]:.3f}]"


def _stats(e):
    if e is None:
        return "—", "—", "—", "—", "—", "—"
    meta = e.get("meta", {})
    cl = e.get("per_class", {})
    return (
        str(meta.get("trainable_params", 0)), f"{meta.get('peak_vram_gb', 0):.2f}",
        f"{meta.get('train_minutes', 0):.1f}", f"{e['fp_per_image']:.2f}",
        f"{cl.get('pothole', {}).get('f1', 0):.3f}",
        f"{cl.get('crack', {}).get('f1', 0):.3f}",
        f"{cl.get('manhole', {}).get('f1', 0):.3f}",
    )


def _ci_overlap(a, b):
    if not a or not b:
        return False
    return max(a[0], b[0]) <= min(a[1], b[1])


def _reward_hacking_summary(e5, sft, rows):
    pred_path = PRED_DIR / "e5_test.jsonl"
    if not pred_path.exists():
        return "Chưa có JSONL E5 để đếm output rỗng và box quá rộng."
    pred_map = {p["image_id"]: p["raw"] for p in read_jsonl(pred_path)}
    empty_on_positive = 0
    giant_boxes = 0
    total_boxes = 0
    for row in rows:
        parsed = parse_output(pred_map.get(row["image_id"], ""))
        if parsed == [] and row["objects"]:
            empty_on_positive += 1
        if parsed:
            for box in parsed:
                total_boxes += 1
                x1, y1, x2, y2 = box["bbox"]
                if max(0, x2 - x1) * max(0, y2 - y1) > 500_000:
                    giant_boxes += 1
    delta = e5["micro"]["f1"] - sft["micro"]["f1"] if sft else 0
    ci_relation = "CI95 giao nhau" if sft and _ci_overlap(e5["f1_ci95"], sft["f1_ci95"]) else "CI95 không giao nhau"
    return (f"Trên test, model trả `[]` ở {empty_on_positive}/{len(rows)} ảnh có nhãn; "
            f"{giant_boxes}/{total_boxes} box dự đoán chiếm hơn 50% diện tích khung hình. "
            f"Chênh F1 E5 − SFT = {delta:+.3f} ({ci_relation}). Đây là dấu hiệu cần xem để đánh giá reward hacking; "
            "đối chiếu thêm `work/runs/e5_grpo/trainer` và ảnh validation trước khi kết luận.")


def _draw_examples(best_name: str, rows, predictions, limit=6):
    pred_map = {p["image_id"]: p["raw"] for p in predictions}
    correct, errors = [], []
    for row in rows:
        parsed = parse_output(pred_map.get(row["image_id"], "")) or []
        gt = gt_boxes(row)
        gt_simple = [{"label": b["label"], "bbox": tuple(b["bbox_2d"])} for b in gt]
        pred_simple = [{"label": p["label"], "bbox": tuple(p["bbox"])} for p in parsed]
        pairs = sorted((
            (iou_xyxy(g["bbox"], p["bbox"]), gi, pi)
            for gi, g in enumerate(gt_simple) for pi, p in enumerate(pred_simple)
            if g["label"] == p["label"]
        ), reverse=True)
        used_g, used_p, tp = set(), set(), 0
        for iou, gi, pi in pairs:
            if iou < 0.5:
                break
            if gi not in used_g and pi not in used_p:
                used_g.add(gi); used_p.add(pi); tp += 1
        has_error = tp < len(gt_simple) or tp < len(pred_simple)
        (errors if has_error else correct).append(row)
    chosen = correct[:3] + errors[:3]
    for row in rows:
        if len(chosen) >= limit:
            break
        if row not in chosen:
            chosen.append(row)
    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    links = []
    for idx, row in enumerate(chosen[:limit], 1):
        img = Image.open(image_path(row)).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in gt_boxes(row):
            x1, y1, x2, y2 = box["bbox_2d"]
            x1, x2 = x1 * img.width / 1000, x2 * img.width / 1000
            y1, y2 = y1 * img.height / 1000, y2 * img.height / 1000
            draw.rectangle((x1, y1, x2, y2), outline=(255, 70, 70), width=2)
            draw.text((x1, max(0, y1 - 14)), f"GT {box['label']}", fill=(255, 70, 70), font=font)
        parsed = parse_output(pred_map.get(row["image_id"], "")) or []
        for box in parsed:
            x1, y1, x2, y2 = box["bbox"]
            x1, x2 = x1 * img.width / 1000, x2 * img.width / 1000
            y1, y2 = y1 * img.height / 1000, y2 * img.height / 1000
            draw.rectangle((x1, y1, x2, y2), outline=(50, 230, 100), width=2)
            draw.text((x1, min(img.height - 14, y2)), f"Pred {box['label']}", fill=(50, 230, 100), font=font)
        label = f"{idx:02d}_{'error' if row in errors else 'correct'}_{row['image_id']}.jpg"
        path = fig_dir / label
        img.save(path, quality=92)
        links.append(f"![Ảnh {idx}: box đúng màu đỏ, dự đoán màu xanh](report/figures/{label})")
    return links


def generate_report():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence = {name: _load_json(EVIDENCE_DIR / f"{name}.json")
                for name in ("e0", "e1", "e2", "e3", "e4_data_size_1.0", "e4_data_size_0.1", "e5")}
    meta = _load_json(SPLITS_DIR / "meta.json", {})
    architecture = _load_json(ROOT / "work" / "architecture.json", {"parameters": {}})
    summary = _load_json(ROOT / "work" / "experiment_summary.json", {})
    best_name = summary.get("best_sft", "e3")
    val_rows = read_jsonl(SPLITS_DIR / "val.jsonl")
    test_rows = read_jsonl(SPLITS_DIR / "test.jsonl")
    best_val_path = PRED_DIR / f"{best_name}_val.jsonl"
    best_val_predictions = read_jsonl(best_val_path) if best_val_path.exists() else []
    examples = _draw_examples(best_name, val_rows, best_val_predictions) if best_val_predictions else []

    e0 = evidence["e0"]
    e0_test_pred_path = PRED_DIR / "e0_test.jsonl"
    wrong_coord = None
    malformed_examples = []
    error_counts = {}
    if e0 and e0_test_pred_path.exists():
        e0_preds = read_jsonl(e0_test_pred_path)
        raw_map = {p["image_id"]: p["raw"] for p in e0_preds}
        wrong_coord = evaluate(test_rows, raw_map, coord="pixel")
        for p in e0_preds:
            raw = p["raw"]
            parsed = parse_output(raw)
            if parsed is None:
                error_counts["JSON lỗi hoặc sai cấu trúc"] = error_counts.get("JSON lỗi hoặc sai cấu trúc", 0) + 1
                if len(malformed_examples) < 2:
                    malformed_examples.append(raw[:400].replace("\n", " "))
            elif not parsed:
                row = next((entry for entry in test_rows if entry["image_id"] == p["image_id"]), None)
                if row and row["objects"]:
                    error_counts["Trả [] khi ảnh có nhãn"] = error_counts.get("Trả [] khi ảnh có nhãn", 0) + 1
            else:
                outside = sum(any(v < 0 or v > 1000 for v in box["bbox"]) for box in parsed)
                if outside:
                    error_counts["Box vượt ngoài thang 0–1000"] = error_counts.get("Box vượt ngoài thang 0–1000", 0) + 1
                unknown = sum(box["label"] not in {"pothole", "crack", "manhole"} for box in parsed)
                if unknown:
                    error_counts["Nhãn ngoài ba lớp"] = error_counts.get("Nhãn ngoài ba lớp", 0) + 1
    top_errors = sorted(error_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
    raw_example = malformed_examples[0] if malformed_examples else (
        next((p["raw"][:400].replace("\n", " ") for p in read_jsonl(e0_test_pred_path)), "Không có output E0.")
        if e0_test_pred_path.exists() else "Chưa có output E0.")

    arch = architecture.get("parameters", {})
    q1 = (f"Theo cây tham số được đếm trong lần chạy này: vision tower {arch.get('vision_tower', 0):,}, "
          f"projector {arch.get('projector', 0):,}, LLM {arch.get('llm', 0):,}; tổng {arch.get('total', 0):,}. "
          "E1 cập nhật LoRA trên 252 projection của LLM; E2 cập nhật cùng LoRA và merger/deepstack merger; "
          "E3 bổ sung LoRA cho 96 projection attention/MLP của ViT.")
    top_error_text = "; ".join(f"{name}: {count}/{e0['n_images']} ảnh" for name, count in top_errors) if top_errors and e0 else "Chưa có thống kê lỗi E0."
    q2 = (f"E0 parse_valid_rate = {e0['parse_valid_rate']:.3f}. Hai nhóm output/lỗi phổ biến nhất theo bộ đếm: "
          f"{top_error_text}. Ví dụ output thô: `{raw_example}`") if e0 else "Chưa chạy E0 trên GPU."
    q3 = (f"Chấm cùng output E0 trên test ({e0['n_images']} ảnh) theo norm1000 cho F1={e0['micro']['f1']:.3f}; "
          f"chấm nhầm là pixel cho F1={wrong_coord['micro']['f1']:.3f}. Kiểm chứng bằng cách dùng lại chính file "
          "e0_test.jsonl và thay duy nhất tham số coord của trọng tài.") if e0 and wrong_coord else "Chưa có evidence E0 để kiểm tra thang tọa độ."
    q4 = ("Mask label bằng -100 ở prompt và token ảnh để loss chỉ tối ưu JSON do assistant tạo. Nếu tính loss cả prompt/vision token, "
          "mô hình bị tối ưu để dự đoán lại đầu vào hội thoại thay vì học phát hiện và định dạng box; token ảnh cũng không phải nhãn sinh cần học.")
    q5 = ("NF4 chỉ lượng tử hóa phần LLM; cấu hình BitsAndBytes bỏ qua visual và lm_head để merger không thành trọng số 4-bit bất khả vi tối ưu như tham số đầy đủ. "
          "E2/E3 lưu riêng visual.merger và deepstack_merger_list ở FP32, đồng thời lưu LoRA bằng adapter. FP32 giữ cập nhật projector có độ chính xác cao hơn; "
          "khi nạp lại, chương trình khôi phục extra_trainable.pt trước khi suy luận.")
    q6_parts = []
    for split, counts in meta.get("counts", {}).items():
        boxes = counts.get("boxes", {})
        q6_parts.append(f"{split}: {counts.get('images', 0)} ảnh; pothole {boxes.get('pothole',0)}, crack {boxes.get('crack',0)}, manhole {boxes.get('manhole',0)} box")
    q6 = ("Chia theo phiên quay để các frame gần nhau không xuất hiện đồng thời ở train và test; chia ngẫu nhiên có thể rò rỉ cảnh/khung hình và làm F1 lạc quan. "
          + "; ".join(q6_parts) + ".")
    comparisons = []
    for name in ("e1", "e2", "e3"):
        e = evidence[name]
        if e:
            comparisons.append(f"{name.upper()} F1 {e['micro']['f1']:.3f} CI95 {_ci(e)}, VRAM {e.get('meta',{}).get('peak_vram_gb',0):.2f} GB, "
                               f"{e.get('meta',{}).get('train_minutes',0):.1f} phút")
    q7 = ("; ".join(comparisons) + ". " +
          ("Các khoảng tin cậy E1–E3 giao nhau; chênh lệch cần được diễn giải thận trọng." if all(evidence[x] for x in ("e1","e2","e3")) and
           any(_ci_overlap(evidence[a]["f1_ci95"], evidence[b]["f1_ci95"]) for a,b in (("e1","e2"),("e1","e3"),("e2","e3"))) else
           "Đọc độ chồng lấp CI95 trước khi kết luận setup tốt hơn.")) if comparisons else "Chưa chạy E1–E3."
    widths = [f"{name.upper()} rộng {e['f1_ci95'][1]-e['f1_ci95'][0]:.3f}" for name, e in evidence.items()
              if name in ("e0", "e1", "e2", "e3") and e]
    q8 = (f"Tập test có {len(test_rows)} ảnh. Độ rộng CI95 bootstrap: " + "; ".join(widths) +
          ". Khoảng tin cậy rộng/chồng lấp nghĩa là thứ hạng nhỏ giữa các setup có thể do biến thiên mẫu; cần ưu tiên độ lớn chênh lệch và CI, không chỉ nhìn điểm F1.")
    best_test = evidence.get(best_name)
    if best_test:
        class_scores = {c: best_test.get("per_class", {}).get(c, {}).get("f1", 0) for c in ("pothole", "crack", "manhole")}
        weakest = min(class_scores, key=class_scores.get)
        sizes = {c: [] for c in class_scores}
        for row in test_rows:
            for obj in row["objects"]:
                if obj["label"] in sizes:
                    _, _, w, h = obj["yolo"]
                    sizes[obj["label"]].append(100 * w * h)
        med = statistics.median(sizes[weakest]) if sizes[weakest] else 0
        class_counts = meta.get("counts", {}).get("test", {}).get("boxes", {})
        q9 = (f"Setup được chọn theo val là {best_name.upper()}; lớp F1 thấp nhất trên test là {weakest} "
              f"(F1={class_scores[weakest]:.3f}), với {class_counts.get(weakest,0)} box test và diện tích box trung vị {med:.2f}% khung hình. "
              "Số mẫu ít làm CI theo lớp kém ổn định; hình minh họa dưới dùng validation, GT đỏ và dự đoán xanh.")
    else:
        q9 = "Chưa có kết quả SFT để phân tích lớp."
    e4a, e4b = evidence["e4_data_size_1.0"], evidence["e4_data_size_0.1"]
    q10 = (f"E4 giữ nguyên setup {best_name.upper()} và chỉ đổi tỷ lệ dữ liệu: 100% F1={_f1(e4a)} CI95={_ci(e4a)}; "
           f"10% F1={_f1(e4b)} CI95={_ci(e4b)}. "
           + ("CI95 giao nhau nên bằng chứng chưa đủ để nói khác biệt chắc chắn." if e4a and e4b and _ci_overlap(e4a["f1_ci95"], e4b["f1_ci95"])
              else "Đối chiếu chênh lệch F1 với CI95 để đánh giá độ tin cậy.")) if e4a and e4b else "Chưa hoàn tất cả hai giá trị E4."

    table_rows = []
    setups = [("E0", "Zero-shot", evidence["e0"]), ("E1", "LoRA LLM", evidence["e1"]),
              ("E2", "+ projector", evidence["e2"]), ("E3", "+ LoRA ViT", evidence["e3"]),
              ("E4 100%", "data_size=1.0", evidence["e4_data_size_1.0"]),
              ("E4 10%", "data_size=0.1", evidence["e4_data_size_0.1"]), ("E5", "GRPO", evidence["e5"])]
    for code, setup, e in setups:
        trainable, vram, mins, fp, pothole, crack, manhole = _stats(e)
        table_rows.append(f"| {code} | {setup} | {trainable} | {vram} | {mins} | {_f1(e)} | {_ci(e)} | {fp} | {pothole} | {crack} | {manhole} |")
    images_block = "\n\n".join(examples) if examples else "Chưa sinh ảnh: cần chạy hoàn tất E1–E3 trên GPU."
    e5_text = (f"E5 F1={_f1(evidence['e5'])}, CI95={_ci(evidence['e5'])}; so với setup SFT chọn {best_name.upper()} "
               f"F1={_f1(best_test)} CI95={_ci(best_test)}. "
               + _reward_hacking_summary(evidence["e5"], best_test, test_rows)) \
        if evidence["e5"] else "E5 chưa có evidence; chạy mặc định `python student/run_lab.py all` để thực hiện GRPO và sinh chỉ số."
    report = f'''# Báo cáo Lab Day 10 — Qwen3-VL-4B

Họ tên: [điền họ tên trước khi nộp]  
`dataset_version`: `{meta.get('dataset_version', 'chưa có')}`  
Setup SFT được chọn theo validation: **{best_name.upper()}**

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Trả lời 10 câu hỏi

### Câu 1 — Kiến trúc
{q1}

### Câu 2 — E0
{q2}

### Câu 3 — Thang tọa độ
{q3}

### Câu 4 — Mask loss
{q4}

### Câu 5 — Projector và lượng tử hóa
{q5}

### Câu 6 — Split theo phiên quay
{q6}

### Câu 7 — So sánh E1/E2/E3
{q7}

### Câu 8 — Khoảng tin cậy
{q8}

### Câu 9 — Phân tích lỗi theo lớp
{q9}

### Câu 10 — E4
{q10}

## Ảnh minh họa validation

Box đúng màu đỏ; dự đoán màu xanh.

{images_block}

## E5 — Bonus GRPO

{e5_text}

## Nhìn lại

Thay đoạn này bằng một quan sát thật về lỗi hoặc giới hạn mà bạn gặp khi xem log và ảnh dự đoán.
'''
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Đã cập nhật REPORT.md và tạo {len(examples)} ảnh validation.")


if __name__ == "__main__":
    generate_report()
