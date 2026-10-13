"""Pure Python GRPO reward for normalized road damage boxes."""
from __future__ import annotations

import json
from typing import Any

from scripts.score import iou_xyxy, parse_output


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") in {None, "assistant"}:
                return str(message.get("content", ""))
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
    return str(completion)


def reward_one(completion: Any, ground_truth: Any) -> float:
    """Score one completion; malformed JSON gets 0, empty/empty gets 1."""
    pred = parse_output(_completion_text(completion))
    if pred is None:
        return 0.0
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    gt = []
    for box in ground_truth:
        if "bbox" in box:
            coords = box["bbox"]
        else:
            coords = box["bbox_2d"]
        gt.append({"label": box["label"], "bbox": tuple(float(x) for x in coords)})
    if not gt and not pred:
        return 1.0
    candidates = []
    for i, actual in enumerate(gt):
        for j, guess in enumerate(pred):
            if actual["label"] == guess["label"]:
                iou = iou_xyxy(actual["bbox"], guess["bbox"])
                if iou >= 0.5:
                    candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_gt, used_pred, total_iou = set(), set(), 0.0
    for iou, i, j in candidates:
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
            total_iou += iou
    return 0.2 + 0.8 * total_iou / max(len(gt), len(pred), 1)


def iou_reward(completions, gt_boxes_json, **kwargs):
    """TRL reward callback. `gt_boxes_json` is preserved as a dataset column."""
    return [reward_one(completion, truth) for completion, truth in zip(completions, gt_boxes_json)]
