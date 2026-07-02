from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


class MaskEvaluator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _prepare_rle_for_decode(rle: dict[str, Any]) -> dict[str, Any]:
        prepared = {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"],
        }

        if isinstance(prepared["counts"], str):
            prepared["counts"] = prepared["counts"].encode("utf-8")

        return prepared

    def _decode_single_mask(self, rle: dict[str, Any]) -> Optional[np.ndarray]:
        try:
            prepared = self._prepare_rle_for_decode(rle)
            decoded = mask_utils.decode(prepared)

            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]

            return decoded.astype(bool)
        except Exception:
            return None

    @staticmethod
    def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    def score(
        self,
        gt_masks: list[dict[str, Any]],
        pred_masks: list[dict[str, Any]],
    ) -> float:
        k = len(gt_masks)
        m = len(pred_masks)

        if k == 0 and m == 0:
            return 1.0

        if k == 0 or m == 0:
            return 0.0

        decoded_gt = [self._decode_single_mask(rle) for rle in gt_masks]
        decoded_pred = [self._decode_single_mask(rle) for rle in pred_masks]

        iou_matrix = np.zeros((k, m), dtype=np.float64)

        for i, gt_mask in enumerate(decoded_gt):
            if gt_mask is None:
                continue
            for j, pred_mask in enumerate(decoded_pred):
                if pred_mask is None:
                    continue

                if gt_mask.shape != pred_mask.shape:
                    continue

                iou_matrix[i, j] = self._iou(gt_mask, pred_mask)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_sum = 0.0
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            iou_val = float(iou_matrix[r, c])
            matched_sum += iou_val

        denominator = max(k, m)
        score = float(matched_sum / denominator) if denominator > 0 else 0.0
        return float(np.clip(score, 0.0, 1.0))
