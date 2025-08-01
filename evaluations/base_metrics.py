import numpy as np
import torch

# -------------------------
# 기본 메트릭 계산 함수
# -------------------------

def recall(gt_items, pred_items):
    hits = 0
    for gt_item in gt_items:
        if gt_item in pred_items:
            hits += 1
    return hits / len(gt_items) if len(gt_items) > 0 else 0


def hit(gt_items, pred_items):
    hr = 0
    for gt_item in gt_items:
        if gt_item in pred_items:
            hr = hr + 1

    return hr / len(gt_items)


def ndcg(gt_items, pred_items):
    dcg = 0
    idcg = 0

    for gt_item in gt_items:
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            dcg = dcg + np.reciprocal(np.log2(index + 2))

    for index in range(len(gt_items)):
        idcg = idcg + np.reciprocal(np.log2(index + 2))

    return dcg / idcg





def recall_at_k(gt_items, pred_items, k=10):
    """
    Recall@k 계산
    gt_items: ground truth 아이템 리스트
    pred_items: 예측된 top-k 아이템 리스트
    """
    hits = len(set(gt_items) & set(pred_items[:k]))
    return hits / len(gt_items) if len(gt_items) > 0 else 0


def hit_ratio_at_k(gt_items, pred_items, k=10):
    """
    Hit Ratio@k 계산
    """
    return 1.0 if len(set(gt_items) & set(pred_items[:k])) > 0 else 0


def ndcg_at_k(gt_items, pred_items, k=10):
    """
    NDCG@k 계산
    """
    dcg = 0.0
    for i, item in enumerate(pred_items[:k]):
        if item in gt_items:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_items), k)))
    return dcg / idcg if idcg > 0 else 0