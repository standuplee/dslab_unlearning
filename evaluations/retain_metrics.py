import torch

def retain_metrics(retrain_predictions: torch.Tensor,
                   unlearned_predictions: torch.Tensor,
                   k: int = 10) -> dict:
    """
    Retain-NDCG@k / Retain-Recall@k 계산
    Retrain 모델(top-k)와 Unlearned 모델(top-k)의 추천 일치 정도
    
    Args:
        retrain_predictions: torch.Tensor (n_users, n_items) retrained model predictions
        unlearned_predictions: torch.Tensor (n_users, n_items) unlearned model predictions
        k: cutoff value
    
    Returns:
        dict: {"Retain-NDCG@k": value, "Retain-Recall@k": value}
    """
    # top-k 아이템 index 추출
    _, retrain_topk = torch.topk(retrain_predictions, k, dim=1)
    _, unlearn_topk = torch.topk(unlearned_predictions, k, dim=1)

    n_users = retrain_predictions.shape[0]
    recall_scores = []
    ndcg_scores = []

    for u in range(n_users):
        retrain_set = set(retrain_topk[u].cpu().tolist())
        unlearned_set = set(unlearn_topk[u].cpu().tolist())

        # Recall: 두 set의 교집합 비율
        intersection = len(retrain_set & unlearned_set)
        recall_scores.append(intersection / k)

        # NDCG: retrain의 순위를 기준으로 unlearned top-k의 DCG 계산
        retrain_rank = {item: idx for idx, item in enumerate(retrain_topk[u].cpu().tolist())}
        dcg = sum(1.0 / torch.log2(torch.tensor(retrain_rank[item] + 2.0)).item()
                  for item in unlearned_set if item in retrain_rank)
        idcg = sum(1.0 / torch.log2(torch.tensor(i + 2.0)).item() for i in range(k))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    return {
        f"Retain-Recall@{k}": sum(recall_scores) / len(recall_scores),
        f"Retain-NDCG@{k}": sum(ndcg_scores) / len(ndcg_scores)
    }