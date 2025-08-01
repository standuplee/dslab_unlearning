import pickle

from evaluations.base_metrics import hit, ndcg
import numpy as np
import ot
import torch
from tqdm import tqdm


# task vector용 baseTest 함수
def baseTestWithPredictions(dataloader, prediction_matrix, device, verbose, pos_dict, 
                           n_items, top_k=20, user_mapping=None, pos_mapping=None):
    """
    미리 계산된 예측 행렬을 사용하여 모델을 평가합니다.
    
    Args:
        dataloader: 테스트 데이터 로더
        prediction_matrix: (n_user, n_item) 차원의 예측 행렬
        device: 사용할 디바이스
        verbose: 출력 레벨
        pos_dict: 사용자별 positive 아이템 딕셔너리
        n_items: 전체 아이템 수
        top_k: 추천할 아이템 수
        user_mapping: 사용자 매핑 (선택사항)
        pos_mapping: positive 매핑 (선택사항)
    
    Returns:
        tuple: (평균 NDCG, 평균 HR)
    """
    full_items = [i for i in range(n_items)]

    HR = []
    NDCG = []

    for user, item, rating in dataloader:
        all_users = user.unique()
        all_users = all_users.to(device)
        user = user.to(device)
        item = item.to(device)

        for uid in all_users:
            user_id = uid.item()
            user_indices = torch.where(user == uid)
            gt_items = item[user_indices].cpu().numpy().tolist()

            # 해당 사용자의 예측값 가져오기
            user_predictions = prediction_matrix[user_id]  # (n_item,)
            
            # positive 아이템 제외
            neg_items = list(set(full_items) - set(pos_dict[user_id]))
            
            # negative 아이템에 대한 예측값만 추출
            neg_predictions = user_predictions[neg_items]
            
            # top-k 추천
            _, indices = torch.topk(neg_predictions, top_k)
            recommends = [neg_items[idx] for idx in indices.cpu().numpy()]

            HR.append(hit(gt_items, recommends))
            NDCG.append(ndcg(gt_items, recommends))

    return np.mean(NDCG), np.mean(HR)