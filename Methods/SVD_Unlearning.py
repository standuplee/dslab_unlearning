import pickle

import numpy as np
import ot
import torch

def svd_based_unlearning(original_interaction_matrix, ideal_interaction_matrix, 
                        predict_prob, 
                        alpha=0.5, beta=0.3, rank_ratio=1.0,
                        use_degree_weighting=False,
                        degree_weight_power=0.5,
                        normalize_weights=True):
    """
    SVD 기반 추천시스템 언러닝 (일반 버전과 degree-weighted 버전 모두 지원)
    
    Parameters:
    -----------
    original_interaction_matrix : torch.Tensor
        원본 상호작용 행렬 (binary)
    ideal_interaction_matrix : torch.Tensor
        이상적인 상호작용 행렬 (삭제할 엣지가 제거된 상태)
    predict_prob : torch.Tensor
        예측 확률 행렬 (0~1)
    alpha : float
        아이템 임베딩 수정 강도
    beta : float
        사용자 임베딩 수정 강도
    rank_ratio : float
        사용할 SVD rank 비율
    use_degree_weighting : bool
        degree-weighted 방법 사용 여부
    degree_weight_power : float
        degree 가중치 지수 (0: 무시, 0.5: sqrt, 1: linear)
    normalize_weights : bool
        가중치 정규화 여부
    """
    
    device = original_interaction_matrix.device
    
    print(f"Original matrix shape: {original_interaction_matrix.shape}")
    print(f"Ideal matrix shape: {ideal_interaction_matrix.shape}")
    print(f"Prediction matrix shape: {predict_prob.shape}")
    print(f"Using degree weighting: {use_degree_weighting}")
    
    # 텐서를 numpy 배열로 변환
    R_original = original_interaction_matrix.cpu().float().numpy()
    R_ideal = ideal_interaction_matrix.cpu().float().numpy()
    R_pred = predict_prob.detach().cpu().float().numpy()
    
    # Degree 계산 (필요한 경우)
    if use_degree_weighting:
        print("Calculating degree weights...")
        
        # 사용자와 아이템의 degree 계산
        user_degrees = np.sum(R_original, axis=1)  # 각 사용자의 평가 수
        item_degrees = np.sum(R_original, axis=0)  # 각 아이템의 평가 수
        
        # Degree 가중치 계산 (영향력과 반비례)
        # epsilon 추가하여 0으로 나누기 방지
        epsilon = 1e-10
        user_weights = 1.0 / (user_degrees ** degree_weight_power + epsilon)
        item_weights = 1.0 / (item_degrees ** degree_weight_power + epsilon)
        
        # 가중치 정규화
        if normalize_weights:
            user_weights = user_weights / np.mean(user_weights)
            item_weights = item_weights / np.mean(item_weights)
        
        print(f"User weights - min: {user_weights.min():.4f}, max: {user_weights.max():.4f}, mean: {user_weights.mean():.4f}")
        print(f"Item weights - min: {item_weights.min():.4f}, max: {item_weights.max():.4f}, mean: {item_weights.mean():.4f}")
    
    print("Performing SVD decomposition...")
    
    # SVD 수행
    U1, S1, V1 = np.linalg.svd(R_original, full_matrices=False)
    U2, S2, V2 = np.linalg.svd(R_ideal, full_matrices=False)
    U3, S3, V3 = np.linalg.svd(R_pred, full_matrices=False)
    
    print(f"SVD shapes - Original: U1{U1.shape}, S1{S1.shape}, V1{V1.shape}")
    print(f"SVD shapes - Ideal: U2{U2.shape}, S2{S2.shape}, V2{V2.shape}")
    print(f"SVD shapes - Prediction: U3{U3.shape}, S3{S3.shape}, V3{V3.shape}")
    
    # Rank 결정
    min_rank = min(len(S1), len(S2), len(S3))
    target_rank = int(min_rank * rank_ratio)
    
    print(f"Using rank: {target_rank} (from min_rank: {min_rank})")
    
    # Truncation
    U1_truncated = U1[:, :target_rank]
    U2_truncated = U2[:, :target_rank]
    U3_truncated = U3[:, :target_rank]
    
    V1_truncated = V1[:target_rank, :]
    V2_truncated = V2[:target_rank, :]
    V3_truncated = V3[:target_rank, :]
    
    S1_truncated = S1[:target_rank]
    S2_truncated = S2[:target_rank]
    S3_truncated = S3[:target_rank]
    
    # Delta 계산
    Delta_U = U2_truncated - U1_truncated
    Delta_V = V2_truncated - V1_truncated
    
    # Degree weighting 적용
    if use_degree_weighting:
        print("Applying degree weighting to deltas...")
        
        # 사용자 가중치를 Delta_U에 적용
        # Delta_U shape: (n_users, rank)
        # user_weights shape: (n_users,)
        Delta_U_weighted = Delta_U * user_weights.reshape(-1, 1)
        
        # 아이템 가중치를 Delta_V에 적용
        # Delta_V shape: (rank, n_items)
        # item_weights shape: (n_items,)
        Delta_V_weighted = Delta_V * item_weights.reshape(1, -1)
        
        print(f"Original Delta_U norm: {np.linalg.norm(Delta_U):.6f}")
        print(f"Weighted Delta_U norm: {np.linalg.norm(Delta_U_weighted):.6f}")
        print(f"Original Delta_V norm: {np.linalg.norm(Delta_V):.6f}")
        print(f"Weighted Delta_V norm: {np.linalg.norm(Delta_V_weighted):.6f}")
        
        Delta_U = Delta_U_weighted
        Delta_V = Delta_V_weighted
    else:
        print(f"Delta_U norm: {np.linalg.norm(Delta_U):.6f}")
        print(f"Delta_V norm: {np.linalg.norm(Delta_V):.6f}")
    
    # 수정된 SVD 구성요소 계산
    U3_corrected = U3_truncated + beta * Delta_U
    V3_corrected = V3_truncated + alpha * Delta_V
    
    # 재구성
    R_corrected = U3_corrected @ np.diag(S3_truncated) @ V3_corrected
    
    # 값 범위 클리핑 (0~1)
    R_corrected = np.clip(R_corrected, 0, 1)
    
    # 다시 텐서로 변환
    corrected_predictions = torch.tensor(R_corrected, device=device, dtype=torch.float32)
    
    return corrected_predictions


def joint_svd_unlearning(original_interaction_matrix,
                         ideal_interaction_matrix, 
                         predict_prob, alpha, 
                         beta, 
                         stacking_method='vertical'):
    """모든 행렬을 함께 고려하는 Joint SVD unlearning"""
    device = original_interaction_matrix.device

    R_original = original_interaction_matrix.cpu().float().numpy()
    R_ideal = ideal_interaction_matrix.cpu().float().numpy()
    R_pred = predict_prob.detach().cpu().float().numpy()    
    n_users, n_items = R_original.shape
    
    # 1. 행렬 결합 방법 선택
    if stacking_method == 'vertical':
        # 세로로 쌓기 (사용자 공간 공유)
        R_combined = np.vstack([R_original, R_ideal, R_pred])
        
    elif stacking_method == 'horizontal':
        # 가로로 쌓기 (아이템 공간 공유)
        R_combined = np.hstack([R_original, R_ideal, R_pred])
        
    elif stacking_method == 'weighted_sum':
        # 가중 합으로 결합
        w1, w2, w3 = 0.4, 0.4, 0.2  # 원본과 이상적 행렬에 더 가중치
        R_combined = w1 * R_original + w2 * R_ideal + w3 * R_pred
        
    # 2. Joint SVD 수행
    U_joint, S_joint, V_joint = np.linalg.svd(R_combined, full_matrices=False)
    
    if stacking_method == 'vertical':
        # 3a. Vertical stacking의 경우 - 공통 아이템 공간 사용
        V_shared = V_joint
        
        # 각 부분의 사용자 임베딩 추출
        U1 = U_joint[:n_users]
        U2 = U_joint[n_users:2*n_users]
        U3 = U_joint[2*n_users:3*n_users]
        
        # Singular values 재분배
        S_shared = S_joint
        
        # Delta 계산 (같은 basis 사용하므로 안전)
        Delta_U = U2 - U1
        U3_corrected = U3 + beta * Delta_U
        
        # 재구성
        result = U3_corrected @ np.diag(S_shared) @ V_shared
        
    elif stacking_method == 'horizontal':
        # 3b. Horizontal stacking의 경우 - 공통 사용자 공간 사용
        U_shared = U_joint
        
        # 각 부분의 아이템 임베딩 추출
        V1 = V_joint[:n_items]
        V2 = V_joint[n_items:2*n_items]
        V3 = V_joint[2*n_items:3*n_items]
        
        # Delta 계산
        Delta_V = V2 - V1
        V3_corrected = V3 + alpha * Delta_V
        
        # 재구성
        result = U_shared @ np.diag(S_joint) @ V3_corrected
        
    elif stacking_method == 'weighted_sum':
        # 3c. Weighted sum의 경우 - 직접적인 분해
        U_combined = U_joint
        V_combined = V_joint
        S_combined = S_joint
        
        # 원본 행렬들도 같은 basis로 투영
        U1 = R_original @ V_combined.T @ np.diag(1/S_combined)
        U2 = R_ideal @ V_combined.T @ np.diag(1/S_combined)
        U3 = R_pred @ V_combined.T @ np.diag(1/S_combined)
        
        # Delta 계산
        Delta_U = U2 - U1
        U3_corrected = U3 + beta * Delta_U
        
        # 재구성
        result = U3_corrected @ np.diag(S_combined) @ V_combined
    
    return torch.sigmoid(torch.tensor(result, device=device, dtype=torch.float32))