import os
import time
import logging
import numpy as np
import torch
from torch import optim
import wandb

from utils import seed_all, baseTrain, baseTest

# 모델 로드
from Models.WMF import WMF
from Models.BPR import BPR
from Models.GMF import GMF
from Models.NMF import NMF
from Models.DMF import DMF
from Models.LightGCN import LightGCN


class Trainer:
    def __init__(self, param):
        self.param = param
        
        # Logger
        self.logger = logging.getLogger(f"Trainer-{param.model}")
        self.logger.setLevel(logging.INFO)

        # Device 설정
        self.device = (
            'mps' if torch.backends.mps.is_available()
            else 'cuda:0' if torch.cuda.is_available()
            else 'cpu'
        )

        # 모델 저장 디렉토리
        self.save_dir = f'model_params/{param.learn_type}/{param.model}_{param.dataset}_{param.del_type}_{param.del_per}'
        os.makedirs(self.save_dir, exist_ok=True)

    def _build_model(self):
        """모델 타입에 따라 인스턴스 생성"""
        n_user, n_item, k = self.param.n_user, self.param.n_item, self.param.k
        layers = getattr(self.param, "layers", None)

        model_type = self.param.model.lower()
        if model_type == 'wmf':
            return WMF(n_user, n_item, k).to(self.device)
        elif model_type == 'bpr':
            return BPR(n_user, n_item, k).to(self.device)
        elif model_type == 'gmf':
            return GMF(n_user, n_item, k).to(self.device)
        elif model_type == 'nmf':
            return NMF(n_user, n_item, k, layers).to(self.device)
        elif model_type == 'dmf':
            return DMF(n_user, n_item, k, layers).to(self.device)
        elif model_type == 'lightgcn':
            return LightGCN(n_user, n_item, embedding_dim=64, K=3).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, train_data, test_data, active_data=None, inactive_data=None, verbose=2):
        """공통 학습 루프"""
        model = self._build_model()
        seed_all(self.param.seed)

        opt = optim.Adam(model.parameters(), lr=self.param.lr)
        loss_fn = 'point-wise' if self.param.model != 'bpr' else 'pair-wise'

        best_ndcg, best_hr, best_recall = 0, 0, 0
        count_dec, total_time = 0, 0
        pos_dict = np.load(self.param.pos_data, allow_pickle=True).item()

        for epoch in range(self.param.epochs):
            self.logger.info(f"[Epoch {epoch+1}/{self.param.epochs}] Start")
            start_time = time.time()

            # Training
            train_loss = baseTrain(train_data, model, loss_fn, opt, self.device, verbose)

            # Evaluation (Test)
            test_ndcg, test_hr, test_recall = baseTest(
                test_data, model, loss_fn, self.device, verbose, pos_dict, self.param.n_item, 20
            )

            elapsed = time.time() - start_time
            total_time += elapsed

            # Logging
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_ndcg": test_ndcg,
                "test_hr": test_hr,
                "test_recall": test_recall
            })

            # Early stopping
            if test_ndcg > best_ndcg:
                best_ndcg, best_hr, best_recall = test_ndcg, test_hr, test_recall
                count_dec = 0
                torch.save(model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
            else:
                count_dec += 1
                if count_dec > 5:
                    self.logger.info("Early stopping triggered.")
                    break

        # Active / Inactive Evaluation
        active_ndcg, active_hr, active_recall = (0, 0, 0)
        if active_data is not None:
            active_ndcg, active_hr, active_recall = baseTest(
                active_data, model, loss_fn, self.device, verbose, pos_dict, self.param.n_item, 20
            )

        inactive_ndcg, inactive_hr, inactive_recall = (0, 0, 0)
        if inactive_data is not None:
            inactive_ndcg, inactive_hr, inactive_recall = baseTest(
                inactive_data, model, loss_fn, self.device, verbose, pos_dict, self.param.n_item, 20
            )

        # 결과 정리
        result = {
            "time": total_time,
            "ndcg": best_ndcg,
            "hr": best_hr,
            "recall": best_recall,
            "active_ndcg": active_ndcg,
            "active_hr": active_hr,
            "active_recall": active_recall,
            "inactive_ndcg": inactive_ndcg,
            "inactive_hr": inactive_hr,
            "inactive_recall": inactive_recall
        }

        wandb.log(result)
        self.logger.info(f"Best Results: {result}")

        return model, result