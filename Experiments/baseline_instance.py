import os, warnings, logging
from os.path import join, exists
import numpy as np
import torch

from Trainer.trainer import Trainer
from read import RatingData, PairData, PairGraphData, loadData, readRating_group
from Methods.SISA import SISA
import wandb

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'result'))

'''
unlearning 방법 적용한 모델(SiSA, RecEraser, UltraRE, SCIF, IFRU, Our Model)
# oracle model(retrain) 로드, 각 언러닝 방법 수행
# oracle과 비교하는 지표 계산
# 평가 - Recall@10, NDCG@10, HitRatio@10, Retain-Recall / Retain-NDCG / TaskVector (Oracle vs Baseline 비교)


# 전체 baseline 실행
baseline.run()

# 특정 baseline만 실행
baseline.run(methods="sisa")

# 여러 baseline만 실행
baseline.run(methods=["sisa", "ours"])
'''
class BaselineInstance(object):
    def __init__(self, param, del_per_list=None):
        self.param = param
        self.del_per_list = del_per_list if del_per_list else [self.param.del_per]
        
        prefix = f"/baseline/{self.param.dataset}/{self.param.model}/"
        self.name = prefix + f"{self.param.learn_type}"
        
        self.param_dir = SAVE_DIR + self.name
        os.makedirs(self.param_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        warnings.simplefilter('ignore')

    # ---------------- Oracle 로드 ---------------- #
    def load_oracle(self, dp):
        oracle_dir = join(SAVE_DIR, f"oracle/{self.param.dataset}/{self.param.model}/dp_{dp}")
        model_path = join(oracle_dir, f"{self.param.model}_oracle.pt")
        metric_path = join(oracle_dir, "metrics.npy")
        
        if not exists(model_path):
            raise FileNotFoundError(f"Oracle model for del_per={dp} not found. Please run OracleInstance first.")
        
        # Oracle 메트릭 로드
        oracle_metrics = np.load(metric_path, allow_pickle=True).item()
        self.logger.info(f"Oracle model metrics loaded from {metric_path}")
        return model_path, oracle_metrics

    # ---------------- 데이터 로드 ---------------- #
    def read(self, dp):
        return readRating_group(
            self.param.train_dir, self.param.test_dir,
            self.param.del_type, dp,
            self.param.learn_type, self.param.n_group,
            self.param.dataset
        )

    # ---------------- Baseline 로직 Placeholder ---------------- #
    def _train_sisa(self, train_data, test_data, active_data, inactive_data, verbose):
        sisa = SISA(self.param)
        return sisa.run(train_data, test_data, active_data, inactive_data, verbose=verbose)

    def _train_receraser(self, train_data, test_data, active_data, inactive_data, verbose):
        self.logger.info("Running RecEraser logic...")
        trainer = Trainer(self.param)
        return trainer.train(train_data, test_data, active_data, inactive_data, verbose)

    def _train_ultrare(self, train_data, test_data, active_data, inactive_data, verbose):
        self.logger.info("Running UltraRE logic...")
        trainer = Trainer(self.param)
        return trainer.train(train_data, test_data, active_data, inactive_data, verbose)

    def _train_scif(self, train_data, test_data, active_data, inactive_data, verbose):
        self.logger.info("Running SCIF logic...")
        trainer = Trainer(self.param)
        return trainer.train(train_data, test_data, active_data, inactive_data, verbose)

    def _train_ifru(self, train_data, test_data, active_data, inactive_data, verbose):
        self.logger.info("Running IFRU logic...")
        trainer = Trainer(self.param)
        return trainer.train(train_data, test_data, active_data, inactive_data, verbose)

    def _train_ours(self, train_data, test_data, active_data, inactive_data, verbose):
        self.logger.info("Running OURS logic (SVD TaskVector)...")
        trainer = Trainer(self.param)
        return trainer.train(train_data, test_data, active_data, inactive_data, verbose)

    # ---------------- Core Baseline Runner ---------------- #
    def _run_baseline_method(self, method_name, train_rating, test_rating, active_rating, inactive_rating, oracle_metrics, dp, verbose):
        for i in range(self.param.n_group):
            # 데이터 로드
            if self.param.model in ['wmf', 'dmf', 'gmf', 'nmf', 'bpr']:
                train_data = loadData(RatingData(train_rating[i]), self.param.batch, self.param.n_worker, True)
                test_data = loadData(RatingData(test_rating[i]), len(test_rating[i][0]), self.param.n_worker, False)
                active_data = loadData(RatingData(active_rating[i]), len(active_rating[i][0]), self.param.n_worker, False)
                inactive_data = loadData(RatingData(inactive_rating[i]), len(inactive_rating[i][0]), self.param.n_worker, False)
            elif self.param.model == 'lightgcn':
                train_data = PairGraphData(train_rating[i], self.param.pos_data, self.param.n_user, self.param.n_item).edge_index
                test_data = PairGraphData(test_rating[i], self.param.pos_data, self.param.n_user, self.param.n_item).edge_index
                active_data = PairGraphData(active_rating[i], self.param.pos_data, self.param.n_user, self.param.n_item).edge_index
                inactive_data = PairGraphData(inactive_rating[i], self.param.pos_data, self.param.n_user, self.param.n_item).edge_index

            # Baseline별 로직 선택
            train_func = getattr(self, f"_train_{method_name}")
            model, result = train_func(train_data, test_data, active_data, inactive_data, verbose)

            # Retain 지표 계산
            retain_recall = abs(result.get("recall", 0) - oracle_metrics.get("recall", 0))
            retain_ndcg = abs(result.get("ndcg", 0) - oracle_metrics.get("ndcg", 0))
            result["Retain_Recall@10"] = retain_recall
            result["Retain_NDCG@10"] = retain_ndcg

            # 저장 경로
            result_dir = join(self.param_dir, f"dp_{dp}")
            os.makedirs(result_dir, exist_ok=True)
            np.save(join(result_dir, f"{method_name}_group{i+1}_metrics.npy"), result)
            wandb.log({f"{method_name}_dp{dp}_Group{i+1}_" + k: v for k, v in result.items()})

    # ---------------- 실행 ---------------- #
    def run(self, verbose=2, methods=None):
        all_methods = ["sisa", "receraser", "ultrare", "scif", "ifru", "ours"]
        run_methods = [methods] if isinstance(methods, str) else (methods if methods else all_methods)
        
        for dp in self.del_per_list:
            train_rating, test_rating, active_rating, inactive_rating = self.read(dp)
            _, oracle_metrics = self.load_oracle(dp)

            for method in run_methods:
                self.logger.info(f"Running baseline {method} for del_per={dp}")
                print(f"==== {method.upper()} START (del_per={dp}) ====")
                
                self._run_baseline_method(method, train_rating, test_rating, active_rating, inactive_rating, oracle_metrics, dp, verbose)
                
                print(f"==== {method.upper()} END (del_per={dp}) ====")