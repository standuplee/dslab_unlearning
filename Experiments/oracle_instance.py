import os, warnings, logging
from os.path import join
import torch
import numpy as np
from Trainer.trainer import Trainer  # 통합 Trainer
from read import RatingData, PairData, PairGraphData, loadData, readRating_full
from utils import saveObject
import wandb

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'result'))

"""
삭제된 데이터셋으로 retrain된 Oracle 모델 생성
# 모델 학습(WMF, LightGCN), 파라미터/체크포인트 저장, 평가 로그 저장
# 모든 del_per에 대해 오라클 학습 & 저장
# 저장 구조 -> result/oracle/{dataset}/{model}/dp{del_per}/
"""
class OracleInstance(object):
    def __init__(self, param, del_per_list=None):
        self.param = param
        self.del_per_list = del_per_list if del_per_list else [param.del_per]
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        warnings.simplefilter('ignore')

    def read(self):
        """Oracle은 full retrain 데이터셋 사용"""
        return readRating_full(self.param.train_dir, self.param.test_dir, self.param.del_type, self.param.del_per)

    def runModel(self, del_per, verbose=2):
        self.param.del_per = del_per
        
        # 경로 설정
        self.name = f"{self.param.model}_oracle_dp{del_per}"
        prefix = f"/oracle/{self.param.dataset}/{self.param.model}/dp_{del_per}/"
        self.param_dir = SAVE_DIR + prefix
        os.makedirs(self.param_dir, exist_ok=True)

        # 파라미터 저장
        saveObject(join(self.param_dir, 'param'), self.param)
        
        print(f"{self.name} begin:")
        self.logger.info(f"{self.name} begin:")

        # 데이터 로드
        train_rating, test_rating, active_rating, inactive_rating = self.read()

        # 데이터셋 변환
        if self.param.model in ['wmf', 'dmf', 'gmf', 'nmf', 'bpr']:
            train_data = loadData(RatingData(train_rating), self.param.batch, self.param.n_worker, True)
            test_data = loadData(RatingData(test_rating), len(test_rating[0]), self.param.n_worker, False)
            active_data = loadData(RatingData(active_rating), len(active_rating[0]), self.param.n_worker, False)
            inactive_data = loadData(RatingData(inactive_rating), len(inactive_rating[0]), self.param.n_worker, False)
        elif self.param.model == 'lightgcn':
            train_data = PairGraphData(train_rating, self.param.pos_data, self.param.n_user, self.param.n_item).edge_index
            test_data = PairGraphData(test_rating, self.param.pos_data, self.param.n_user, self.param.n_item).edge_index
            active_data = PairGraphData(active_rating, self.param.pos_data, self.param.n_user, self.param.n_item).edge_index
            inactive_data = PairGraphData(inactive_rating, self.param.pos_data, self.param.n_user, self.param.n_item).edge_index

        # Trainer로 학습 수행
        trainer = Trainer(self.param)
        model, result = trainer.train(
            train_data=train_data,
            test_data=test_data,
            active_data=active_data,
            inactive_data=inactive_data,
            verbose=verbose
        )

        # 모델 저장
        model_path = join(self.param_dir, f"{self.param.model}_oracle.pt")
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Oracle model saved at {model_path}")

        # 메트릭 저장
        result.update({"model_path": model_path})
        np.save(join(self.param_dir, 'metrics.npy'), result)

        # wandb 로깅
        wandb.log(result)

        print(f"Oracle training finished: {self.name}")
        return model_path

    def run(self, verbose=2):
        """여러 del_per에 대해 Oracle 실행"""
        model_paths = {}
        for dp in self.del_per_list:
            model_path = self.runModel(dp, verbose)
            model_paths[dp] = model_path
        return model_paths