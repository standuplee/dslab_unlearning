import logging
import torch
from copy import deepcopy
from Trainer.trainer import Trainer

class SISA:
    def __init__(self, param):
        self.param = param
        self.logger = logging.getLogger(__name__)

    def _shard_data(self, train_rating, test_rating):
        """
        데이터 샤딩 로직 (단순 n_group 분할 예시)
        """
        n_group = self.param.n_group
        self.logger.info(f"SISA: 데이터 {n_group}개 그룹으로 샤딩")
        # 여기서는 단순 그룹 나누기 (실제 구현에서는 사용자 기반 그룹 인덱스 사용)
        group_size = len(train_rating) // n_group
        return [
            train_rating[i*group_size:(i+1)*group_size]
            for i in range(n_group)
        ], [
            test_rating[i*group_size:(i+1)*group_size]
            for i in range(n_group)
        ]

    def _train_shards(self, shard_train_list, shard_test_list, verbose):
        """
        각 샤드별 학습
        """
        self.logger.info("SISA: 샤드별 학습 시작")
        model_list = []
        for i in range(self.param.n_group):
            trainer = Trainer(self.param)
            model, _ = trainer.train(
                train_data=shard_train_list[i],
                test_data=shard_test_list[i],
                verbose=verbose
            )
            model_list.append(model)
        return model_list

    def _aggregate_models(self, model_list):
        """
        샤드별 모델 병합
        """
        self.logger.info("SISA: 모델 병합 시작")
        base_model = deepcopy(model_list[0])
        merged_weight = base_model.user_mat.weight.clone()

        for gid, model in enumerate(model_list):
            merged_weight = merged_weight + model.user_mat.weight  # 예시: 평균 병합
        merged_weight /= len(model_list)

        base_model.user_mat.weight = torch.nn.Parameter(merged_weight)
        return base_model

    def run(self, train_data, test_data, active_data, inactive_data, del_user=None, verbose=2):
        """
        SISA 실행
        """
        self.logger.info("SISA: 실행 시작")

        # 1. 샤드 데이터 생성
        shard_train_list, shard_test_list = self._shard_data(train_data, test_data)

        # 2. 샤드별 학습
        model_list = self._train_shards(shard_train_list, shard_test_list, verbose)

        # 3. (선택) 삭제 요청 반영
        if del_user:
            self.logger.info(f"SISA: 삭제 대상 유저 {del_user} 처리 (샤드 재학습 로직 필요)")
            # TODO: 삭제 대상 유저 포함된 샤드만 재학습

        # 4. 모델 병합
        final_model = self._aggregate_models(model_list)

        # 5. 최종 평가
        trainer = Trainer(self.param)
        model, result = trainer.train(
            train_data=train_data,
            test_data=test_data,
            active_data=active_data,
            inactive_data=inactive_data,
            verbose=verbose
        )

        self.logger.info("SISA: 완료")
        return model, result