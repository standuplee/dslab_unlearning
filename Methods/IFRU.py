import logging
import torch
import torch.nn.functional as F
from Trainer.trainer import Trainer

class IFRU:
    def __init__(self, param):
        self.param = param
        self.logger = logging.getLogger(__name__)

    def preprocess(self, train_data, active_data):
        """
        IFRU-specific retain 데이터 준비
        active_data에 해당하는 샘플을 train_data에서 제외하여 retain set 생성
        """
        self.logger.info("IFRU: retain 데이터 전처리 중 (active 데이터 제외)")

        # active set의 (user, item) 쌍 추출
        active_pairs = set()
        for batch in active_data:
            user_ids, item_ids, _ = batch
            for u, i in zip(user_ids.tolist(), item_ids.tolist()):
                active_pairs.add((u, i))

        # retain set 생성 (active_pairs 제외)
        retain_samples = []
        for batch in train_data:
            user_ids, item_ids, labels = batch
            mask = [
                (u.item(), i.item()) not in active_pairs
                for u, i in zip(user_ids, item_ids)
            ]
            if any(mask):
                retain_samples.append((
                    user_ids[mask],
                    item_ids[mask],
                    labels[mask]
                ))

        # retain_samples → DataLoader 형태로 재구성
        # 여기서는 간단히 train_data와 같은 타입 유지
        retain_data = retain_samples
        self.logger.info(f"IFRU: retain 데이터 생성 완료 ({len(retain_samples)} 배치)")

        return retain_data

    def compute_influence(self, model, active_data):
        """
        영향도 계산
        """
        self.logger.info("IFRU: 영향도 계산 시작")
        influences = []
        model.eval()
        for batch in active_data:
            user, item, label = batch
            pred = model(user, item)
            loss = F.mse_loss(pred, label.float(), reduction='none')
            influences.append(loss.detach().cpu())
        score = torch.cat(influences).mean().item()
        self.logger.info(f"IFRU: 평균 영향도 = {score:.6f}")
        return score

    def fine_tune_with_influence(self, model, active_data):
        """
        Influence 기반 fine-tuning (Negative Gradient)
        """
        self.logger.info("IFRU: Influence 기반 fine-tuning 시작")
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.param.lr)

        for epoch in range(3):  # 논문에서는 소규모 step fine-tune
            for batch in active_data:
                user, item, label = batch
                pred = model(user, item)
                loss = F.mse_loss(pred, label.float())
                neg_loss = -loss
                opt.zero_grad()
                neg_loss.backward()
                opt.step()

        self.logger.info("IFRU: Influence 기반 fine-tuning 완료")
        return model

    def run(self, train_data, test_data, active_data, inactive_data, verbose=2):
        """
        IFRU 실행 메인 로직
        """
        self.logger.info("IFRU: 실행 시작")

        # 1. retain 데이터 생성
        retain_data = self.preprocess(train_data, active_data)

        # 2. retain 데이터로 incremental retrain
        trainer = Trainer(self.param)
        model, result = trainer.train(
            train_data=retain_data,
            test_data=test_data,
            active_data=active_data,
            inactive_data=inactive_data,
            verbose=verbose
        )

        # 3. Influence 계산
        self.compute_influence(model, active_data)

        # 4. Influence 기반 fine-tuning
        model = self.fine_tune_with_influence(model, active_data)

        self.logger.info("IFRU: 실행 완료")
        return model, result