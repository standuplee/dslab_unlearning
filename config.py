## 실험 파라미터 설정

import os
import warnings
import time
from os.path import abspath, join, dirname, exists

import torch
import numpy as np
from lightgcn import LightGCN
from scratch import Scratch
from utils import saveObject

from read import RatingData, PairData, PairGraphData
from read import loadData, readRating_full, readRating_group

import wandb
import logging

DATA_DIR = abspath(join(dirname(__file__), 'data'))
SAVE_DIR = abspath(join(dirname(__file__), 'result'))


class InsParam(object):
    def __init__(self, dataset='ml-100k', model='wmf', epochs=50, n_worker=24, layers=[32], n_group=10, del_per=5,
                 learn_type='retrain',
                 del_type='random'):
        # model param
        self.k = 32  # dimension of embedding
        self.lam = 0.1  # regularization coefficient
        self.layers = layers  # structure of FC layers in DMF

        # training param
        self.seed = 42
        self.n_worker = n_worker
        self.batch = 1024
        self.lr = 0.001
        self.epochs = epochs
        self.n_group = n_group
        self.learn_type = learn_type

        # dataset-varied param
        self.del_rating = []  # 2d array/list [[uid, iid], ...]
        self.dataset = dataset
        self.max_rating = 5
        self.del_per = del_per
        self.del_type = del_type
        self.model = model

        if dataset == 'ml-100k':
            self.train_dir = DATA_DIR + '/ml-100k/train.csv'
            self.test_dir = DATA_DIR + '/ml-100k/test.csv'
            self.pos_data = DATA_DIR + '/ml-100k/pos_dict.npy'
            self.n_user = 943
            self.n_item = 1349

        elif dataset == 'ml-1m':
            self.train_dir = DATA_DIR + '/ml-1m/train.csv'
            self.test_dir = DATA_DIR + '/ml-1m/test.csv'
            self.pos_data = DATA_DIR + '/ml-1m/pos_dict.npy'

            self.n_user = 6040
            self.n_item = 3416

        elif dataset == 'adm':
            self.train_dir = DATA_DIR + '/adm/train.csv'
            self.test_dir = DATA_DIR + '/adm/test.csv'
            self.pos_data = DATA_DIR + '/adm/pos_dict.npy'

            self.n_user = 22878
            self.n_item = 115082

        elif dataset == 'gowalla':
            self.train_dir = DATA_DIR + '/gowalla/train.csv'
            self.test_dir = DATA_DIR + '/gowalla/test.csv'
            self.pos_data = DATA_DIR + '/gowalla/pos_dict.npy'

            self.n_user = 64115
            self.n_item = 164532


class Instance(object):
    def __init__(self, param):
        self.param = param
        prefix = '/test/' if self.param.del_type == 'test' else '/' + str(
            self.param.del_per) + '/' + self.param.del_type + '/'
        self.name = prefix + self.param.dataset + '_g_' + str(
            self.param.n_group)  # time.strftime("%Y%m%d_%H%M%S", time.localtime())
        param_dir = SAVE_DIR + self.name
        if exists(param_dir) == False:
            os.makedirs(param_dir)

        # save param
        saveObject(param_dir + '/param', self.param)  # loadObject(dir + '/param')
        
        self.logger = logging.getLogger(__name__)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # np.save(param_dir + '/deletion', deletion)  # np.load('deletion.npy')

    def read(self):
        learn_type = self.param.learn_type
        del_type = self.param.del_type
        del_per = self.param.del_per
        group = self.param.n_group
        if learn_type == 'retrain':
            train_rating, test_rating, active_rating, inactive_rating = readRating_full(self.param.train_dir,
                                                                                        self.param.test_dir, del_type,
                                                                                        del_per)
        else:
            train_rating, test_rating, active_rating, inactive_rating = readRating_group(self.param.train_dir,
                                                                                         self.param.test_dir, del_type,
                                                                                         del_per, learn_type, group,
                                                                                         self.param.dataset)

        return train_rating, test_rating, active_rating, inactive_rating

    def runModel(self, model_type='wmf', verbose=2):
        print(self.name, 'begin:')
        self.logger.info(f"{self.name} begin:")
        # read raw data
        train_rating, test_rating, active_rating, inactive_rating = self.read()
        
        
        # 공통 데이터 로드 함수 분리
        def load_datasets(train_r, test_r, active_r, inactive_r, model_type):
            if model_type in ['wmf', 'dmf', 'gmf', 'nmf']:
                train_data = loadData(RatingData(train_r), self.param.batch, self.param.n_worker, True)
                test_data = loadData(RatingData(test_r), len(test_r[0]), self.param.n_worker, False)
                active_data = loadData(RatingData(active_r), len(active_r[0]), self.param.n_worker, False) if active_r else None
                inactive_data = loadData(RatingData(inactive_r), len(inactive_r[0]), self.param.n_worker, False) if inactive_r else None
                return train_data, test_data, active_data, inactive_data

            elif model_type == 'bpr':
                train_data = loadData(PairData(train_r, self.param.pos_data), self.param.batch, self.param.n_worker, True)
                test_data = loadData(RatingData(test_r), len(test_r[0]), self.param.n_worker, False)
                active_data = loadData(RatingData(active_r), len(active_r[0]), self.param.n_worker, False) if active_r else None
                inactive_data = loadData(RatingData(inactive_r), len(inactive_r[0]), self.param.n_worker, False) if inactive_r else None
                return train_data, test_data, active_data, inactive_data

            #lightGCN의 경우에 edge_index 넘겨주기
            elif model_type == 'lightgcn':
                train_dataset = PairGraphData(train_r, self.param.pos_data, self.param.n_user, self.param.n_item)
                test_dataset = PairGraphData(test_r, self.param.pos_data, self.param.n_user, self.param.n_item)
                active_dataset = PairGraphData(active_r, self.param.pos_data, self.param.n_user, self.param.n_item) if active_r else None
                inactive_dataset = PairGraphData(inactive_r, self.param.pos_data, self.param.n_user, self.param.n_item) if inactive_r else None
                return train_dataset.edge_index, test_dataset.edge_index, \
                    active_dataset.edge_index if active_dataset else None, \
                    inactive_dataset.edge_index if inactive_dataset else None
        
        # 학습 실행
        def train_and_log(model, train_data, test_data, active_test_data, inactive_test_data, group_idx=None):
            if hasattr(model, "train_model"):  # LightGCN
                model, result = model.train_model(train_data, test_data, active_test_data, inactive_test_data, verbose)
            else:  # Scratch 스타일
                model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose, given_model='')
            
            # WandB metric 로깅
            prefix = f"Group{group_idx}_" if group_idx is not None else ""
            wandb.log({
                f"{prefix}Recall@10": result.get("Recall", 0),
                f"{prefix}NDCG@10": result.get("NDCG", 0),
                f"{prefix}HitRatio@10": result.get("HitRatio", 0)
            })
            
            # 저장
            save_name = (f"group{group_idx}_{self.param.n_group}_" if group_idx is not None else "") + \
                        f"{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.npy"
            result.update({'model': model_type, 'dataset': self.param.dataset,
                        'deltype': self.param.del_type, 'method': self.param.learn_type})
            np.save(f'results/{self.param.learn_type}/{save_name}', result)
            
        # ===== learn type = retrain =====
        if self.param.learn_type == 'retrain':
            train_data, test_data, active_test_data, inactive_test_data = load_datasets(
                train_rating, test_rating, active_rating, inactive_rating, model_type)
            
            # 모델 초기화
            if model_type in ['wmf', 'dmf', 'gmf', 'nmf', 'bpr']:
                model = Scratch(self.param, model_type)
            elif model_type == 'lightgcn':
                model = LightGCN(self.param.n_user, self.param.n_item, embedding_dim=64, K=3)
                
            self.logger.info(f"Train_data_unique: {len(np.unique(train_rating[0].values))}")
            self.logger.info(f"Test_data_unique: {len(np.unique(test_rating[0].values))}")
            print(f"Train_data_unique: {len(np.unique(train_rating[0].values))}")
            print(f"Test_data_unique: {len(np.unique(test_rating[0].values))}")

            train_and_log(model, train_data, test_data, active_test_data, inactive_test_data)

            print('End of training', self.name)
            self.logger.info(f"Training finished for {self.name}")

        # ===== Group-based =====
        else:
            group = self.param.n_group
            for i in range(group):
                train_data, test_data, active_test_data, inactive_test_data = load_datasets(
                    train_rating[i], test_rating[i], active_rating[i], inactive_rating[i], model_type)

                # 모델 초기화
                if model_type in ['wmf', 'dmf', 'gmf', 'nmf', 'bpr']:
                    model = Scratch(self.param, model_type)
                elif model_type == 'lightgcn':
                    model = LightGCN(self.param.n_user, self.param.n_item, embedding_dim=64, K=3)
            
                # group-specific 데이터 로깅
                self.logger.info(f"Group {i+1} Train_data_unique: {len(np.unique(train_rating[i][0].values))}")
                self.logger.info(f"Group {i+1} Test_data_unique: {len(np.unique(test_rating[i][0].values))}")
                print(f"Group {i+1} Train_data_unique: {len(np.unique(train_rating[i][0].values))}")
                print(f"Group {i+1} Test_data_unique: {len(np.unique(test_rating[i][0].values))}")

                train_and_log(model, train_data, test_data, active_test_data, inactive_test_data, group_idx=i+1)

                print(f'End of Group {i+1} / {group} training', self.name)
                self.logger.info(f'End of Group {i+1} / {group} training {self.name}')

    def run(self, verbose=2):
        self.runModel(self.param.model, verbose)