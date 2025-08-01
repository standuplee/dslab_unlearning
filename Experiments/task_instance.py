from os.path import abspath, join, dirname, exists

from Methods.SVD_Unlearning import joint_svd_unlearning, svd_based_unlearning
from config import Instance
from evaluations.retain_metrics import retain_metrics
from evaluations.taskvector_metrics import baseTestWithPredictions
import torch
import numpy as np
from lightgcn import LightGCN
from scratch import Scratch
from utils import BPR, DMF, GMF, NMF, WMF, convert_rating_to_tensor, saveObject

from read import RatingData, PairData, PairGraphData
from read import loadData, readRating_full, readRating_group

import wandb
import logging


class task_instance(Instance):
    def __init__(self, param):
        super().__init__(param)
        

    def runModel(self, model_type='wmf', verbose=2):
        print(self.name, 'begin:')
        self.logger.info(f"{self.name} begin:")
        # read raw data
        train_rating, test_rating, active_rating, inactive_rating = self.read() # list of [user_ids, item_ids, ratings]
        

        if self.param.learn_type == 'task_vector':
            # load data
            if model_type in ['wmf', 'dmf', 'gmf', 'nmf']:
                train_data = loadData(RatingData(train_rating), self.param.batch,
                                      self.param.n_worker,
                                      True)

            elif model_type in ['bpr', 'lightgcn']:
                train_data = loadData(PairData(train_rating, self.param.pos_data), self.param.batch,
                                      self.param.n_worker,
                                      True)

            test_data = loadData(RatingData(test_rating), len(test_rating[0]), self.param.n_worker, False)
            active_test_data = loadData(RatingData(active_rating), len(active_rating[0]), self.param.n_worker,
                                        False)
            inactive_test_data = loadData(RatingData(inactive_rating), len(inactive_rating[0]), self.param.n_worker,
                                          False)

            self.logger.info(f"Train_data_unique: {len(np.unique(train_rating[0].values))}")
            self.logger.info(f'Test_data_unique: {len(np.unique(test_rating[0].values))}')
            
            print(f'Train_data_unique: {len(np.unique(train_rating[0].values))}')
            print(f'Test_data_unique: {len(np.unique(test_rating[0].values))}')
            
            
            if model_type == 'wmf':
                model = WMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
                model_oracle = WMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
            elif model_type == 'bpr':
                model = BPR(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
                model_oracle = BPR(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
            elif model_type == 'gmf':
                model = GMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
                model_oracle = GMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
            elif model_type == 'nmf':
                model = NMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
                model_oracle = NMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
            elif model_type == 'dmf':
                model = DMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
                model_oracle = DMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
                
            # trained on original dataset
            model.load_state_dict(torch.load(self.param.origin_model_path))
            model_oracle.load_state_dict(torch.load(self.param.oracle_model_path))
        
        pos_dict = np.load(self.param.pos_data, allow_pickle=True).item()
        
        # original dataset
        original_rating_matrix = convert_rating_to_tensor(train_rating, self.param.n_user, self.param.n_item).to(self.param.device)
        # Unlearned dataset
        unlearn_ratings = pd.read_csv(self.param.train_dir[:-4]+f'_after_delete_del_per{self.param.del_per}.csv', sep=',')
        unlearn_rating_lists = [unlearn_ratings['uid'], unlearn_ratings['iid'], unlearn_ratings['val']]
        unlearn_rating_matrix = convert_rating_to_tensor(unlearn_rating_lists, self.param.n_user, self.param.n_item).to(self.param.device)
        
        # Original model prediction
        original_predictions = model.predict_matrix()
        oracle_predictions = model_oracle.predict_matrix()
        
        original_predictions = torch.sigmoid(original_predictions)
        oracle_predictions = torch.sigmoid(oracle_predictions)
        
        corrected_predictions = svd_based_unlearning(original_rating_matrix, 
                                                     unlearn_rating_matrix, 
                                                     original_predictions, 
                                                     self.param.alpha, 
                                                     self.param.beta, 
                                                     self.param.rank_ratio, 
                                                     self.param.use_degree_weighting, 
                                                     self.param.degree_weight_power, self.param.normalize_weights)
        
        ## Evaluation
        target_rating_matrix = original_rating_matrix - unlearn_rating_matrix
        print(target_rating_matrix.sum())
        target_rating_matrix = target_rating_matrix.to(self.param.device)
        original_predictions = original_predictions.to(self.param.device)
        oracle_predictions = oracle_predictions.to(self.param.device)
        corrected_predictions = corrected_predictions.to(self.param.device)
        
        original_prob = torch.sum(original_predictions * target_rating_matrix) / torch.sum(target_rating_matrix)
        oracle_prob = torch.sum(oracle_predictions * target_rating_matrix) / torch.sum(target_rating_matrix)
        corrected_prob = torch.sum(corrected_predictions * target_rating_matrix) / torch.sum(target_rating_matrix)
        print(f'original_prob: {original_prob}, oracle_prob: {oracle_prob}, corrected_prob: {corrected_prob}')
        
        negative_rating = 1- original_rating_matrix
        original_negative_prob = torch.sum(original_predictions * negative_rating) / torch.sum(negative_rating)
        oracle_negative_prob = torch.sum(oracle_predictions * negative_rating) / torch.sum(negative_rating)
        corrected_negative_prob = torch.sum(corrected_predictions * negative_rating) / torch.sum(negative_rating)
        print(f'original_negative_prob: {original_negative_prob}, oracle_negative_prob: {oracle_negative_prob}, corrected_negative_prob: {corrected_negative_prob}')

        
        ndcg, hr = baseTestWithPredictions(test_data, original_predictions, self.param.device, verbose, pos_dict, self.param.n_item, 20)
        print(f'NDCG: {ndcg}, HR: {hr}')
         
        ndcg, hr = baseTestWithPredictions(test_data, oracle_predictions, self.param.device, verbose, pos_dict, self.param.n_item, 20)
        print(f'NDCG: {ndcg}, HR: {hr}')
                
        ndcg, hr = baseTestWithPredictions(test_data, corrected_predictions, self.param.device, verbose, pos_dict, self.param.n_item, 20)
        print(f'NDCG: {ndcg}, HR: {hr}')
        
        for k in [20, 50, 100]:
            metrics = retain_metrics(oracle_predictions, corrected_predictions, k)
            print(f'Retain-NDCG@{k}: {metrics[f"Retain-NDCG@{k}"]}, Retain-Recall@{k}: {metrics[f"Retain-Recall@{k}"]}')
            wandb.log({
                f"Retain-NDCG@{k}": metrics[f"Retain-NDCG@{k}"],
                f"Retain-Recall@{k}": metrics[f"Retain-Recall@{k}"]
            })
        
        
        
        
        wandb.log({
            "original_prob": original_prob,
            "oracle_prob": oracle_prob,
            "corrected_prob": corrected_prob,
            "NDCG": ndcg,
            "HR": hr,
            "original_negative_prob": original_negative_prob,
            "oracle_negative_prob": oracle_negative_prob,
            "corrected_negative_prob": corrected_negative_prob
        })

        self.logger.info(f"Training finished for {self.name}")
        
    
    def read(self, origin=True):
        learn_type = self.param.learn_type
        del_type = self.param.del_type
        del_per = self.param.del_per
        group = self.param.n_group
        train_rating, test_rating, active_rating, inactive_rating = readRating_full(self.param.train_dir,
                                                                                        self.param.test_dir, del_type,
                                                                                        0)

        return train_rating, test_rating, active_rating, inactive_rating
        
        


class task_instance2(Instance):
    def __init__(self, param):
        super().__init__(param)
        

    def runModel(self, model_type='wmf', verbose=2):
        print(self.name, 'begin:')
        self.logger.info(f"{self.name} begin:")
        # read raw data
        train_rating, test_rating, active_rating, inactive_rating = self.read() # list of [user_ids, item_ids, ratings]
        

        if self.param.learn_type == 'task_vector':
            # load data
            if model_type in ['wmf', 'dmf', 'gmf', 'nmf']:
                train_data = loadData(RatingData(train_rating), self.param.batch,
                                      self.param.n_worker,
                                      True)

            elif model_type in ['bpr', 'lightgcn']:
                train_data = loadData(PairData(train_rating, self.param.pos_data), self.param.batch,
                                      self.param.n_worker,
                                      True)

            test_data = loadData(RatingData(test_rating), len(test_rating[0]), self.param.n_worker, False)
            active_test_data = loadData(RatingData(active_rating), len(active_rating[0]), self.param.n_worker,
                                        False)
            inactive_test_data = loadData(RatingData(inactive_rating), len(inactive_rating[0]), self.param.n_worker,
                                          False)

            self.logger.info(f"Train_data_unique: {len(np.unique(train_rating[0].values))}")
            self.logger.info(f'Test_data_unique: {len(np.unique(test_rating[0].values))}')
            
            print(f'Train_data_unique: {len(np.unique(train_rating[0].values))}')
            print(f'Test_data_unique: {len(np.unique(test_rating[0].values))}')
            
            
            if model_type == 'wmf':
                model = WMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
                model_oracle = WMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
            elif model_type == 'bpr':
                model = BPR(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
                model_oracle = BPR(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
            elif model_type == 'gmf':
                model = GMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
                model_oracle = GMF(self.param.n_user, self.param.n_item, self.param.k).to(self.param.device)
            elif model_type == 'nmf':
                model = NMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
                model_oracle = NMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
            elif model_type == 'dmf':
                model = DMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
                model_oracle = DMF(self.param.n_user, self.param.n_item, self.param.k, self.param.layers).to(self.param.device)
                
            # trained on original dataset
            model.load_state_dict(torch.load(self.param.origin_model_path))
            model_oracle.load_state_dict(torch.load(self.param.oracle_model_path))
        
        pos_dict = np.load(self.param.pos_data, allow_pickle=True).item()
        
        # original dataset
        original_rating_matrix = convert_rating_to_tensor(train_rating, self.param.n_user, self.param.n_item).to(self.param.device)
        # Unlearned dataset
        unlearn_ratings = pd.read_csv(self.param.train_dir[:-4]+f'_after_delete_del_per{self.param.del_per}.csv', sep=',')
        unlearn_rating_lists = [unlearn_ratings['uid'], unlearn_ratings['iid'], unlearn_ratings['val']]
        unlearn_rating_matrix = convert_rating_to_tensor(unlearn_rating_lists, self.param.n_user, self.param.n_item).to(self.param.device)
        
        # Original model prediction
        original_predictions = torch.sigmoid(model.predict_matrix())
        oracle_predictions = torch.sigmoid(model_oracle.predict_matrix())
        
        corrected_predictions = joint_svd_unlearning(original_rating_matrix, 
                                                     unlearn_rating_matrix, 
                                                     original_predictions, 
                                                     self.param.alpha, 
                                                     self.param.beta)
        
        ## Evaluation
        target_rating_matrix = original_rating_matrix - unlearn_rating_matrix
        print(target_rating_matrix.sum())
        target_rating_matrix = target_rating_matrix.to(self.param.device)
        original_predictions = original_predictions.to(self.param.device)
        oracle_predictions = oracle_predictions.to(self.param.device)
        corrected_predictions = corrected_predictions.to(self.param.device)
        
        original_prob = torch.sum(original_predictions * target_rating_matrix) / torch.sum(target_rating_matrix)
        oracle_prob = torch.sum(oracle_predictions * target_rating_matrix) / torch.sum(target_rating_matrix)
        corrected_prob = torch.sum(corrected_predictions * target_rating_matrix) / torch.sum(target_rating_matrix)
        print(f'original_prob: {original_prob}, oracle_prob: {oracle_prob}, corrected_prob: {corrected_prob}')
        
        negative_rating = 1- original_rating_matrix
        original_negative_prob = torch.sum(original_predictions * negative_rating) / torch.sum(negative_rating)
        oracle_negative_prob = torch.sum(oracle_predictions * negative_rating) / torch.sum(negative_rating)
        corrected_negative_prob = torch.sum(corrected_predictions * negative_rating) / torch.sum(negative_rating)
        print(f'original_negative_prob: {original_negative_prob}, oracle_negative_prob: {oracle_negative_prob}, corrected_negative_prob: {corrected_negative_prob}')

        
        ndcg, hr = baseTestWithPredictions(test_data, original_predictions, self.param.device, verbose, pos_dict, self.param.n_item, 20)
        print(f'NDCG: {ndcg}, HR: {hr}')
         
        ndcg, hr = baseTestWithPredictions(test_data, oracle_predictions, self.param.device, verbose, pos_dict, self.param.n_item, 20)
        print(f'NDCG: {ndcg}, HR: {hr}')
                
        ndcg, hr = baseTestWithPredictions(test_data, corrected_predictions, self.param.device, verbose, pos_dict, self.param.n_item, 20)
        print(f'NDCG: {ndcg}, HR: {hr}')
        
        for k in [20, 50, 100]:
            metrics = retain_metrics(oracle_predictions, corrected_predictions, k)
            print(f'Retain-NDCG@{k}: {metrics[f"Retain-NDCG@{k}"]}, Retain-Recall@{k}: {metrics[f"Retain-Recall@{k}"]}')
            wandb.log({
                f"Retain-NDCG@{k}": metrics[f"Retain-NDCG@{k}"],
                f"Retain-Recall@{k}": metrics[f"Retain-Recall@{k}"]
            })
        
        
        
        
        wandb.log({
            "original_prob": original_prob,
            "oracle_prob": oracle_prob,
            "corrected_prob": corrected_prob,
            "NDCG": ndcg,
            "HR": hr,
            "original_negative_prob": original_negative_prob,
            "oracle_negative_prob": oracle_negative_prob,
            "corrected_negative_prob": corrected_negative_prob
        })

        self.logger.info(f"Training finished for {self.name}")
        
    
    def read(self, origin=True):
        learn_type = self.param.learn_type
        del_type = self.param.del_type
        del_per = self.param.del_per
        group = self.param.n_group
        train_rating, test_rating, active_rating, inactive_rating = readRating_full(self.param.train_dir,
                                                                                        self.param.test_dir, del_type,
                                                                                        0)

        return train_rating, test_rating, active_rating, inactive_rating
        
        
