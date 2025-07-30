import sys
import os
import numpy as np
import time
from torch import nn, optim
import torch

from utils import seed_all, baseTrain, baseTest
from utils import WMF, DMF, GMF, NMF, BPR

import wandb

class Scratch(object):
    def __init__(self, param, model_type):
        # model param
        self.n_user = param.n_user
        self.n_item = param.n_item
        self.k = param.k
        self.model_type = model_type

        # training param
        self.seed = param.seed
        self.lr = param.lr
        self.epochs = param.epochs
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.pos_dir = param.pos_data
        
        self.param_learn = param.learn_type
        self.param_deltype = param.del_type
        self.param_delper = param.del_per
        self.dataset = param.dataset
        
        self.save_dir = f'model_params/{self.param_learn}/{self.model_type}_{self.dataset}_{self.param_deltype}_{self.param_delper}'
        os.makedirs(self.save_dir, exist_ok=True)

        # log
        self.log = {
            'train_loss': [],
            'test_ndcg': [],
            'test_hr': [],
            'test_recall': [],
            'active_ndcg': [],
            'active_hr': [],
            'active_recall': [],
            'inactive_ndcg': [],
            'inactive_hr': [],
            'inactive_recall': [],
            'time': []
        }

        if self.model_type in ['wmf']:
            self.loss_fn = 'point-wise'
        elif self.model_type == 'bpr':
            self.loss_fn = 'pair-wise'
        elif self.model_type in ['dmf', 'gmf', 'nmf']:
            self.layers = param.layers
            self.loss_fn = 'point-wise'

    def train(self, train_data, test_data, active_test_data, inactive_test_data, verbose=2, save_dir='',
              id=0, given_model=''):
        print('Using device:', self.device)
        seed_all(self.seed)

        # build model
        if given_model == '':
            if self.model_type == 'wmf':
                model = WMF(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'bpr':
                model = BPR(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'gmf':
                model = GMF(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'nmf':
                model = NMF(self.n_user, self.n_item, self.k, self.layers).to(self.device)
            elif self.model_type == 'dmf':
                model = DMF(self.n_user, self.n_item, self.k, self.layers).to(self.device)
        else:
            model = given_model.to(self.device)

        opt = optim.Adam(model.parameters(), lr=self.lr)

        # main loop
        best_ndcg, best_hr, best_recall = 0, 0, 0
        count_dec, total_time = 0, 0

        pos_dict = np.load(self.pos_dir, allow_pickle=True).item()

        for t in range(self.epochs):
            print(f'Epoch: [{t + 1:>3d}/{self.epochs:>3d}] --------------------')
            epoch_start = time.time()

            # train
            train_loss = baseTrain(train_data, model, self.loss_fn, opt, self.device, verbose)
            train_time = time.time() - epoch_start
            total_time += train_time

            user_mapping = None
            pos_mapping = None

            # test metrics
            test_ndcg, test_hr, test_recall = baseTest(test_data, model, self.loss_fn, self.device, verbose, pos_dict, self.n_item, 20)
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start))

            print('Time:', epoch_time)
            print('train_loss:', train_loss)
            print('test_ndcg:', test_ndcg)
            print('test_hr:', test_hr)
            print('test_recall:', test_recall)

            # 로그 저장
            self.log['train_loss'].append(train_loss)
            self.log['test_ndcg'].append(test_ndcg)
            self.log['test_hr'].append(test_hr)
            self.log['test_recall'].append(test_recall)
            self.log['time'].append(epoch_time)

            # wandb log
            wandb.log({
                "epoch": t + 1,
                "train_loss": train_loss,
                "test_ndcg": test_ndcg,
                "test_hr": test_hr,
                "test_recall": test_recall
            })

            # Early stopping
            if test_ndcg > best_ndcg:
                count_dec = 0
                best_ndcg, best_hr, best_recall = test_ndcg, test_hr, test_recall
                torch.save(model.state_dict(), self.save_dir + '/model.pth')
                torch.save(model.user_mat.weight.detach().cpu().numpy(), self.save_dir + '/user_mat.npy')
            else:
                count_dec += 1

            if count_dec > 5:
                break

        # Active evaluation
        if active_test_data is not None:
            active_ndcg, active_hr, active_recall = baseTest(active_test_data, model, self.loss_fn, self.device, verbose, pos_dict, self.n_item, 20)
            print('active_test_ndcg:', active_ndcg)
        else:
            active_ndcg, active_hr, active_recall = 0, 0, 0

        # Inactive evaluation
        inactive_ndcg, inactive_hr, inactive_recall = baseTest(inactive_test_data, model, self.loss_fn, self.device, verbose, pos_dict, self.n_item, 20)
        print('inactive_test_ndcg:', inactive_ndcg)

        # 저장
        self.log['active_ndcg'].append(active_ndcg)
        self.log['active_hr'].append(active_hr)
        self.log['active_recall'].append(active_recall)
        self.log['inactive_ndcg'].append(inactive_ndcg)
        self.log['inactive_hr'].append(inactive_hr)
        self.log['inactive_recall'].append(inactive_recall)

        # wandb log active/inactive
        wandb.log({
            "active_ndcg": active_ndcg,
            "active_hr": active_hr,
            "active_recall": active_recall,
            "inactive_ndcg": inactive_ndcg,
            "inactive_hr": inactive_hr,
            "inactive_recall": inactive_recall
        })

        print('-------best--------')
        result = {
            'time': total_time,
            'ndcg': best_ndcg,
            'hr': best_hr,
            'recall': best_recall,
            'active_ndcg': active_ndcg,
            'active_hr': active_hr,
            'active_recall': active_recall,
            'inactive_ndcg': inactive_ndcg,
            'inactive_hr': inactive_hr,
            'inactive_recall': inactive_recall
        }

        return model, result