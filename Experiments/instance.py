import os, warnings, logging
from os.path import exists
import numpy as np
from scratch import Scratch
from read import RatingData, PairData, loadData, readRating_full, readRating_group
from utils import saveObject
import wandb

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'result'))

class Instance(object):
    def __init__(self, param):
        self.param = param
        prefix = '/test/' if self.param.del_type == 'test' else f"/{self.param.del_per}/{self.param.del_type}/"
        self.name = prefix + f"{self.param.dataset}_g_{self.param.n_group}"
        param_dir = SAVE_DIR + self.name
        if not exists(param_dir):
            os.makedirs(param_dir)
        saveObject(param_dir + '/param', self.param)
        self.logger = logging.getLogger(__name__)
        warnings.simplefilter('ignore')

    def read(self):
        if self.param.learn_type == 'retrain':
            return readRating_full(self.param.train_dir, self.param.test_dir, self.param.del_type, self.param.del_per)
        return readRating_group(self.param.train_dir, self.param.test_dir, self.param.del_type,
                                self.param.del_per, self.param.learn_type, self.param.n_group, self.param.dataset)

    def runModel(self, model_type='wmf', verbose=2):
        print(self.name, 'begin:')
        self.logger.info(f"{self.name} begin:")
        train_rating, test_rating, active_rating, inactive_rating = self.read()
        group = self.param.n_group

        if self.param.learn_type == 'retrain':
            model = Scratch(self.param, model_type)
            train_data = loadData(RatingData(train_rating), self.param.batch, self.param.n_worker, True)
            test_data = loadData(RatingData(test_rating), len(test_rating[0]), self.param.n_worker, False)
            active_test = loadData(RatingData(active_rating), len(active_rating[0]), self.param.n_worker, False)
            inactive_test = loadData(RatingData(inactive_rating), len(inactive_rating[0]), self.param.n_worker, False)

            model, result = model.train(train_data, test_data, active_test, inactive_test, verbose)
            wandb.log(result)
        else:
            for i in range(group):
                model = Scratch(self.param, model_type)
                train_data = loadData(RatingData(train_rating[i]), self.param.batch, self.param.n_worker, True)
                test_data = loadData(RatingData(test_rating[i]), len(test_rating[i][0]), self.param.n_worker, False)
                active_test = loadData(RatingData(active_rating[i]), len(active_rating[i][0]), self.param.n_worker, False)
                inactive_test = loadData(RatingData(inactive_rating[i]), len(inactive_rating[i][0]), self.param.n_worker, False)

                model, result = model.train(train_data, test_data, active_test, inactive_test, verbose)
                wandb.log({f"Group{i+1}_" + k: v for k, v in result.items()})

    def run(self, verbose=2):
        self.runModel(self.param.model, verbose)