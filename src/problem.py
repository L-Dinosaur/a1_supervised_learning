import yaml
from data_processing import *
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import os


class MLProblem(object):

    def __init__(self, _config_path):
        super().__init__()
        self.df = None
        model_save_path_base = '../output'

        with open(_config_path) as strm:
            self.config = yaml.safe_load(strm)
        if 'model_save_path_base' in self.config:
            model_save_path_base = self.config['model_save_path_base']

        try:
            os.mkdir(model_save_path_base)
        except FileExistsError:
            print(f'Model save path already exists.')

        now = pd.Timestamp.now()
        self.model_save_path = os.path.join(model_save_path_base, f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}')

        os.mkdir(self.model_save_path)

        self.nn = None
        self.svm = None
        self.knn = None
        self.X = None
        self.y = None
        self.X_train_val = None
        # self.X_val = None
        self.X_test = None
        self.y_train_val = None
        # self.y_val = None
        self.y_test = None
        # self.train_idx = []
        # self.val_idx = []
        self.neural_net_config = self.config['neural_net']
        self.svm_config = self.config['svm']
        self.knn_config = self.config['knn']

        self.nn_param_grid = {}
        self.svm_param_grid = {}
        self.knn_param_grid = {}

        # self.nn_param_grid = self.config['neural_net_param_grid']
        # self.svm_param_grid = self.config['svm_param_grid']
        # self.knn_param_grid = self.config['knn_param_grid']

        self.nn_gs = None
        self.svm_gs = None
        self.knn_gs = None

    def set_up_model(self):
        if 'neural_net' in self.config['models']:
            self.nn = MLPClassifier(**self.neural_net_config)
        if 'svm' in self.config['models']:
            self.svm = SVC(**self.svm_config)
        if 'knn' in self.config['models']:
            self.knn = KNeighborsClassifier(**self.knn_config)

    def preprocess(self):
        # Implemented at child class

        # Cleanse and prep data for fitting

        # Set up grid search parameters
        pass

    def train(self):
        self.nn_gs = GridSearchCV(
            self.nn,
            param_grid=self.nn_param_grid,
            cv=5,
            n_jobs=1,
            scoring='accuracy',
            verbose=1
        )

        self.nn_gs.fit(self.X_train_val, self.y_train_val)
        joblib.dump(self.nn_gs, os.path.join(self.model_save_path, 'neural_net_gridsearch.pkl'))

        self.svm_gs = GridSearchCV(
            self.svm,
            param_grid=self.svm_param_grid,
            cv=5,
            n_jobs=1,
            scoring='accuracy',
            verbose=1
        )
        self.svm_gs.fit(self.X_train_val, self.y_train_val)
        joblib.dump(self.svm_gs, os.path.join(self.model_save_path, 'svm_gridsearch.pkl'))

        self.knn_gs = GridSearchCV(
            self.knn,
            param_grid=self.knn_param_grid,
            cv=5,
            n_jobs=1,
            scoring='accuracy',
            verbose=1
        )
        self.knn_gs.fit(self.X_train_val, self.y_train_val)
        joblib.dump(self.knn_gs, os.path.join(self.model_save_path, 'knn_gridsearch.pkl'))

    # def testing(self):
    #     # Neural Net
    #
    #     nn_y_pred =
