import yaml
import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import confusion_matrix, classification_report
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

        self.cv_splitter = None

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
            cv=self.cv_splitter,
            n_jobs=1,
            scoring='accuracy',
            verbose=1
        )

        self.nn_gs.fit(self.X_train_val, self.y_train_val)
        self.nn = self.nn_gs.best_estimator_
        joblib.dump(self.nn_gs, os.path.join(self.model_save_path, 'neural_net_gridsearch.pkl'))

        self.svm_gs = GridSearchCV(
            self.svm,
            param_grid=self.svm_param_grid,
            cv=self.cv_splitter,
            n_jobs=1,
            scoring='accuracy',
            verbose=1,
        )
        self.svm_gs.fit(self.X_train_val, self.y_train_val)
        joblib.dump(self.svm_gs, os.path.join(self.model_save_path, 'svm_gridsearch.pkl'))
        self.svm = self.svm_gs.best_estimator_

        self.knn_gs = GridSearchCV(
            self.knn,
            param_grid=self.knn_param_grid,
            cv=self.cv_splitter,
            n_jobs=1,
            scoring='accuracy',
            verbose=1,
        )
        self.knn_gs.fit(self.X_train_val, self.y_train_val)
        joblib.dump(self.knn_gs, os.path.join(self.model_save_path, 'knn_gridsearch.pkl'))
        self.knn = self.knn_gs.best_estimator_

    def test(self):
        # Neural Net
        print('Performance Evaluation:')

        print(f'Neural Network with parameters {self.nn_gs.best_params_}')
        nn_y_pred = self.nn.predict(self.X_test)
        print('Classification Report:')
        nn_report = classification_report(self.y_test, nn_y_pred)
        print(nn_report)

        nn_confusion = confusion_matrix(self.y_test, nn_y_pred)
        print('Confusion Matrix:')
        print(nn_confusion)

        # SVM
        print(f'SVM with parameters {self.svm_gs.best_params_}')
        svm_y_pred = self.svm.predict(self.X_test)
        print('Classification Report:')
        svm_report = classification_report(self.y_test, svm_y_pred)
        print(svm_report)

        svm_confusion = confusion_matrix(self.y_test, svm_y_pred)
        print('Confusion Matrix:')
        print(svm_confusion)

        # KNN
        print(f'K Nearest Neighbors with parameters {self.knn_gs.best_params_}')
        knn_y_pred = self.knn.predict(self.X_test)
        print('Classification Report:')
        knn_report = classification_report(self.y_test, knn_y_pred)
        print(knn_report)

        knn_confusion = confusion_matrix(self.y_test, knn_y_pred)
        print('Confusion Matrix:')
        print(knn_confusion)

    def learning_curve(self, training_size):
        # Learning Curve
        # Neural Net
        train_sizes, nn_train_scores, nn_val_scores = learning_curve(
            self.nn,
            self.X_train_val,
            self.y_train_val,
            train_sizes=training_size,
            n_jobs=1,
            shuffle=True,
            random_state=329
        )
        nn_learning_curve_df = pd.DataFrame(nn_train_scores, index=train_sizes)
        nn_val_curve_df = pd.DataFrame(nn_val_scores)
        nn_val_curve_df.to_csv(self.model_save_path + '/val_curve_general_nn.csv')
        nn_learning_curve_df.to_csv(self.model_save_path + '/learning_curve_nn.csv')

        # SVM
        train_sizes, svm_train_scores, svm_val_scores = learning_curve(
            self.svm,
            self.X_train_val,
            self.y_train_val,
            train_sizes=training_size,
            n_jobs=1,
            shuffle=True,
            random_state=329
        )
        svm_learning_curve_df = pd.DataFrame(svm_train_scores, index=train_sizes)
        svm_val_curve_df = pd.DataFrame(svm_val_scores)
        svm_val_curve_df.to_csv(self.model_save_path + '/val_curve_general_svm.csv')
        svm_learning_curve_df.to_csv(self.model_save_path + '/learning_curve_svm.csv')

        # K Nearest Neighbors
        train_sizes, knn_train_scores, knn_val_scores = learning_curve(
            self.knn,
            self.X_train_val,
            self.y_train_val,
            train_sizes=training_size,
            n_jobs=1,
            shuffle=True,
            random_state=329
        )
        knn_learning_curve_df = pd.DataFrame(knn_train_scores, index=train_sizes)
        knn_val_curve_df = pd.DataFrame(knn_val_scores)
        knn_val_curve_df.to_csv(self.model_save_path + '/val_curve_general_knn.csv')
        knn_learning_curve_df.to_csv(self.model_save_path + '/learning_curve_knn.csv')

    def validation_curve(self):
        # Neural Net
        for pname, prange in self.nn_param_grid.items():
            train_scores, val_scores = validation_curve(
                estimator=self.nn,
                X=self.X_train_val,
                y=self.y_train_val,
                param_name=pname,
                param_range=prange,
                cv=self.cv_splitter,
                scoring='accuracy'
            )
            nn_val_curve = pd.DataFrame(val_scores, index=prange)
            nn_val_curve.to_csv(self.model_save_path + f'/nn_{pname}_val_curve.csv')

        # SVM
        for pname, prange in self.svm_param_grid.items():
            train_scores, val_scores = validation_curve(
                estimator=self.svm,
                X=self.X_train_val,
                y=self.y_train_val,
                param_name=pname,
                param_range=prange,
                cv=self.cv_splitter,
                scoring='accuracy'
            )
            svm_val_curve = pd.DataFrame(val_scores, index=prange)
            svm_val_curve.to_csv(self.model_save_path + f'/svm_{pname}_val_curve.csv')

        # Neural Net
        for pname, prange in self.knn_param_grid.items():
            train_scores, val_scores = validation_curve(
                estimator=self.knn,
                X=self.X_train_val,
                y=self.y_train_val,
                param_name=pname,
                param_range=prange,
                cv=self.cv_splitter,
                scoring='accuracy'
            )
            knn_val_curve = pd.DataFrame(val_scores, index=prange)
            knn_val_curve.to_csv(self.model_save_path + f'/knn_{pname}_val_curve.csv')
