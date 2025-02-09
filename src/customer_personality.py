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


class CustomerPersonality(object):

    def __init__(self):
        super().__init__()
        self.df = None
        with open('../config/customer_personality.yaml', 'r') as strm:
            self.config = yaml.safe_load(strm)
        self.models = {}
        self.X = None
        self.y = None
        self.X_train_val = None
        self.X_val = None
        self.X_test = None
        self.y_train_val = None
        self.y_val = None
        self.y_test = None
        self.train_idx = []
        self.val_idx = []
        self.neural_net_config = self.config['neural_net']
        self.svm_config = self.config['svm']
        self.knn_config = self.config['knn']

    def process_data(self):
        self.df = pd.read_csv(self.config['data_path'], delimiter='\t')

        # Step 1: Drop unused variables:
        self.df.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

        # Step 1.5: Data Cleaning
        self.df.fillna(0, inplace=True)

        # Education
        self.df.loc[self.df['Education'] == '2n Cycle', 'Education'] = 'Master'

        # Marital Status
        self.df = self.df[~self.df['Marital_Status'].isin(['Absurd', 'YOLO'])]
        self.df.loc[self.df['Marital_Status'] == 'Together', 'Marital_Status'] = 'Married'
        self.df.loc[self.df['Marital_Status'] == 'Alone', 'Marital_Status'] = 'Single'

        # Step 2: Normalize Features
        self.df['Age'] = 2025 - self.df['Year_Birth']
        self.df.drop(['Year_Birth'], axis=1, inplace=True)

        self.df['Tenure'] = (pd.to_datetime('2025-02-08') - pd.to_datetime(self.df['Dt_Customer'])).dt.days
        self.df.drop(['Dt_Customer'], axis=1, inplace=True)

        summarize_df(self.df)

        self.X = pd.get_dummies(self.df.drop(['Response'], axis=1), columns=['Education', 'Marital_Status'])
        self.y = self.df['Response']

        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X, self.y)

        scaler = StandardScaler()
        self.X_train_val = scaler.fit_transform(self.X_train_val)
        self.X_test = scaler.transform(self.X_test)

        joblib.dump(scaler, 'scaler.pkl')

    def set_up_model(self):
        if 'neural_net' in self.config['models']:
            self.models['neural_net'] = MLPClassifier(**self.neural_net_config)
        if 'svm' in self.config['models']:
            self.models['svm'] = SVC(**self.svm_config)
        if 'knn' in self.config['models']:
            self.models['knn'] = KNeighborsClassifier(**self.knn_config)

    def train(self):
        # Neural Net
        nn_param_grid = {'hidden_layer_sizes': [(25,), (50,), (100,), (50, 50), (50, 50, 50)],
                         'alpha': np.logspace(-3, 3, 7),
                         'activation': ['relu', 'logistic']}
        nn_gs = GridSearchCV(
            self.models['neural_net'],
            param_grid=nn_param_grid,
            cv=5,
            n_jobs=1,
            scoring='accuracy',
            verbose=1
        )

        nn_gs.fit(self.X_train_val, self.y_train_val)
        joblib.dump(nn_gs, 'nn_model_grid_search.pkl')

        # SVM
        svm_param_grid = {'C': np.logspace(-3, 3, 7),
                          'gamma': np.logspace(-3, 3, 7),
                          'kernel': ['rbf', 'sigmoid']}
        svm_gs = GridSearchCV(
            self.models['svm'],
            param_grid=svm_param_grid,
            cv=5,
            n_jobs=1,
            verbose=1
        )

        svm_gs.fit(self.X_train_val, self.y_train_val)
        joblib.dump(svm_gs, 'svm_model_grid_search.pkl')

        # KNN
        knn_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
        knn_gs = GridSearchCV(
            self.models['knn'],
            param_grid=knn_grid,
            cv=5,
            n_jobs=1,
            verbose=1
        )

        knn_gs.fit(self.X_train_val, self.y_train_val)

        joblib.dump(knn_gs, 'knn_model_grid_search.pkl')


if __name__ == '__main__':
    cp = CustomerPersonality()
    cp.process_data()
    cp.set_up_model()
    cp.train()
