import yaml
from data_processing import *
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split


class CustomerPersonality(object):

    def __init__(self):
        super().__init__()
        self.df = None
        with open('../config/customer_personality.yaml', 'r') as strm:
            self.config = yaml.safe_load(strm)
        self.neural_net = None
        self.svm = None
        self.knn = None
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

        # Step 2: Normalize Features
        self.df['Age'] = 2025 - self.df['Year_Birth']
        self.df.drop(['Year_Birth'], axis=1, inplace=True)
        # self.df.loc[:, 'Age'] = normalize(self.df['Age'])

        # self.df.loc[:, 'Income'] = normalize(self.df['Income'])

        self.df['Tenure'] = (pd.to_datetime('2025-02-08') - pd.to_datetime(self.df['Dt_Customer'])).dt.days
        # self.df.loc[:, 'Tenure'] = normalize(self.df['Tenure'])

        # self.df.loc[self.df['Recency'] == 0, 'Recency'] = 1
        # self.df.loc[:, 'Recency'] = inverse_normalize(self.df['Recency'])

        # self.df.loc[:, 'MntWines'] = normalize(self.df['MntWines'])
        # self.df.loc[:, 'MntFruits'] = normalize(self.df['MntFruits'])
        # self.df.loc[:, 'MntMeatProducts'] = normalize(self.df['MntMeatProducts'])
        # self.df.loc[:, 'MntFishProducts'] = normalize(self.df['MntFishProducts'])
        # self.df.loc[:, 'MntSweetProducts'] = normalize(self.df['MntSweetProducts'])
        # self.df.loc[:, 'NumDealsPurchases'] = normalize(self.df['NumDealsPurchases'])
        # self.df.loc[:, 'NumWebPurchases'] = normalize(self.df['NumWebPurchases'])
        # self.df.loc[:, 'NumCatalogPurchases'] = normalize(self.df['NumCatalogPurchases'])
        # self.df.loc[:, 'NumStorePurchases'] = normalize(self.df['NumStorePurchases'])
        # self.df.loc[:, 'NumWebVisitsMonth'] = normalize(self.df['NumWebVisitsMonth'])
        # self.df.loc[:, 'MntGoldProds'] = normalize(self.df['MntGoldProds'])

        summarize_df(self.df)

        self.X = self.df.drop(['Response'], axis=1)
        self.y = self.df['Response']

        x_train_val, self.X_test, y_train_val, self.y_test = train_test_split(self.X, self.y)

        kf = KFold(n_splits=5, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_val)):
            self.train_idx[fold] = train_idx
            self.val_idx[fold] = val_idx

    def set_up_model(self):
        self.neural_net = MLPClassifier(**self.neural_net_config)

    def run(self):
        # Neural Network

        print('run')


if __name__ == '__main__':
    cp = CustomerPersonality()
    cp.process_data()
