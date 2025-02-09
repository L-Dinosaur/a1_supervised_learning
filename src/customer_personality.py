from data_processing import *
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from problem import MLProblem


class CustomerPersonality(MLProblem):

    def __init__(self, _config_path):
        super().__init__(_config_path)
        self.cv_splitter = KFold(**self.config['cv'])

    def preprocess(self):
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
        self.df.drop(['Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Age'], axis=1, inplace=True)
        # self.df.drop(['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
        #               'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        #               'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
        #               'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Recency', 'Tenure'], axis=1, inplace=True)
        summarize_df(self.df)

        # self.X = pd.get_dummies(self.df.drop(['Response'], axis=1), columns=['Education', 'Marital_Status'])
        self.X = self.df.drop(['Response'], axis=1)
        self.y = self.df['Response']

        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X, self.y)

        scaler = StandardScaler()
        self.X_train_val = scaler.fit_transform(self.X_train_val)
        self.X_test = scaler.transform(self.X_test)

        joblib.dump(scaler, self.model_save_path + '/scaler.pkl')

        self.nn_param_grid = {'hidden_layer_sizes': [(25,), (50,), (100,), (50, 50), (50, 50, 50)],
                              'alpha': np.logspace(-3, 3, 7),
                              'activation': ['relu', 'logistic']}
        self.svm_param_grid = {'C': np.logspace(-3, 3, 7),
                               'gamma': np.logspace(-3, 3, 7),
                               'kernel': ['rbf', 'sigmoid']}
        self.knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}


if __name__ == '__main__':
    np.random.seed(329)
    cp = CustomerPersonality('../config/customer_personality.yaml')
    cp.preprocess()
    cp.set_up_model()
    cp.train()
    cp.test()
    cp.learning_curve(np.linspace(0.1, 1, cp.X_train_val.shape[0]))
    cp.validation_curve()
