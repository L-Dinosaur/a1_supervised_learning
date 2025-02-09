from data_processing import *
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from problem import MLProblem


class Spotify(MLProblem):
    def __init__(self, _config_path):
        super().__init__(_config_path)

    def preprocess(self):
        self.df = pd.read_csv(self.config['data_path'])
