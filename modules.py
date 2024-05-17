import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb  
import numpy as np

import os


if __name__ == "__main__":
    # Load JSON input
    with open('input.json', 'r') as f:
        input_data = json.load(f)

    # Load dataset
    dataset_path = input_data['design_state_data']['session_info']['dataset']
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
    else:
        raise FileExistsError("The Specified CSV not found")
    
        # Separate features and target
    target_column = input_data['design_state_data']['target']['target']
    X = data
    y = data[target_column]