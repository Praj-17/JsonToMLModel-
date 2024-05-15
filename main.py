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

# Load JSON input
with open('input.json', 'r') as f:
    input_data = json.load(f)

# Load dataset
dataset_path = input_data['design_state_data']['session_info']['dataset']
data = pd.read_csv(dataset_path)

# Separate features and target
target_column = input_data['design_state_data']['target']['target']
X = data.drop(columns=[target_column])
y = data[target_column]

# Preprocessing pipeline
numeric_features = []
numeric_transformers = []
categorical_features = []
categorical_transformers = []

for feature, details in input_data['design_state_data']['feature_handling'].items():
    if details['feature_variable_type'] == 'numerical':
        numeric_features.append(feature)
        imputer = SimpleImputer(strategy='mean' if details['feature_details']['missing_values'] == 'Impute' else 'constant', fill_value=details['feature_details']['impute_value'])
        
        scaler = StandardScaler() if details['feature_details']['rescaling'] == 'No rescaling' else None  # Add more rescaling options if needed
        transformers = [('imputer', imputer)]
        if scaler:
            transformers.append(('scaler', scaler))
        numeric_transformers.append((feature, Pipeline(transformers)))
    elif details['feature_variable_type'] == 'text':
        categorical_features.append(feature)
        text_vectorizer = HashingVectorizer(n_features=details['feature_details']['hash_columns'], alternate_sign=False, norm=None)
        categorical_transformers.append((feature, text_vectorizer))

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline(numeric_transformers), numeric_features),
        ('categorical', ColumnTransformer(transformers=categorical_transformers), categorical_features)])

# Feature Engineering
X = preprocessor.fit_transform(X)

# Partitioning
train_config = input_data['design_state_data']['train']
if train_config['policy'] == 'Split the dataset':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Creation and Hyperparameter Tuning
rf_params = input_data['design_state_data']['algorithms']['RandomForestRegressor']
param_grid = {
    'n_estimators': range(rf_params['min_trees'], rf_params['max_trees'] + 1),
    'max_depth': range(rf_params['min_depth'], rf_params['max_depth'] + 1),
    'min_samples_leaf': range(rf_params['min_samples_per_leaf_min_value'], rf_params['min_samples_per_leaf_max_value'] + 1)
}

rf_model = RandomForestRegressor()
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Model Evaluation and Selection
y_pred = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
