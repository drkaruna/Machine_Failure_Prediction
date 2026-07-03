import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi()

Xtrain_path = "hf://datasets/Verallon/Machine-Failure-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/Verallon/Machine-Failure-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/Verallon/Machine-Failure-Prediction/ytrain.csv"
ytest_path = "hf://datasets/Verallon/Machine-Failure-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

numeric_features = ['Air temperature','Process temperature','Rotational speed','Torque','Tool wear']
categorical_features = ['Type']

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

joblib.dump(best_model, "best_machine_failure_model_v1.joblib")

repo_id = "Verallon/Machine-Failure-Prediction"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{repo_id}' not found. Creating new one...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model repo '{repo_id}' created successfully.")

api.upload_file(
    path_or_fileobj="best_machine_failure_model_v1.joblib",
    path_in_repo="best_machine_failure_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
