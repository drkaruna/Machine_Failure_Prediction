import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "week_2_mls/data/machine-failure-prediction.csv"

api.upload_file(
    path_or_fileobj=DATASET_PATH,
    path_in_repo="machine-failure-prediction.csv",
    repo_id="Verallon/Machine-Failure-Prediction",
    repo_type="dataset"
)

DATASET_PATH = "hf://datasets/Verallon/Machine-Failure-Prediction/machine-failure-prediction.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

df.drop(columns=['UDI'], inplace=True)

label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

target_col = 'Failure'

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="Verallon/Machine-Failure-Prediction",
        repo_type="dataset",
    )
