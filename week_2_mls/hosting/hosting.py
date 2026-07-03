from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="week_2_mls/deployment",
    repo_id="Verallon/Machine-Failure-Prediction",
    repo_type="space",
    path_in_repo="",
)
