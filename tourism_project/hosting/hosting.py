from huggingface_hub import HfApi
import os

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="Shivam174/tourism-prediction-model-host",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
