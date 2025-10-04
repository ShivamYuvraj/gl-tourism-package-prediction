from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = 'Shivam174/gltourism-prediction-data'
repo_type = "dataset"

create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(folder_path='tourism_project/data', repo_id=repo_id, repo_type='dataset')
