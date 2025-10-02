from huggingface_hub import HfApi, login, create_repo
from google.colab import userdata
import os


hf_token = userdata.get("HFTOKEN")
login(hf_token)
# api = HfApi(token=userdata.get('HFTOKEN'))
print(userdata.get('HFTOKEN'))
repo_id = 'Shivam174/tourism-prediction'

create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
api = HfApi()
api.upload_folder(folder_path='/content/tourism_project', repo_id=repo_id, repo_type='dataset')
