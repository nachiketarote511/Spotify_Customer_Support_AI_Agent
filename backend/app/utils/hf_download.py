import os
import shutil
from huggingface_hub import snapshot_download

def download_models_if_missing(repo_id: str, repo_root: str):
    """
    Downloads model artifacts from Hugging Face if the local models directory is empty or missing.
    """
    models_dir = os.path.join(repo_root, 'models')
    
    # Check if models exist (assuming intent_classifier and embedding_index are required)
    intent_dir = os.path.join(models_dir, 'intent_classifier', 'baseline')
    faiss_dir = os.path.join(models_dir, 'embedding_index')
    
    missing_models = False
    if not os.path.exists(intent_dir) or not os.listdir(intent_dir):
        missing_models = True
    if not os.path.exists(faiss_dir) or not os.listdir(faiss_dir):
        missing_models = True
        
    if not missing_models:
        print(f"Models already exist in {models_dir}. Skipping download.")
        return

    print(f"Models missing in {models_dir}. Downloading from Hugging Face repo: {repo_id}...")
    try:
        # Download the snapshot of the repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["models/*", "data/golden/*"], # Include golden set for evaluation if needed
            local_dir=repo_root,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded models to {local_dir}")
    except Exception as e:
        print(f"Error downloading models from Hugging Face: {e}")
        raise e
