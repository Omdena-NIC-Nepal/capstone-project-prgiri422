import os
import gdown
import pandas as pd
import joblib
import pickle
from pathlib import Path
from config import USE_GOOGLE_DRIVE, FILE_MAPPINGS, LOCAL_PATHS, get_file_path
from urllib.error import HTTPError

def load_file(category, file_key):
    """Load a file from configured source (Google Drive or local)"""
    try:
        file_info = get_file_path(category, file_key)
        
        if USE_GOOGLE_DRIVE:
            if not isinstance(file_info, dict) or 'file_id' not in file_info:
                raise ValueError("Invalid Google Drive file info")
            return _download_from_drive(file_info['file_id'], file_info['filename'])
        else:
            if not isinstance(file_info, str):
                raise ValueError("Invalid local file path")
            return _load_by_file_type(file_info)
            
    except Exception as e:
        raise IOError(f"Error loading {category}/{file_key}: {str(e)}")

def load_model(model_name):
    """Load a trained model by name"""
    try:
        model_mapping = {
            'temp': 'temp_model',
            'climate_zone': 'climate_zone_model',
            'vulnerability': 'vulnerability_model'
        }
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model type. Available: {list(model_mapping.keys())}")
        return load_file("models", model_mapping[model_name])
    except Exception as e:
        raise IOError(f"Error loading model {model_name}: {str(e)}")

def save_model(model, category, file_key):
    """Save a model to local filesystem"""
    try:
        if USE_GOOGLE_DRIVE:
            raise NotImplementedError("Google Drive save not implemented yet")
        
        file_path = get_file_path(category, file_key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_path.endswith(('.pkl', '.pickle')):
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        elif file_path.endswith('.joblib'):
            joblib.dump(model, file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return True
    except Exception as e:
        raise IOError(f"Error saving model: {str(e)}")

def _download_from_drive(file_id, filename):
    """Download a file from Google Drive"""
    temp_dir = "temp_downloads"
    Path(temp_dir).mkdir(exist_ok=True)
    output_path = os.path.join(temp_dir, filename)
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Download failed for {filename}")
            
        return _load_by_file_type(output_path)
    except HTTPError as e:
        if e.code == 403:
            manual_url = f"https://drive.google.com/file/d/{file_id}/view"
            raise PermissionError(
                f"Permission denied. Please manually download from:\n"
                f"{manual_url}\n"
                f"And place the file in your local folder"
            ) from e
        raise
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

def _load_by_file_type(filepath):
    """Load different file types with appropriate handlers"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.pkl', '.pickle')):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.joblib'):
            return joblib.load(filepath)
        elif filepath.endswith(('.png', '.jpg', '.jpeg')):
            return filepath  # Return path for images
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    except Exception as e:
        raise IOError(f"Error loading {filepath}: {str(e)}")

# Convenience functions
def load_raw_data():
    return load_file("data_raw", "climate_data")

def load_clean_data():
    return load_file("data_processed", "clean_data")

def load_featured_data():
    return load_file("data_processed", "featured_data")