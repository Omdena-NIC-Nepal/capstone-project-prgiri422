import os
from pathlib import Path

# System Configuration
USE_GOOGLE_DRIVE = True  # Set True if using Google Drive

# Google Drive File IDs (if using Google Drive)
GDRIVE_FILE_IDS = {
    "climate_raw.csv": "1WgWsb1splBnnS6g52EeiqE6V0JGNCfv6",
    "climate_clean.csv": "1sw52qLiDwdOP2LzodnK1EhUuLf8Nq1-O",
    "feature_engineered_climate_data.csv": "1TEZmWB95L_ZZEzXbrZlMKfXmEaOz7hRN",
    "ridge_temp.pkl": "1zrRF9ORh58D0E0_s0zit3RCeM1k6w-PO",
    "rf_climate_zone.pkl": "1HA8IhZTEq9KZ5Rl0l9bbd5AP0WKypkSu",
    "gbc_vulnerability.pkl": "1A_sN2CqkFzi7LEAfGD8FyhtOR1n7YBdY",
    "ridge_temp.py": "1ZFICLAskL2bwMAcjEOwnIiEtI1X4negV",
    "rf_climate_zone.py": "1hMsBfaNgIRpDvV1gy1iAsdFFv8Eiwi9g",
    "processed_climate_data.py": "18v-_jjQmYnY8OL7_hdnnlHxxg20jTqmf",
    "gbr_temp.py": "1DX7U11_aCVdFTnSoGNnKN5dO2wKvVzTQ",
    "gbc_vulnerability.py": "1jbR8mv1dW2Gor5JzkXhJias0QffV0h5H",
    "feature_engineering.py": "1u8qpc93SGcS9SAsspqieACC_w9ZYbrMA",
    "eda.py": "1kGIfuXXYGdfwVFianGaTnaleA0ti9Ufx",
    "ridge_temp_pred.png": "19a3DRRmUI55XVNSa4Ll9Sb7E0xrZhN1r",
    "rf_climate_zone_cm.png": "1sOjWZvO5bgd7jAUE1vvIpv7MKPrbEwwE",
    "gbr_temp_pred.png": "19m670h4g4Bw7ihCwegxp1KRIk2V92RAh",
    "gbr_temp_diagnostics.png": "1WB620-3FQQcne26YNT1qDBvYkJB2YHVR",
    "gbc_vulnerability_cm.png": "1b2Hg6B9YD0gsxlpr4392vZziLXhbIta1",
    "yearly_avg_temp.png": "1TG0AgM6gUQFlbdLmFFHqCqMATKMe_F32",
    "yearly_avg_precip.png": "15bamrvLVIdSrwbJjFxrMeqEWUVeudQKD",
    "temp_vs_precip_scatter.png": "1czG8fUDnAgrYlFFY0_9gv7MOYQvHe9wf",
    "temp_distribution.png": "1UW38MV2KCjXFYsgX7v--vECcy8TzM1Op",
    "season_temp.png": "1ePd6uW-awxuQV9cDEFRdPH0l6V1qVJIO",
    "precip_distribution.png": "1w05m_F52u9Wy6jle-OT3tJTiSPDUYPME",
    "max_temp_distribution.png": "14xrDFnSvixnnNpJ8_WzD474JkbWm_85L",
    "district_temp.png": "1aoE_EcFbzZx0N2KrRL4l2rYSSY3qY-IS",
    "correlation_heatmap.png": "1FbYq1cNAFCdEb3MYSaoUH1UojgClhqgM"
}

# File Mappings
FILE_MAPPINGS = {
    "data_raw": {
        "raw_data": "climate_raw.csv"
    },
    "data_processed": {
        "clean_data": "climate_clean.csv",
        "featured_data": "feature_engineered_climate_data.csv"
    },
    "models": {
        "temp_model": "ridge_temp.pkl",
        "climate_zone_model": "rf_climate_zone.pkl",
        "vulnerability_model": "gbc_vulnerability.pkl"
    },
    "scripts": {
        "ridge_temp": "ridge_temp.py",
        "rf_climate_zone": "rf_climate_zone.py",
        "processed_climate_data": "processed_climate_data.py",
        "gbr_temp": "gbr_temp.py",
        "gbc_vulnerability": "gbc_vulnerability.py",
        "feature_engineering": "feature_engineering.py",
        "eda": "eda.py"
    },
    "plots": {
        "ridge_temp_pred": "ridge_temp_pred.png",
        "rf_climate_zone_cm": "rf_climate_zone_cm.png",
        "gbr_temp_pred": "gbr_temp_pred.png",
        "gbr_temp_diagnostics": "gbr_temp_diagnostics.png",
        "gbc_vulnerability_cm": "gbc_vulnerability_cm.png",
        "yearly_avg_temp": "yearly_avg_temp.png",
        "yearly_avg_precip": "yearly_avg_precip.png",
        "temp_vs_precip_scatter": "temp_vs_precip_scatter.png",
        "temp_distribution": "temp_distribution.png",
        "season_temp": "season_temp.png",
        "precip_distribution": "precip_distribution.png",
        "max_temp_distribution": "max_temp_distribution.png",
        "district_temp": "district_temp.png",
        "correlation_heatmap": "correlation_heatmap.png"
    }
}

# Local Paths
LOCAL_PATHS = {
    "data_raw": "data/raw/",
    "data_processed": "data/processed/",
    "models": "models/",
    "scripts": "src/",
    "plots": "data/processed/plots/"
}

# Create directories if they don't exist
for path in LOCAL_PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_path(category, file_key):
    """Get the correct file path based on configuration"""
    if category not in FILE_MAPPINGS or file_key not in FILE_MAPPINGS[category]:
        raise ValueError(f"Invalid category/file_key: {category}/{file_key}")

    filename = FILE_MAPPINGS[category][file_key]

    if USE_GOOGLE_DRIVE:
        if filename not in GDRIVE_FILE_IDS:
            raise ValueError(f"No Google Drive ID found for file: {filename}")
        return {
            "filename": filename,
            "file_id": GDRIVE_FILE_IDS[filename]
        }
    else:
        return os.path.join(LOCAL_PATHS[category], filename)
