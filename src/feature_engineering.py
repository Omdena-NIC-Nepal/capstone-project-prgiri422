import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
from pathlib import Path
from utils import load_file
from config import get_file_path

def feature_engineering(output_filename='feature_engineered_climate_data.csv'):
    """
    Performs feature engineering using config-driven file loading and saving.
    Automatically downloads from Google Drive if required.
    """
    # --- LOAD DATA ---
    try:
        df = load_file("data_processed", "clean_data")
        print("✅ Successfully loaded climate_clean.csv")
    except Exception as e:
        print(f"❌ Failed to load input data: {e}")
        return None

    # --- 1. Climate Indices ---
    df['heat_stress_index'] = df['Temp_2m'] + 0.1 * df['Humidity_2m']

    precip_scaled = (df['Precip'] - df['Precip'].min()) / (df['Precip'].max() - df['Precip'].min())
    temp_scaled = (df['Temp_2m'] - df['Temp_2m'].min()) / (df['Temp_2m'].max() - df['Temp_2m'].min())
    df['drought_index'] = (1 - precip_scaled) + temp_scaled

    # --- 2. Seasonal Indicators ---
    df['is_monsoon'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
    df['is_winter'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)

    # --- 3. Lag Features ---
    df['Temp_2m_lag1'] = df['Temp_2m'].shift(1)
    df['Precip_lag1'] = df['Precip'].shift(1)

    # --- 4. Derived Features ---
    if 'WetBulbTemp_2m' in df.columns:
        df['wetbulb_diff'] = df['Temp_2m'] - df['WetBulbTemp_2m']
    if all(col in df.columns for col in ['WindSpeed_10m', 'WindSpeed_50m']):
        df['avg_windspeed'] = (df['WindSpeed_10m'] + df['WindSpeed_50m']) / 2
    if all(col in df.columns for col in ['MaxWindSpeed_10m', 'MaxWindSpeed_50m']):
        df['max_avg_windspeed'] = (df['MaxWindSpeed_10m'] + df['MaxWindSpeed_50m']) / 2
    df['temp_precip_interaction'] = df['Temp_2m'] * df['Precip']

    # --- 5. Rolling Features (7-day) ---
    df['Temp_2m_roll7'] = df['Temp_2m'].rolling(window=7).mean()
    df['Precip_roll7'] = df['Precip'].rolling(window=7).mean()
    df['Humidity_2m_roll7'] = df['Humidity_2m'].rolling(window=7).mean()

    # --- 6. Encode Categorical Features ---
    for col in ['Climate_Zone', 'Vulnerability']:
        if col in df.columns:
            df[f'{col}_Encoded'] = LabelEncoder().fit_transform(df[col])

    # --- 7. Normalize Numeric Features ---
    scaler = MinMaxScaler()
    numeric_cols = ['Temp_2m', 'Precip', 'Humidity_2m', 'WindSpeed_10m',
                    'WindSpeed_50m', 'EarthSkinTemp', 'heat_stress_index', 'drought_index']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if numeric_cols:
        scaled = scaler.fit_transform(df[numeric_cols])
        df_scaled = pd.DataFrame(scaled, columns=[f"{col}_scaled" for col in numeric_cols])
        df = pd.concat([df, df_scaled], axis=1)

    # --- 8. Final Cleanup ---
    df = df.dropna().reset_index(drop=True)

    # --- 9. Save Engineered Data ---
    try:
        save_path = get_file_path("data_processed", "featured_data")
        df.to_csv(save_path, index=False)
        print(f"✅ Feature-engineered data saved to {save_path}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return None

    print("✅ Feature Engineering Complete! Final shape:", df.shape)
    return df

if __name__ == "__main__":
    feature_engineering()
