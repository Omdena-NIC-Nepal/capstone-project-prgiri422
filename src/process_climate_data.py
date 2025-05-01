import pandas as pd
from utils import load_from_folder, save_to_folder  # Updated imports
from config import USE_GOOGLE_DRIVE

def process_climate_data():
    """
    Processes climate data from raw to clean format
    Uses the new folder-based file management system
    """
    try:
        # --- LOAD DATA ---
        print("⏳ Loading raw data...")
        df = load_from_folder("data_raw", "climate_data")  # Uses folder-based loading
        
        # --- DATA VALIDATION ---
        required_columns = ['Date', 'Temp_2m', 'RH_2m', 'Precip', 'District', 'MaxTemp_2m']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # --- DATE HANDLING ---
        print("🕒 Processing dates...")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        # --- SEASON CLASSIFICATION ---
        print("🌦️ Classifying seasons...")
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Pre-Monsoon'
            elif month in [6, 7, 8, 9]:
                return 'Monsoon'
            return 'Post-Monsoon'
        df['Season'] = df['Month'].apply(get_season)

        # --- HEAT STRESS CALCULATION ---
        print("🔥 Calculating heat stress...")
        T = df['Temp_2m']
        RH = df['RH_2m']
        df['Heat_Stress'] = 0.5 * (T + 61 + ((T - 68) * 1.2) + (RH * 0.094)

        # --- DROUGHT INDEX ---
        print("🏜️ Calculating drought index...")
        df['Drought_Index'] = df.groupby('District')['Precip'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )

        # --- CLIMATE ZONE CLASSIFICATION ---
        print("🌍 Classifying climate zones...")
        def classify_zone(MaxTemp_2m):
            if MaxTemp_2m > 35:
                return 'High Heat Zone'
            elif 10 <= MaxTemp_2m <= 35:
                return 'Medium Heat Zone'
            return 'Cold Zone'
        df['Climate_Zone'] = df['MaxTemp_2m'].apply(classify_zone)

        # --- VULNERABILITY ANALYSIS ---
        print("⚠️ Analyzing vulnerability...")
        district_avg_temp = df.groupby('District')['Temp_2m'].mean().reset_index()
        district_avg_temp.columns = ['District', 'Avg_Temp_2m']
        
        def classify_vulnerability(avg_temp):
            if avg_temp > 40:
                return 'High Vulnerable'
            elif 35 <= avg_temp <= 40 or avg_temp < 10:
                return 'Vulnerable'
            elif 25 <= avg_temp < 35 or 10 <= avg_temp < 20:
                return 'Less Vulnerable'
            elif 20 <= avg_temp < 25:
                return 'Comfort'
            return 'Unknown'
            
        district_avg_temp['Vulnerability'] = district_avg_temp['Avg_Temp_2m'].apply(classify_vulnerability)
        df = df.merge(district_avg_temp[['District', 'Vulnerability']], on='District', how='left')

        # --- SAVE PROCESSED DATA ---
        print("💾 Saving processed data...")
        save_to_folder(df, "data_processed", "clean_data")  # Uses folder-based saving
        
        print("✅ Processing complete!")
        print(f"📊 Final dataset shape: {df.shape}")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        return df
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        if USE_GOOGLE_DRIVE:
            print("ℹ️ Google Drive mode is active. Check:")
            print("- File sharing permissions")
            print("- Correct folder IDs in config.py")
        else:
            print("ℹ️ Local mode is active. Check:")
            print("- File exists in data/raw/climate_raw.csv")
            print("- Correct folder structure")
        raise

if __name__ == "__main__":
    process_climate_data()