import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import load_from_folder, save_to_folder  # Updated imports
from config import USE_GOOGLE_DRIVE

def perform_eda():
    """
    Performs Exploratory Data Analysis using folder-based file management
    """
    try:
        # --- LOAD DATA ---
        print("⏳ Loading data...")
        df = load_from_folder("data_processed", "clean_data")  # Uses folder-based loading
        
        # --- DATA PREP ---
        print("🧹 Preparing data...")
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Optional: rename key columns
        rename_map = {
            'maxtemp_2m': 'maxtemp',
            'temp_2m': 'temp',
            'precip': 'precip',
            'wind': 'wind',
            'heat_stress': 'heat_stress',
            'drought_index': 'drought_index',
            'temprange_2m': 'temp_range',
        }
        df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)
        
        print("✅ Available columns:", df.columns.tolist())

        # --- PLOTTING FUNCTIONS ---
        def save_plot(fig, plot_name, description=""):
            """Save plot using folder-based system"""
            try:
                plot_data = {
                    'figure': fig,
                    'description': description,
                    'plot_name': plot_name
                }
                save_to_folder(plot_data, "eda_plots", plot_name)
                print(f"💾 Saved plot: {plot_name}")
            except Exception as e:
                print(f"❌ Failed to save plot {plot_name}: {str(e)}")
            finally:
                plt.close(fig)

        # --- 1. DISTRIBUTION PLOTS ---
        print("📊 Creating distribution plots...")
        def plot_distribution(col, color, title, xlabel):
            if col in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[col], kde=True, color=color, ax=ax)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Frequency')
                save_plot(fig, f'{col}_distribution.png', title)

        plot_distribution('temp', 'skyblue', 'Temperature Distribution', 'Temperature (°C)')
        plot_distribution('maxtemp', 'orange', 'Max Temperature Distribution', 'Max Temperature (°C)')
        plot_distribution('precip', 'green', 'Precipitation Distribution', 'Precipitation (mm)')
        plot_distribution('wind', 'red', 'Wind Speed Distribution', 'Wind Speed (m/s)')

        # --- 2. DISTRICT/SEASON TRENDS ---
        print("🌡️ Analyzing district/season trends...")
        if 'district' in df.columns and 'maxtemp' in df.columns:
            fig, ax = plt.subplots(figsize=(12,6))
            sns.boxplot(x='district', y='maxtemp', data=df, palette='Set2', ax=ax)
            ax.set_title('District-wise Max Temperature Variation')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            save_plot(fig, 'district_temp.png', 'District Temperature Variation')

        if 'season' in df.columns and 'maxtemp' in df.columns:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(x='season', y='maxtemp', data=df, palette='Set1', ax=ax)
            ax.set_title('Season-wise Max Temperature Variation')
            save_plot(fig, 'season_temp.png', 'Seasonal Temperature Variation')

        # --- 3. OUTLIER DETECTION ---
        print("🔍 Detecting outliers...")
        def detect_outliers(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            return data[(data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)]

        outlier_results = {}
        if 'maxtemp' in df.columns:
            outlier_results['maxtemp'] = len(detect_outliers(df, 'maxtemp'))
        if 'precip' in df.columns:
            outlier_results['precip'] = len(detect_outliers(df, 'precip'))
        
        # Save outlier results
        if outlier_results:
            save_to_folder(outlier_results, "eda_results", "outlier_counts")
            print("📌 Outlier counts:", outlier_results)

        # --- 4. CORRELATION ANALYSIS ---
        print("📈 Analyzing correlations...")
        corr_cols = ['temp', 'maxtemp', 'precip', 'wind', 'heat_stress', 'drought_index', 'temp_range']
        corr_cols = [col for col in corr_cols if col in df.columns]
        
        if len(corr_cols) > 1:
            corr_df = df[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(12,10))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            ax.set_title('Feature Correlation Heatmap')
            save_plot(fig, 'correlation_heatmap.png', 'Feature Correlations')
            
            # Save correlation matrix
            save_to_folder(corr_df, "eda_results", "correlation_matrix")

        # --- 5. TIME SERIES ANALYSIS ---
        print("⏳ Analyzing time series...")
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            # Yearly temperature trends
            if 'maxtemp' in df.columns:
                yearly_temp = df.groupby('year')['maxtemp'].mean().reset_index()
                fig, ax = plt.subplots(figsize=(10,6))
                sns.lineplot(x='year', y='maxtemp', data=yearly_temp, marker='o', ax=ax)
                ax.set_title('Yearly Average Max Temperature')
                save_plot(fig, 'yearly_avg_temp.png', 'Temperature Trends')
                
                # Save temperature data
                save_to_folder(yearly_temp, "eda_results", "yearly_temperature")

            # Yearly precipitation trends
            if 'precip' in df.columns:
                yearly_precip = df.groupby('year')['precip'].mean().reset_index()
                fig, ax = plt.subplots(figsize=(10,6))
                sns.lineplot(x='year', y='precip', data=yearly_precip, marker='o', ax=ax)
                ax.set_title('Yearly Average Precipitation')
                save_plot(fig, 'yearly_avg_precip.png', 'Precipitation Trends')
                
                # Save precipitation data
                save_to_folder(yearly_precip, "eda_results", "yearly_precipitation")

                # Temperature vs Precipitation
                if 'maxtemp' in df.columns:
                    fig, ax = plt.subplots(figsize=(10,6))
                    sns.scatterplot(x=yearly_temp['maxtemp'], y=yearly_precip['precip'], ax=ax)
                    ax.set_xlabel('Avg Max Temperature (°C)')
                    ax.set_ylabel('Avg Precipitation (mm)')
                    ax.set_title('Temperature vs Precipitation Correlation')
                    save_plot(fig, 'temp_vs_precip_scatter.png', 'Temp-Precip Relationship')
                    
                    # Save relationship data
                    temp_precip = yearly_temp.merge(yearly_precip, on='year')
                    save_to_folder(temp_precip, "eda_results", "temp_precip_relationship")

        print("✅ EDA completed successfully!")
        return True

    except Exception as e:
        print(f"❌ EDA failed: {str(e)}")
        if USE_GOOGLE_DRIVE:
            print("ℹ️ Google Drive mode is active. Check:")
            print("- File sharing permissions")
            print("- Correct folder IDs in config.py")
        else:
            print("ℹ️ Local mode is active. Check:")
            print("- File exists in data/processed/climate_clean.csv")
        return False

if __name__ == "__main__":
    success = perform_eda()
    if not success:
        exit(1)