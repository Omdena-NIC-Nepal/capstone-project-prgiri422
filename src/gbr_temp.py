import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from utils import load_from_folder, save_to_folder  # Updated imports
from config import USE_GOOGLE_DRIVE

# SAFE FEATURES (after removing leaky ones)
SAFE_FEATURES = [
    'Precip',
    'Humidity_2m',
    'avg_windspeed',
    'max_avg_windspeed',
    'temp_precip_interaction',
    'Precip_lag1',
    'Precip_roll7',
    'Humidity_2m_roll7'
]

def load_and_validate_data():
    """Load data while ensuring no target leakage using folder-based system"""
    try:
        # Load processed data using folder-based system
        df = load_from_folder("data_processed", "featured_data")  # Updated to use folder-based loading
        
        # Verify target column exists
        if 'Temp_2m' not in df.columns:
            raise ValueError("Target column 'Temp_2m' missing")
            
        # Check for leaky features
        leaky_features = ['Temp_2m', 'EarthSkinTemp', 'heat_stress_index', 
                         'drought_index', 'Temp_2m_lag1', 'Temp_2m_roll7']
        for feature in leaky_features:
            if feature in SAFE_FEATURES:
                raise ValueError(f"Leaky feature '{feature}' in SAFE_FEATURES")
                
        return df
        
    except Exception as e:
        print(f"🚨 Data Error: {str(e)}")
        if USE_GOOGLE_DRIVE:
            print("ℹ️ Google Drive mode is active. Check:")
            print("- File sharing permissions")
            print("- Correct folder IDs in config.py")
        else:
            print("ℹ️ Local mode is active. Check:")
            print("- File exists in data/processed/feature_engineered_climate_data.csv")
        exit()

def save_model_artifacts(model, plot, model_name="gbr_temp"):
    """Save model and plots using folder-based system"""
    try:
        # Save model
        model_data = {
            'model': model,
            'features': SAFE_FEATURES,
            'metrics': {'r2': r2_score, 'mse': mean_squared_error}
        }
        save_to_folder(model_data, "models", model_name)  # Uses folder-based saving
        
        # Save plot
        plot_data = {
            'figure': plot,
            'description': 'GBR Temperature Prediction Diagnostics'
        }
        save_to_folder(plot_data, "model_plots", f"{model_name}_diagnostics")
        
        print(f"💾 Saved model artifacts for {model_name}")
    except Exception as e:
        print(f"❌ Failed to save artifacts: {str(e)}")

def train_gbr_model():
    """Main training function with folder-based system"""
    try:
        print("🔍 Loading and validating data...")
        df = load_and_validate_data()
        
        # Prepare features and target
        features = [f for f in SAFE_FEATURES if f in df.columns]
        X = df[features]
        y = df['Temp_2m']
        
        # Feature correlation check
        corr_threshold = 0.7
        high_corr = X.corrwith(y).abs() > corr_threshold
        if high_corr.any():
            print(f"⚠️ Highly correlated features (> {corr_threshold}):")
            print(X.corrwith(y).abs()[high_corr])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Build pipeline
        gbr_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('variance_filter', VarianceThreshold(threshold=0.01)),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                validation_fraction=0.2,
                n_iter_no_change=10
            ))
        ])
        
        print("🏋️ Training GBR model...")
        gbr_pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = gbr_pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print("\n📊 Validation Metrics:")
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        
        # Baseline comparison
        from sklearn.dummy import DummyRegressor
        dummy = DummyRegressor()
        dummy.fit(X_train, y_train)
        print(f"\n🧪 Baseline R²: {dummy.score(X_test, y_test):.4f}")
        
        # Create diagnostic plot
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
        plt.title(f"GBR Temperature Prediction\nR²={r2:.2f} (Δ={y_test.std():.1f}°C)")
        plt.xlabel("Actual Temperature (°C)")
        plt.ylabel("Predicted Temperature (°C)")
        plt.legend()
        plt.grid()
        
        # Save artifacts if performance is reasonable
        if 0.3 < r2 < 0.9:
            save_model_artifacts(gbr_pipeline, fig)
            print("\n✅ Model saved with realistic performance")
            return gbr_pipeline
        else:
            print("\n❌ Model not saved - suspicious performance detected")
            plt.close(fig)
            return None
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

if __name__ == "__main__":
    trained_model = train_gbr_model()
    if trained_model:
        print("✨ Training completed successfully!")