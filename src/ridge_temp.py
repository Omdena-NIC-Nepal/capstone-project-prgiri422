import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import load_from_folder, save_to_folder  # Updated imports
from config import USE_GOOGLE_DRIVE

def load_and_validate_data():
    """Load and validate data using folder-based system"""
    try:
        # Load processed data using folder-based system
        df = load_from_folder("data_processed", "featured_data")
        
        # SAFE FEATURES (after correlation analysis)
        safe_features = [
            'Precip', 
            'Humidity_2m',
            'avg_windspeed',
            'max_avg_windspeed',
            'temp_precip_interaction',
            'Precip_lag1',
            'Precip_roll7',
            'Humidity_2m_roll7'
        ]
        
        # Remove known leaky features
        leaky_features = ['EarthSkinTemp', 'heat_stress_index', 'drought_index', 'Temp_2m_lag1']
        features = [f for f in safe_features if f in df.columns and f not in leaky_features]
        
        if 'Temp_2m' not in df.columns:
            raise ValueError("Target column 'Temp_2m' missing")
            
        return df[features], df['Temp_2m']
        
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

def save_model(model, model_name="ridge_temp"):
    """Save model using folder-based system"""
    try:
        # Prepare model package with metadata
        model_pkg = {
            'model': model,
            'features': model.feature_names_in_,
            'metrics': {
                'r2': r2_score,
                'mse': mean_squared_error
            }
        }
        
        save_to_folder(model_pkg, "models", model_name)
        print(f"💾 Saved model {model_name}")
    except Exception as e:
        print(f"❌ Failed to save model: {str(e)}")

def train_ridge_model():
    """Main training function with folder-based system"""
    try:
        print("🔍 Loading and validating data...")
        X, y = load_and_validate_data()
        
        # Feature correlation check
        max_corr = X.corrwith(y).abs().max()
        print(f"📊 Max correlation with target: {max_corr:.4f}")
        if max_corr > 0.7:
            print("⚠️ Warning: High correlation with target detected")
        
        # Train-test split
        print("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        print("🏗️ Building pipeline...")
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        # Train model
        print("🏋️ Training model...")
        ridge_pipeline.fit(X_train, y_train)
        
        # Evaluate
        print("📊 Evaluating model...")
        preds = ridge_pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        
        print("\n📈 Model Performance:")
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        
        # Save if performance is reasonable
        if 0.3 < r2 < 0.9:
            save_model(ridge_pipeline)
            print("\n✅ Model saved with realistic performance")
            return ridge_pipeline
        else:
            print("\n❌ Model not saved - suspicious performance detected")
            return None
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

if __name__ == "__main__":
    trained_model = train_ridge_model()
    if trained_model:
        print("✨ Training completed successfully!")