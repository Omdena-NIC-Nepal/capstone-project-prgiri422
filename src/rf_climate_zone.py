import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from utils import load_from_folder, save_to_folder  # Updated imports
from config import USE_GOOGLE_DRIVE

def load_and_validate_data():
    """Load and validate data using folder-based system"""
    try:
        # Load processed data using folder-based system
        df = load_from_folder("data_processed", "featured_data")
        
        # Define required features
        features = [
            'Temp_2m', 'Precip', 'Humidity_2m', 'EarthSkinTemp',
            'heat_stress_index', 'drought_index', 'wetbulb_diff',
            'avg_windspeed', 'max_avg_windspeed', 'temp_precip_interaction',
            'Temp_2m_lag1', 'Precip_lag1', 'Temp_2m_roll7', 'Precip_roll7', 'Humidity_2m_roll7'
        ]
        
        # Validate columns
        required_cols = features + ['Climate_Zone_Encoded']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df, [f for f in features if f in df.columns]
        
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

def save_model_artifacts(model, cm_plot, report, model_name="rf_climate_zone"):
    """Save all artifacts using folder-based system"""
    try:
        # Prepare model package
        model_pkg = {
            'model': model,
            'features': model.feature_names_in_,
            'metrics': {
                'accuracy': accuracy_score,
                'classification_report': classification_report
            }
        }
        
        # Save model
        save_to_folder(model_pkg, "models", model_name)
        
        # Save plot
        plot_pkg = {
            'figure': cm_plot,
            'description': 'Confusion Matrix',
            'model_name': model_name
        }
        save_to_folder(plot_pkg, "model_plots", f"{model_name}_cm")
        
        # Save report
        save_to_folder(report, "model_reports", f"{model_name}_report")
        
        print(f"💾 Saved artifacts for {model_name}")
    except Exception as e:
        print(f"❌ Failed to save artifacts: {str(e)}")

def train_rf_model():
    """Main training function with folder-based system"""
    try:
        print("🔍 Loading and validating data...")
        df, features = load_and_validate_data()
        X = df[features]
        y = df['Climate_Zone_Encoded']
        
        # Train-test split
        print("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Build pipeline
        print("🏗️ Building pipeline...")
        rf_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=10)),
            ('model', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
        
        # Train model
        print("🏋️ Training model...")
        rf_pipeline.fit(X_train, y_train)
        
        # Evaluate
        print("📊 Evaluating model...")
        y_pred = rf_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n📈 Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Generate reports
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        cm = confusion_matrix(y_test, y_pred)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Random Forest - Climate Zone\nAccuracy: {accuracy:.2f}')
        
        # Save artifacts
        save_model_artifacts(rf_pipeline, fig, report)
        
        # Feature importance
        if hasattr(rf_pipeline.named_steps['model'], 'feature_importances_'):
            importances = rf_pipeline.named_steps['model'].feature_importances_
            selected_features = X.columns[rf_pipeline.named_steps['selector'].get_support()]
            feat_imp = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\n🔝 Top 5 Features:")
            print(feat_imp.head(5))
            
            # Save feature importance
            save_to_folder(feat_imp, "model_reports", f"{model_name}_feature_importance")
        
        return rf_pipeline
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

if __name__ == "__main__":
    trained_model = train_rf_model()
    if trained_model:
        print("✨ Training completed successfully!")