# train_model_svm.py
import pandas as pd
import joblib
import time
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------- Setup
def main():
    start_time = time.time()
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # ------------------------- Data Loading with Validation
    try:
        df = pd.read_csv("data/processed/feature_engineered_climate_data.csv")
        
        # Validate required columns
        required_columns = [
            'Temp_2m', 'Precip', 'Humidity_2m', 'EarthSkinTemp', 'Heat_Stress',
            'heat_stress_index', 'drought_index', 'wetbulb_diff', 'avg_windspeed'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return

    # ------------------------- Feature Engineering
    features = [
        'Temp_2m', 'Precip', 'Humidity_2m', 'EarthSkinTemp',
        'heat_stress_index', 'drought_index', 'wetbulb_diff',
        'avg_windspeed', 'max_avg_windspeed', 'temp_precip_interaction',
        'Temp_2m_lag1', 'Precip_lag1', 'Temp_2m_roll7', 'Precip_roll7', 'Humidity_2m_roll7'
    ]
    
    # Only use available features
    features = [f for f in features if f in df.columns]
    X = df[features]
    
    # Create binary target (median split)
    y_heat_stress = (df['Heat_Stress'] > df['Heat_Stress'].median()).astype(int)
    
    # Check class balance
    print("\n🔍 Class Distribution:")
    print(y_heat_stress.value_counts(normalize=True))
    
    # ------------------------- Model Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selector', SelectKBest(f_classif, k=10)),  # Select top 10 features
        ('model', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    
    # ------------------------- Hyperparameter Tuning
    param_grid = {
        'model__C': [0.1, 1, 10],
        'model__gamma': ['scale', 'auto', 0.1, 1]
    }
    
    # Stratified K-Fold for imbalanced data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # ------------------------- Training with Cross-Validation
    print("\n🔧 Training SVM with GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_heat_stress, 
        test_size=0.2, 
        stratify=y_heat_stress, 
        random_state=42
    )
    
    grid_search.fit(X_train, y_train)
    
    # ------------------------- Evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n🔥 Optimized SVM – Heat Stress Classification")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ------------------------- Visualization
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - SVM Heat Stress Prediction')
    plt.savefig('plots/svm_confusion_matrix.png')
    plt.close()
    
    # ------------------------- Save Model
    joblib.dump(best_model, "models/svm_heat_stress_pipeline.pkl")
    print(f"\n✅ SVM training completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()