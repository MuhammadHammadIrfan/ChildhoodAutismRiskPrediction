"""
Train the best model and save it for deployment
This script trains the model and saves it with the scaler for use in the Streamlit app
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent

def train_and_save_best_model():
    """Train all models, identify the best one, and save it"""
    
    print("="*80)
    print("TRAINING AND SAVING BEST MODEL FOR DEPLOYMENT")
    print("="*80)
    
    # Load data
    project_root = get_project_root()
    data_path = project_root / "data" / "clean" / "autism_screening_encoded.csv"
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X = df.drop(['class'], axis=1)
    y = df['class']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features for distance-based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": {
            'model': LogisticRegression(max_iter=1000),
            'needs_scaling': True
        },
        "SVM": {
            'model': SVC(probability=True),
            'needs_scaling': True
        },
        "KNN": {
            'model': KNeighborsClassifier(),
            'needs_scaling': True
        },
        "Random Forest": {
            'model': RandomForestClassifier(random_state=42),
            'needs_scaling': False
        },
        "Decision Tree": {
            'model': DecisionTreeClassifier(random_state=42),
            'needs_scaling': False
        },
        "Naive Bayes": {
            'model': GaussianNB(),
            'needs_scaling': False
        }
    }
    
    # Train and evaluate all models
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    for name, config in models.items():
        model = config['model']
        needs_scaling = config['needs_scaling']
        
        # Select appropriate data
        X_train_use = X_train_scaled if needs_scaling else X_train
        X_test_use = X_test_scaled if needs_scaling else X_test
        
        # Train
        model.fit(X_train_use, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_use)
        y_prob = model.predict_proba(X_test_use)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='accuracy')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'Needs_Scaling': needs_scaling
        })
        
        trained_models[name] = {
            'model': model,
            'needs_scaling': needs_scaling,
            'accuracy': acc
        }
        
        print(f"\n{name}")
        print(f"   Test Accuracy: {acc:.4f}")
        print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Find best model
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    
    best_model_name = results_df.iloc[0]['Model']
    best_model_data = trained_models[best_model_name]
    
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model_name}")
    print("="*80)
    print(f"Accuracy: {best_model_data['accuracy']:.4f}")
    print(f"Needs Scaling: {best_model_data['needs_scaling']}")
    
    # Save the best model
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "best_model.pkl"
    
    # Prepare model data for saving
    model_to_save = {
        'model': best_model_data['model'],
        'scaler': scaler if best_model_data['needs_scaling'] else None,
        'model_name': best_model_name,
        'accuracy': best_model_data['accuracy'],
        'needs_scaling': best_model_data['needs_scaling'],
        'feature_names': X.columns.tolist()
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_to_save, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Test loading the model
    print("\n" + "="*80)
    print("TESTING MODEL LOADING")
    print("="*80)
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    print(f"Model loaded successfully")
    print(f"   Model name: {loaded_model['model_name']}")
    print(f"   Accuracy: {loaded_model['accuracy']:.4f}")
    print(f"   Number of features: {len(loaded_model['feature_names'])}")
    
    # Make a test prediction
    if loaded_model['needs_scaling']:
        test_input = scaler.transform(X_test.iloc[[0]])
    else:
        test_input = X_test.iloc[[0]]
    
    test_pred = loaded_model['model'].predict(test_input)
    test_prob = loaded_model['model'].predict_proba(test_input)
    
    print(f"\nTest prediction successful")
    print(f"   Prediction: {test_pred[0]}")
    print(f"   Probability: {test_prob[0]}")
    
    print("\n" + "="*80)
    print("MODEL TRAINING AND SAVING COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Run the Streamlit app: streamlit run app/app.py")
    print(f"2. Test the predictions in the web interface")
    print(f"3. Deploy to Streamlit Cloud")

if __name__ == "__main__":
    train_and_save_best_model()
