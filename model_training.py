"""
Diabetes Prediction - Model Training Module
Single file containing all model training functionality
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DiabetesModelTrainer:
    """
    A complete class for training diabetes prediction models.
    Handles data loading, cleaning, feature engineering, model training,
    evaluation, and saving.
    """
    
    def __init__(self, data_path='data/diabetes_prediction_dataset.csv'):
        """
        Initialize the trainer with data path.
        
        Parameters:
        -----------
        data_path : str
            Path to the diabetes dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 
                             'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                             'age_bmi', 'glucose_hba1c']
        
    def load_and_clean_data(self):
        """
        Step 1: Load dataset and perform initial cleaning.
        
        Cleaning steps:
        - Remove duplicates
        - Cap BMI outliers at clinically reasonable range (10-60)
        """
        print("📥 Loading and cleaning data...")
        self.df = pd.read_csv(self.data_path)
        
        # Check for missing values
        missing = self.df.isnull().sum().sum()
        print(f"  - Missing values: {missing}")
        
        # Remove duplicates
        initial_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        duplicates = initial_shape[0] - self.df.shape[0]
        print(f"  - Removed {duplicates} duplicates")
        
        # Cap BMI outliers (clinically reasonable range)
        self.df['bmi'] = self.df['bmi'].clip(lower=10, upper=60)
        
        print(f"  - Final dataset shape: {self.df.shape}")
        return self.df
    
    def preprocess_features(self):
        """
        Step 2: Preprocess features for model training.
        
        Preprocessing steps:
        - Encode categorical variables (gender, smoking_history)
        - Create interaction features (age_bmi, glucose_hba1c)
        """
        print("🔄 Preprocessing features...")
        
        # Encode gender
        self.label_encoders['gender'] = LabelEncoder()
        self.df['gender'] = self.label_encoders['gender'].fit_transform(self.df['gender'])
        
        # Encode smoking history
        self.label_encoders['smoking_history'] = LabelEncoder()
        self.df['smoking_history'] = self.label_encoders['smoking_history'].fit_transform(
            self.df['smoking_history']
        )
        
        # Create interaction features
        self.df['age_bmi'] = self.df['age'] * self.df['bmi'] / 100
        self.df['glucose_hba1c'] = self.df['blood_glucose_level'] * self.df['HbA1c_level']
        print("  - Created interaction features: age_bmi, glucose_hba1c")
        
        return self.df
    
    def prepare_train_test(self, test_size=0.2, use_smote=True, random_state=42):
        """
        Step 3: Prepare train/test split with optional SMOTE balancing.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing (default: 0.2)
        use_smote : bool
            Whether to apply SMOTE for class imbalance (default: True)
        random_state : int
            Random seed for reproducibility
        """
        print("✂️ Preparing train/test split...")
        
        # Select features
        X = self.df[self.feature_names]
        y = self.df['diabetes']
        
        # Split data (stratified to maintain class distribution)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"  - Train set size: {len(self.X_train)}")
        print(f"  - Test set size: {len(self.X_test)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("  - Features scaled using StandardScaler")
        
        # Apply SMOTE for class imbalance
        if use_smote:
            smote = SMOTE(random_state=random_state)
            self.X_train_scaled, self.y_train = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
            print(f"  - SMOTE applied - New training set size: {len(self.X_train_scaled)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_multiple_models(self):
        """
        Step 4: Train multiple models and compare performance.
        
        Models trained:
        - Logistic Regression (baseline)
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        
        Returns:
        --------
        pd.DataFrame : Comparison of model performances
        """
        print("🤖 Training multiple models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = []
        
        for name, model in models.items():
            print(f"  - Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='f1')
            
            # Store results
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC': roc_auc,
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std()
            })
            
            # Store model
            self.models[name] = model
        
        # Create results dataframe and sort by F1 score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
        
        # Select best model (by F1 score for imbalanced data)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        print(f"  ✅ Best model: {results_df.iloc[0]['Model']} (F1 Score: {results_df.iloc[0]['F1 Score']:.3f})")
        
        return results_df
    
    def tune_hyperparameters(self, param_grid=None):
        """
        Step 5: Tune hyperparameters for Random Forest.
        
        Parameters:
        -----------
        param_grid : dict, optional
            Custom parameter grid for GridSearchCV
        
        Returns:
        --------
        dict : Best parameters found
        """
        print("🔧 Tuning hyperparameters...")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"  - Best parameters: {grid_search.best_params_}")
        print(f"  - Best F1 score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_
    
    def get_feature_importance(self):
        """
        Extract feature importance from the best model (if available).
        
        Returns:
        --------
        pd.DataFrame : Feature importance values
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).reset_index(drop=True)
        return None
    
    def save_model(self, filename='models/diabetes_model.pkl'):
        """
        Step 6: Save the trained model and preprocessing objects.
        
        Parameters:
        -----------
        filename : str
            Path to save the model artifacts
        """
        print(f"💾 Saving model to {filename}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Package all artifacts in a dictionary
        model_artifacts = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_artifacts, filename)
        print("  ✅ Model saved successfully!")
        
    def load_model(self, filename='models/diabetes_model.pkl'):
        """
        Load a previously saved model.
        
        Parameters:
        -----------
        filename : str
            Path to the saved model artifacts
        
        Returns:
        --------
        dict : Model artifacts
        """
        print(f"📂 Loading model from {filename}...")
        if os.path.exists(filename):
            artifacts = joblib.load(filename)
            self.best_model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.label_encoders = artifacts['label_encoders']
            self.feature_names = artifacts['feature_names']
            print("  ✅ Model loaded successfully!")
            return artifacts
        else:
            print("  ❌ Model file not found!")
            return None