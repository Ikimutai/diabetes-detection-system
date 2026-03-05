import streamlit as st
import traceback
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_analysis.data_overview import get_data_overview, get_sample_data, get_feature_description
from data_analysis.diabetes_distribution import plot_diabetes_distribution
from data_analysis.age_analysis import plot_age_analysis
from data_analysis.health_analysis import plot_health_analysis, plot_all_health_indicators
from data_analysis.risk_factors import plot_risk_factors, plot_all_risk_factors
from model_training import DiabetesModelTrainer

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .prediction-high {
        background-color: #FF0000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .prediction-low {
        background-color: #00FF00;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .stButton > button {
        background-color: #2c3e50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #34495e;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: #27ae60;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: #2ecc71;
    }
    .nav-link {
        font-size: 1.1rem !important;
        padding: 0.75rem 1rem !important;
    }
    .nav-link-selected {
        background-color: #2c3e50 !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'importance_df' not in st.session_state:
    st.session_state.importance_df = None

# Load dataset
@st.cache_data
def load_data():
    data_path = os.path.join('data', 'diabetes_prediction_dataset.csv')
    df = pd.read_csv(data_path)
    return df

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'diabetes_model.pkl')
    if os.path.exists(model_path):
        try:
            artifacts = joblib.load(model_path)
            # Check if artifacts is a dictionary with 'model' key
            if isinstance(artifacts, dict) and 'model' in artifacts:
                return artifacts['model']
            # If it's just the model itself
            elif hasattr(artifacts, 'predict'):
                return artifacts
            else:
                st.warning("⚠️ Model file found but in unexpected format. Please retrain.")
                return None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    return None

# Main function
def main():
    # Load data
    df = load_data()
    model = load_model()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=80)
        st.title("Diabetes Prediction System")
        
        # Use radio buttons for navigation
        selected = st.radio(
            "Navigation",
            options=["Home", "Data Analysis", "Model Training", "Prediction"],
            index=0,
            key="navigation",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This system predicts diabetes risk based on patient health indicators using machine learning.")
    
    # Page routing
    if selected == "Home":
        show_home_page(df)
    elif selected == "Data Analysis":
        show_data_analysis_page(df)
    elif selected == "Model Training":
        show_model_training_page()
    elif selected == "Prediction":
        show_prediction_page(model, df)

def show_home_page(df):
    """Display the home page with dataset overview"""
    st.markdown("<h1 class='main-header'>Diabetes Prediction System</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overview = get_data_overview(df)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{overview['total_samples']:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Samples</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{overview['total_features']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Features</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{overview['diabetes_percentage']}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Diabetes Prevalence</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("**Dataset Information**")
        st.write(f"• **Total Samples:** {overview['total_samples']:,}")
        st.write(f"• **Total Features:** {overview['total_features']}")
        st.write(f"• **Diabetes Positive:** {overview['diabetes_positive']:,} ({overview['diabetes_percentage']}%)")
        st.write(f"• **Diabetes Negative:** {overview['diabetes_negative']:,}")
        st.write(f"• **Missing Values:** {overview['missing_values']}")
        st.write(f"• **Duplicate Rows:** {overview['duplicates']}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("**Feature Information**")
        st.write(f"• **Categorical Features:** gender, smoking_history")
        st.write(f"• **Numerical Features:** age, bmi, HbA1c_level, blood_glucose_level")
        st.write(f"• **Binary Features:** hypertension, heart_disease, diabetes")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Sample Data</h2>", unsafe_allow_html=True)
    sample_data = get_sample_data(df)
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
    
    st.markdown("<h2 class='sub-header'>Feature Descriptions</h2>", unsafe_allow_html=True)
    feature_desc = get_feature_description()
    
    desc_df = pd.DataFrame([
        {"Feature": feature, "Description": description}
        for feature, description in feature_desc.items()
    ])
    st.table(desc_df)

def show_data_analysis_page(df):
    """Display the data analysis page with visualizations"""
    st.markdown("<h1 class='main-header'>Data Analysis</h1>", unsafe_allow_html=True)
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Diabetes Distribution", 
        "📈 Age Analysis", 
        "🏥 Health Analysis",
        "⚠️ Risk Factors"
    ])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Diabetes Distribution</h2>", unsafe_allow_html=True)
        plot_diabetes_distribution(df)
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Age Analysis</h2>", unsafe_allow_html=True)
        plot_age_analysis(df)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Health Indicators Analysis</h2>", unsafe_allow_html=True)
        plot_all_health_indicators(df)
    
    with tab4:
        st.markdown("<h2 class='sub-header'>Risk Factors Analysis</h2>", unsafe_allow_html=True)
        plot_all_risk_factors(df)

def show_model_training_page():
    """Display the model training page using DiabetesModelTrainer class"""
    st.markdown("<h1 class='main-header'>Model Training</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Training Pipeline</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Settings")
        
        # Training parameters
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
        use_smote = st.checkbox("Use SMOTE (Handle Imbalance)", True, 
                                help="SMOTE creates synthetic samples of the minority class to balance the dataset")
        tune_hyper = st.checkbox("Tune Hyperparameters", False,
                                 help="Grid search for optimal parameters (takes longer)")
        
        # Train button
        if st.button("🚀 Train Model", type="primary", key="train_model_button"):
            with st.spinner("Training in progress... This may take a moment."):
                try:
                    # Step 1: Initialize trainer
                    trainer = DiabetesModelTrainer(data_path='data/diabetes_prediction_dataset.csv')
                    
                    # Step 2: Load and clean data
                    trainer.load_and_clean_data()
                    
                    # Step 3: Preprocess features
                    trainer.preprocess_features()
                    
                    # Step 4: Prepare train/test split
                    trainer.prepare_train_test(test_size=test_size, use_smote=use_smote)
                    
                    # Step 5: Train multiple models
                    results_df = trainer.train_multiple_models()
                    
                    # Step 6: Tune hyperparameters (optional)
                    if tune_hyper:
                        trainer.tune_hyperparameters()
                    
                    # Step 7: Get feature importance
                    importance_df = trainer.get_feature_importance()
                    
                    # Step 8: Save model artifacts
                    os.makedirs('models', exist_ok=True)
                    trainer.save_model('models/diabetes_model.pkl')
                    
                    # Step 9: Store in session state
                    st.session_state.trainer = trainer
                    st.session_state.results_df = results_df
                    st.session_state.importance_df = importance_df
                    st.session_state.model_trained = True
                    
                    st.success("✅ Training complete! Best model saved.")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    st.error("Please check the console for more details.")
                    traceback.print_exc()
    
    with col2:
        # Display training results if available
        if st.session_state.model_trained and st.session_state.results_df is not None:
            st.subheader("Model Performance Comparison")
            
            # Format and display results
            results = st.session_state.results_df.copy()
            st.dataframe(results.style.format({
                'Accuracy': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1 Score': '{:.3f}',
                'ROC-AUC': '{:.3f}',
                'CV Mean': '{:.3f}',
                'CV Std': '{:.3f}'
            }), use_container_width=True)
            
            # Highlight best model
            best_model = results.iloc[0]['Model']
            st.success(f"🏆 Best Model: **{best_model}** (selected by F1 Score)")
            
            # Display feature importance if available
            if st.session_state.importance_df is not None:
                st.subheader("Feature Importance")
                importance = st.session_state.importance_df.sort_values('importance', ascending=True)
                
                fig = px.bar(importance, x='importance', y='feature', 
                           orientation='h', 
                           title='Feature Importance from Best Model',
                           labels={'importance': 'Importance Score', 'feature': 'Feature'},
                           color='importance',
                           color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
    
    # Data preprocessing steps expander
    with st.expander("📝 View Detailed Preprocessing Steps"):
        st.markdown("""
        ### 🔄 Data Preprocessing Pipeline
        
        #### **Step 1: Data Cleaning**
        - ✓ Removed duplicate records
        - ✓ Capped BMI outliers at 10-60 range (clinically reasonable values)
        - ✓ Checked for missing values (none found in this dataset)
        
        #### **Step 2: Feature Engineering**
        - ✓ Encoded categorical variables:
          - `gender`: Encoded to numeric values (Female→0, Male→1, Other→2)
          - `smoking_history`: 6 categories encoded to 0-5
        - ✓ Created interaction features:
          - `age_bmi = (age × bmi) / 100` - captures combined effect of age and weight
          - `glucose_hba1c = blood_glucose × HbA1c_level` - captures combined glucose effect
        
        #### **Step 3: Train/Test Split**
        - ✓ Stratified split to maintain class distribution
        - ✓ Features scaled using StandardScaler (mean=0, std=1)
        - ✓ Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance
        
        #### **Step 4: Model Training**
        - ✓ Trained 4 different models:
          1. **Logistic Regression** - Baseline linear model
          2. **Decision Tree** - Simple tree-based model
          3. **Random Forest** - Ensemble of decision trees
          4. **Gradient Boosting** - Sequential tree building
        
        #### **Step 5: Model Evaluation**
        - ✓ Tested on holdout data
        - ✓ Evaluated using multiple metrics:
          - **Accuracy**: Overall correctness
          - **Precision**: Accuracy of positive predictions
          - **Recall**: Ability to find all positive cases
          - **F1 Score**: Harmonic mean of precision and recall (best for imbalanced data)
          - **ROC-AUC**: Ability to distinguish between classes
          - **Cross-validation**: 5-fold CV for robustness
        """)

def show_prediction_page(model, df):
    """Display the prediction page with proper feature engineering"""
    st.markdown("<h1 class='main-header'>Diabetes Prediction</h1>", unsafe_allow_html=True)
    
    if model is None:
        st.warning("⚠️ Model not found. Please train the model first in the Model Training page.")
        return
    
    st.markdown("<h2 class='sub-header'>Enter Patient Information</h2>", unsafe_allow_html=True)
    
    # Create form for input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", options=["Female", "Male"], key="gender_input")
            age = st.number_input("Age", min_value=0, max_value=120, value=40, key="age_input")
            hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hypertension_input")
            heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="heart_disease_input")
        
        with col2:
            smoking_history = st.selectbox(
                "Smoking History",
                options=["never", "No Info", "current", "former", "ever", "not current"],
                key="smoking_input"
            )
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, key="bmi_input")
            hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1, key="hba1c_input")
            blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=300, value=100, key="glucose_input")
        
        submitted = st.form_submit_button("🔍 Predict Diabetes Risk", type="primary")
    
    if submitted:
        try:
            # === STEP 1: Encode categorical variables ===
            # Gender encoding (same as training)
            gender_encoded = 1 if gender == 'Male' else 0
            
            # Smoking history encoding (same as training)
            smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
            smoking_encoded = smoking_map[smoking_history]
            
            # === STEP 2: Create interaction features (same as training) ===
            age_bmi = age * bmi / 100
            glucose_hba1c = blood_glucose * hba1c
            
            # === STEP 3: Create risk score (same as training) ===
            risk_score = (
                (1 if bmi > 30 else 0) +           # Obesity indicator
                (1 if age > 45 else 0) +            # Age risk factor
                hypertension +                       # Hypertension presence
                heart_disease +                      # Heart disease presence
                (1 if hba1c > 6.5 else 0) +         # High HbA1c
                (1 if blood_glucose > 140 else 0)    # High blood glucose
            )
            
            # === STEP 4: Prepare features in the EXACT order the model expects ===
            # These feature names must match what the model was trained with
            feature_names = ['gender_encoded', 'age', 'hypertension', 'heart_disease', 
                'smoking_encoded', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                'age_bmi', 'glucose_hba1c', 'risk_score']
            
            # Create feature array with all engineered features
            features = pd.DataFrame([[
                gender_encoded,      # gender
                age,                 # age
                hypertension,        # hypertension
                heart_disease,       # heart_disease
                smoking_encoded,     # smoking_history
                bmi,                 # bmi
                hba1c,               # HbA1c_level
                blood_glucose,       # blood_glucose_level
                age_bmi,             # age_bmi
                glucose_hba1c,       # glucose_hba1c
                risk_score           # risk_score
            ]], columns=feature_names)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            # Display result
            st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                <div class='prediction-high'>
                    <h3 style='color: #f44336;'>⚠️ High Risk of Diabetes</h3>
                    <p style='font-size: 1.2rem;'>Probability: <b>{probability:.2%}</b></p>
                    <p>This patient shows indicators that suggest a high risk of diabetes. 
                    Please consult with a healthcare provider for proper diagnosis and management.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-low'>
                    <h3 style='color: #4caf50;'>✅ Low Risk of Diabetes</h3>
                    <p style='font-size: 1.2rem;'>Probability: <b>{probability:.2%}</b></p>
                    <p>This patient appears to have a low risk of diabetes based on the provided information.
                    Maintain a healthy lifestyle for continued wellness.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display risk meter
            risk_level = probability * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_level,
                title={'text': "Diabetes Risk Score", 'font': {'size': 20}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2c3e50"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#e8f5e9"},
                        {'range': [30, 60], 'color': "#fff3e0"},
                        {'range': [60, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_level
                    }
                }
            ))
            fig.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature summary
            with st.expander("📋 View Engineered Features"):
                feature_summary = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': features.iloc[0].values
                })
                st.dataframe(feature_summary, use_container_width=True)
                
            # Show risk factors explanation
            with st.expander("🔍 Risk Factors Explanation"):
                st.markdown("""
                **How the risk score is calculated:**
                - **BMI > 30**: +1 point (obese)
                - **Age > 45**: +1 point (age risk factor)
                - **Hypertension**: +1 point if present
                - **Heart Disease**: +1 point if present
                - **HbA1c > 6.5**: +1 point (high blood sugar)
                - **Blood Glucose > 140**: +1 point (high glucose)
                
                **Risk Score Range: 0-6 points**
                """)
                
                # Show current risk score breakdown
                risk_factors_table = pd.DataFrame({
                    'Risk Factor': ['BMI > 30', 'Age > 45', 'Hypertension', 'Heart Disease', 'HbA1c > 6.5', 'Glucose > 140'],
                    'Value': [
                        'Yes' if bmi > 30 else 'No',
                        'Yes' if age > 45 else 'No',
                        'Yes' if hypertension == 1 else 'No',
                        'Yes' if heart_disease == 1 else 'No',
                        'Yes' if hba1c > 6.5 else 'No',
                        'Yes' if blood_glucose > 140 else 'No'
                    ],
                    'Points': [
                        1 if bmi > 30 else 0,
                        1 if age > 45 else 0,
                        hypertension,
                        heart_disease,
                        1 if hba1c > 6.5 else 0,
                        1 if blood_glucose > 140 else 0
                    ]
                })
                st.table(risk_factors_table)
                st.info(f"**Total Risk Score: {risk_score} / 6**")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.error("Please check that the model was trained correctly.")
            traceback.print_exc()

if __name__ == "__main__":
    main()