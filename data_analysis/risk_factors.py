import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_risk_factors(df, factor):
    """Plot risk factor analysis"""
    
    if factor == "smoking":
        plot_smoking_risk(df)
    elif factor == "bmi":
        plot_bmi_risk(df)
    elif factor == "age_group":
        plot_age_group_risk(df)

def plot_all_risk_factors(df):
    """Plot all risk factors one after another"""
    
    st.markdown("### Smoking History Risk Analysis")
    plot_smoking_risk(df)
    
    st.markdown("---")
    st.markdown("### BMI Categories Risk Analysis")
    plot_bmi_risk(df)
    
    st.markdown("---")
    st.markdown("### Age Groups Risk Analysis")
    plot_age_group_risk(df)

def plot_smoking_risk(df):
    """Plot smoking history risk analysis"""
    
    # Smoking history distribution
    smoking_counts = df.groupby(['smoking_history', 'diabetes']).size().reset_index(name='count')
    smoking_counts['diabetes_label'] = smoking_counts['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(smoking_counts, x='smoking_history', y='count', 
                        color='diabetes_label',
                        title='Diabetes Cases by Smoking History',
                        barmode='group',
                        color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Diabetes rate by smoking history
        smoking_rate = df.groupby('smoking_history')['diabetes'].mean() * 100
        smoking_rate_df = smoking_rate.reset_index()
        smoking_rate_df.columns = ['smoking_history', 'rate']
        
        fig_rate = px.bar(smoking_rate_df, x='smoking_history', y='rate',
                          title='Diabetes Rate by Smoking History',
                          labels={'rate': 'Diabetes Rate (%)'},
                          color_discrete_sequence=['#ff9999'])
        st.plotly_chart(fig_rate, use_container_width=True)
    
    # Statistics
    risk_data = []
    for smoking in df['smoking_history'].unique():
        subset = df[df['smoking_history'] == smoking]
        risk_data.append({
            'Smoking History': smoking,
            'Total Patients': len(subset),
            'Diabetes Cases': int(subset['diabetes'].sum()),
            'Diabetes Rate': f"{subset['diabetes'].mean() * 100:.2f}%"
        })
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True)

def plot_bmi_risk(df):
    """Plot BMI risk analysis"""
    
    # Create BMI categories
    bins = [0, 18.5, 25, 30, 35, 40, 100]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
    df_bmi = df.copy()
    df_bmi['bmi_category'] = pd.cut(df_bmi['bmi'], bins=bins, labels=labels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution
        bmi_dist = df_bmi.groupby(['bmi_category', 'diabetes']).size().reset_index(name='count')
        bmi_dist['diabetes_label'] = bmi_dist['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
        
        fig = px.bar(bmi_dist, x='bmi_category', y='count', color='diabetes_label',
                     title='Diabetes Cases by BMI Category',
                     barmode='group',
                     color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rate
        bmi_rate = df_bmi.groupby('bmi_category')['diabetes'].mean() * 100
        bmi_rate_df = bmi_rate.reset_index()
        bmi_rate_df.columns = ['bmi_category', 'rate']
        
        fig_rate = px.bar(bmi_rate_df, x='bmi_category', y='rate',
                          title='Diabetes Rate by BMI Category',
                          labels={'rate': 'Diabetes Rate (%)'},
                          color_discrete_sequence=['#ff9999'])
        st.plotly_chart(fig_rate, use_container_width=True)
    
    # Statistics
    risk_data = []
    for category in labels:
        subset = df_bmi[df_bmi['bmi_category'] == category]
        if len(subset) > 0:
            risk_data.append({
                'BMI Category': category,
                'Total Patients': len(subset),
                'Diabetes Cases': int(subset['diabetes'].sum()),
                'Diabetes Rate': f"{subset['diabetes'].mean() * 100:.2f}%"
            })
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True)

def plot_age_group_risk(df):
    """Plot age group risk analysis"""
    
    # Create age groups
    bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
    labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
    df_age = df.copy()
    df_age['age_group'] = pd.cut(df_age['age'], bins=bins, labels=labels, right=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution
        age_dist = df_age.groupby(['age_group', 'diabetes']).size().reset_index(name='count')
        age_dist['diabetes_label'] = age_dist['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
        
        fig = px.bar(age_dist, x='age_group', y='count', color='diabetes_label',
                     title='Diabetes Cases by Age Group',
                     barmode='group',
                     color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rate
        age_rate = df_age.groupby('age_group')['diabetes'].mean() * 100
        age_rate_df = age_rate.reset_index()
        age_rate_df.columns = ['age_group', 'rate']
        
        fig_rate = px.line(age_rate_df, x='age_group', y='rate',
                           title='Diabetes Rate by Age Group',
                           labels={'rate': 'Diabetes Rate (%)'},
                           markers=True)
        st.plotly_chart(fig_rate, use_container_width=True)
    
    # Statistics
    risk_data = []
    for group in labels:
        subset = df_age[df_age['age_group'] == group]
        if len(subset) > 0:
            risk_data.append({
                'Age Group': group,
                'Total Patients': len(subset),
                'Diabetes Cases': int(subset['diabetes'].sum()),
                'Diabetes Rate': f"{subset['diabetes'].mean() * 100:.2f}%"
            })
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True)