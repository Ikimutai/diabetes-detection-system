import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_diabetes_distribution(df):
    """Plot diabetes distribution charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        diabetes_counts = df['diabetes'].value_counts().reset_index()
        diabetes_counts.columns = ['diabetes', 'count']
        diabetes_counts['label'] = diabetes_counts['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
        
        fig_pie = px.pie(diabetes_counts, 
                         values='count', 
                         names='label',
                         title='Diabetes Distribution',
                         color_discrete_sequence=['#66b3ff', '#ff9999'],
                         hole=0.3)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart by gender
        gender_dist = df.groupby(['gender', 'diabetes']).size().reset_index(name='count')
        gender_dist['diabetes_label'] = gender_dist['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
        
        fig_bar = px.bar(gender_dist, 
                         x='gender', 
                         y='count', 
                         color='diabetes_label',
                         title='Diabetes Distribution by Gender',
                         barmode='group',
                         color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Diabetes Patients", int(df['diabetes'].sum()))
    with col3:
        female_rate = round(df[df['gender'] == 'Female']['diabetes'].mean() * 100, 2)
        st.metric("Female Diabetes Rate", f"{female_rate}%")
    with col4:
        male_rate = round(df[df['gender'] == 'Male']['diabetes'].mean() * 100, 2)
        st.metric("Male Diabetes Rate", f"{male_rate}%")