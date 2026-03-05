import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_age_analysis(df):
    """Plot age analysis charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age histogram
        fig_hist = px.histogram(df, x='age', nbins=30, 
                               title='Age Distribution',
                               labels={'age': 'Age (years)', 'count': 'Number of Patients'},
                               color_discrete_sequence=['#66b3ff'])
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot by diabetes status
        df_plot = df.copy()
        df_plot['diabetes_label'] = df_plot['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
        fig_box = px.box(df_plot, x='diabetes_label', y='age',
                         title='Age Distribution by Diabetes Status',
                         labels={'diabetes_label': 'Diabetes Status', 'age': 'Age (years)'},
                         color='diabetes_label',
                         color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'})
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Age group analysis
    st.subheader("Diabetes Rate by Age Group")
    
    # Create age groups
    bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
    labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
    df_age = df.copy()
    df_age['age_group'] = pd.cut(df_age['age'], bins=bins, labels=labels, right=False)
    
    # Calculate rates
    age_group_stats = df_age.groupby('age_group')['diabetes'].agg(['mean', 'count']).reset_index()
    age_group_stats.columns = ['age_group', 'diabetes_rate', 'count']
    age_group_stats['diabetes_rate'] = age_group_stats['diabetes_rate'] * 100
    age_group_stats = age_group_stats.sort_values('age_group')
    
    # Line chart
    fig_line = px.line(age_group_stats, x='age_group', y='diabetes_rate',
                       title='Diabetes Rate by Age Group',
                       labels={'age_group': 'Age Group', 'diabetes_rate': 'Diabetes Rate (%)'},
                       markers=True)
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Age", f"{df['age'].mean():.1f} years")
    with col2:
        st.metric("Median Age", f"{df['age'].median():.1f} years")
    with col3:
        mean_age_diabetes = df[df['diabetes'] == 1]['age'].mean()
        st.metric("Mean Age (Diabetes)", f"{mean_age_diabetes:.1f} years")
    with col4:
        mean_age_no_diabetes = df[df['diabetes'] == 0]['age'].mean()
        st.metric("Mean Age (No Diabetes)", f"{mean_age_no_diabetes:.1f} years")