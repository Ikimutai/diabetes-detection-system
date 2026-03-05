import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_health_analysis(df, indicator):
    """Plot health analysis for a specific indicator"""
    
    # Define indicator mapping
    indicators = {
        'bmi': {'name': 'BMI', 'range': [10, 60]},
        'HbA1c_level': {'name': 'HbA1c Level', 'range': [3, 15]},
        'blood_glucose_level': {'name': 'Blood Glucose Level', 'range': [50, 300]},
        'hypertension': {'name': 'Hypertension', 'categorical': True},
        'heart_disease': {'name': 'Heart Disease', 'categorical': True}
    }
    
    df_plot = df.copy()
    df_plot['diabetes_label'] = df_plot['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
    
    if indicators[indicator].get('categorical', False):
        # For categorical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            # Grouped bar chart
            grouped = df_plot.groupby([indicator, 'diabetes_label']).size().reset_index(name='count')
            grouped[indicator] = grouped[indicator].map({0: 'No', 1: 'Yes'})
            
            fig = px.bar(grouped, x=indicator, y='count', color='diabetes_label',
                         title=f'{indicators[indicator]["name"]} Distribution by Diabetes Status',
                         barmode='group',
                         color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rate chart
            rates = df_plot.groupby(indicator)['diabetes'].mean() * 100
            rates_df = rates.reset_index()
            rates_df.columns = [indicator, 'rate']
            rates_df[indicator] = rates_df[indicator].map({0: 'No', 1: 'Yes'})
            
            fig_rate = px.bar(rates_df, x=indicator, y='rate',
                              title=f'Diabetes Rate by {indicators[indicator]["name"]}',
                              labels={'rate': 'Diabetes Rate (%)'},
                              color_discrete_sequence=['#ff9999'])
            st.plotly_chart(fig_rate, use_container_width=True)
    else:
        # For numerical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(df_plot, x=indicator, color='diabetes_label',
                                    title=f'{indicators[indicator]["name"]} Distribution',
                                    labels={indicator: indicators[indicator]["name"], 'count': 'Number of Patients'},
                                    color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'},
                                    nbins=30)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(df_plot, x='diabetes_label', y=indicator,
                             title=f'{indicators[indicator]["name"]} by Diabetes Status',
                             labels={'diabetes_label': 'Diabetes Status', indicator: indicators[indicator]["name"]},
                             color='diabetes_label',
                             color_discrete_map={'No Diabetes': '#66b3ff', 'Diabetes': '#ff9999'})
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistics
    if not indicators[indicator].get('categorical', False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(f"Mean {indicators[indicator]['name']} (Diabetes)", 
                     f"{df[df['diabetes'] == 1][indicator].mean():.2f}")
        with col2:
            st.metric(f"Mean {indicators[indicator]['name']} (No Diabetes)", 
                     f"{df[df['diabetes'] == 0][indicator].mean():.2f}")
        with col3:
            st.metric(f"Median {indicators[indicator]['name']} (Diabetes)", 
                     f"{df[df['diabetes'] == 1][indicator].median():.2f}")
        with col4:
            st.metric(f"Median {indicators[indicator]['name']} (No Diabetes)", 
                     f"{df[df['diabetes'] == 0][indicator].median():.2f}")

def plot_all_health_indicators(df):
    """Plot all health indicators one after another"""
    
    st.markdown("### BMI Analysis")
    plot_health_analysis(df, 'bmi')
    
    st.markdown("---")
    st.markdown("### HbA1c Level Analysis")
    plot_health_analysis(df, 'HbA1c_level')
    
    st.markdown("---")
    st.markdown("### Blood Glucose Level Analysis")
    plot_health_analysis(df, 'blood_glucose_level')
    
    st.markdown("---")
    st.markdown("### Hypertension Analysis")
    plot_health_analysis(df, 'hypertension')
    
    st.markdown("---")
    st.markdown("### Heart Disease Analysis")
    plot_health_analysis(df, 'heart_disease')