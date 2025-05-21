# clv_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

st.set_page_config(layout="wide")
st.title("Customer Lifetime Value (CLV) Calculator for MOEVE")

# Upload transaction CSV file
uploaded_file = st.file_uploader("Upload a CSV file with transactions (CustomerID, InvoiceDate, Amount)", type=["csv"])

# Upload optional strategic data
strategic_file = st.file_uploader("Upload a CSV file with strategic variables (CustomerID, renewal_rate, contract_length, engagement_score, adoption_rate)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    required_cols = {'CustomerID', 'InvoiceDate', 'Amount'}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain these columns: {required_cols}")
    else:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Compute RFM summary
        summary = summary_data_from_transaction_data(
            df,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            monetary_value_col='Amount',
            observation_period_end=df['InvoiceDate'].max()
        )

        summary = summary[summary['frequency'] > 0]

        # Fit models
        bgf = BetaGeoFitter(penalizer_coef=0.001)
        bgf.fit(summary['frequency'], summary['recency'], summary['T'])

        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(summary['frequency'], summary['monetary_value'])

        # Predict transactional CLV
        summary['expected_avg_value'] = ggf.conditional_expected_average_profit(
            summary['frequency'], summary['monetary_value']
        )
        summary['clv_transactional'] = ggf.customer_lifetime_value(
            bgf,
            summary['frequency'],
            summary['recency'],
            summary['T'],
            summary['monetary_value'],
            time=6,
            freq='D',
            discount_rate=0.01
        )

        # Add percentile and segment
        summary['clv_percentile'] = summary['clv_transactional'].rank(pct=True)
        summary['clv_quartile'] = pd.qcut(summary['clv_percentile'], q=4, labels=[
            'Q1 - Low', 'Q2 - Mid-Low', 'Q3 - Mid-High', 'Q4 - Top'])

        # If strategic data provided, merge and compute strategic CLV
        if strategic_file is not None:
            strategic_df = pd.read_csv(strategic_file)
            strategic_cols = {'CustomerID', 'renewal_rate', 'contract_length', 'engagement_score', 'adoption_rate'}
            if not strategic_cols.issubset(strategic_df.columns):
                st.error(f"Strategic CSV must contain these columns: {strategic_cols}")
            else:
                summary = summary.reset_index().merge(strategic_df, on='CustomerID', how='left').set_index('CustomerID')
                summary['strategic_score'] = (
                    (summary['renewal_rate'] > 0.8).astype(int) * 2 +
                    (summary['contract_length'] >= 3).astype(int) * 2 +
                    (summary['adoption_rate'] > 0.7).astype(int) * 1
                )
                summary['clv_strategic'] = summary['clv_transactional'] * (1 + summary['strategic_score'] * 0.1)

        st.success("CLV computation completed!")

        # Donut chart with Plotly
        fig = px.pie(summary, names='clv_quartile', hole=0.5, title="CLV Distribution by Quartile")
        st.plotly_chart(fig, use_container_width=True)

        # Slider to filter CLV range
        clv_min, clv_max = float(summary['clv_transactional'].min()), float(summary['clv_transactional'].max())
        clv_range = st.slider("Filter customers by transactional CLV", clv_min, clv_max, (clv_min, clv_max))
        filtered_summary = summary[(summary['clv_transactional'] >= clv_range[0]) & (summary['clv_transactional'] <= clv_range[1])]

        # Show filtered table
        display_cols = ['frequency', 'recency', 'T', 'monetary_value', 'clv_transactional', 'clv_quartile']
        if 'clv_strategic' in summary.columns:
            display_cols.append('clv_strategic')
            display_cols.append('strategic_score')
        st.dataframe(filtered_summary[display_cols])

        # Download button
        csv = filtered_summary.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Filtered Results as CSV",
            data=csv,
            file_name='clv_filtered_results.csv',
            mime='text/csv'
        )
