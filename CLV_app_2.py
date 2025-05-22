#!/usr/bin/env python
# coding: utf-8

# In[15]:


# clv_app.py (Transactional CLV with dynamic charts and custom tier segmentation)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

st.set_page_config(layout="wide")
st.title("Customer Lifetime Value (CLV) Calculator - Transactional Only")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with transactions (CustomerID, InvoiceDate, Amount)", type=["csv"])

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

        # Add percentile and custom tier segmentation
        summary['clv_percentile'] = summary['clv_transactional'].rank(pct=True)

        def assign_clv_tier(p):
            if p >= 0.95:
                return 'Platinum Champions'
            elif p >= 0.85:
                return 'Gold Contributors'
            elif p >= 0.70:
                return 'Silver customers'
            elif p >= 0.40:
                return 'Bronze Supporters'
            else:
                return 'Low Value'

        summary['clv_tier'] = summary['clv_percentile'].apply(assign_clv_tier)

        st.success("Transactional CLV computed successfully!")

        # Dynamic Charts
        st.subheader("📊 Visualizations")

        # Histogram of CLV
        fig_hist = px.histogram(summary, x='clv_transactional', nbins=20, title="Distribution du CLV Transactionnel")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Bar chart: average CLV per tier
        avg_tier = summary.groupby('clv_tier')['clv_transactional'].mean().reset_index()
        fig_bar = px.bar(avg_tier, x='clv_tier', y='clv_transactional',
                         title="Average CLV by segment", labels={'clv_transactional': 'CLV'})
        st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter plot: Frequency vs CLV
        fig_scatter = px.scatter(summary, x='frequency', y='clv_transactional',
                                 title="CLV vs purchase frequency", hover_name=summary.index)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Slider to filter CLV range
        clv_min, clv_max = float(summary['clv_transactional'].min()), float(summary['clv_transactional'].max())
        clv_range = st.slider("Filter clients by transactionnal CLV ", clv_min, clv_max, (clv_min, clv_max))
        filtered_summary = summary[(summary['clv_transactional'] >= clv_range[0]) & (summary['clv_transactional'] <= clv_range[1])]

        # Show filtered table
        st.subheader("📋   filtered clients details ")
        display_cols = ['frequency', 'recency', 'T', 'monetary_value', 'clv_transactional', 'clv_tier']
        st.dataframe(filtered_summary[display_cols])
        
        # Download all clvs results
        csv_all = summary.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 download all results",
            data=csv_all,
            file_name='clv_transactional_results.csv',
            mime='text/csv'
        )

    

        # Download filtered results
        csv = filtered_summary.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 download filtered results",
            data=csv,
            file_name='flitered_clv_transactional_results.csv',
            mime='text/csv'
        )


# In[ ]:





# In[ ]:





# In[ ]:




