import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# -------------------------------
# File paths (LOCAL for Streamlit Cloud)
# -------------------------------
MODEL_FILE = "youtube_model.pkl"
DATA_FILE = "YouTube_Monetization_Modeler.csv"

# -------------------------------
# Load Model Safely
# -------------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error("❌ Model file not found. Please upload 'youtube_model.pkl' to your repo.")
    st.stop()

# -------------------------------
# Load Dataset Safely
# -------------------------------
if os.path.exists(DATA_FILE):
    original_df = pd.read_csv(DATA_FILE)
else:
    original_df = pd.DataFrame()
    st.warning("⚠️ Dataset not found. Some visualizations will be disabled.")

# -------------------------------
# App Title
# -------------------------------
st.title("📹 YouTube Ad Revenue Predictor")
st.write("Estimate your potential YouTube ad revenue using ML 📊")

# -------------------------------
# User Inputs
# -------------------------------
views = st.number_input("Views", min_value=0, value=10000)
likes = st.number_input("Likes", min_value=0, value=1100)
comments = st.number_input("Comments", min_value=0, value=274)
watch_time = st.slider("Watch Time (Minutes)", 10000.0, 70000.0, 37500.0, step=100.0)
vid_length = st.number_input("Video Length (Minutes)", min_value=0.0, value=16.0)
subs = st.number_input("Subscribers", min_value=0, value=500000)

cat = st.selectbox("Category", ['Gaming', 'Education', 'Entertainment', 'Tech', 'Music', 'Lifestyle'])
dev = st.selectbox("Device", ['Mobile', 'Desktop', 'Tablet'])
country = st.selectbox("Country", ['USA', 'India', 'UK', 'Brazil', 'CA', 'DE', 'AU'])

# -------------------------------
# Feature Engineering
# -------------------------------
engagement_rate = (likes + comments) / views if views > 0 else 0

# -------------------------------
# Prediction
# -------------------------------
input_data = pd.DataFrame(
    [[cat, dev, country, views, likes, comments, watch_time, vid_length, subs, engagement_rate]],
    columns=[
        'category', 'device', 'country', 'views', 'likes', 'comments',
        'watch_time_minutes', 'video_length_minutes', 'subscribers', 'engagement_rate'
    ]
)

prediction = model.predict(input_data)[0]

# -------------------------------
# Output
# -------------------------------
st.success(f"💰 Estimated Ad Revenue: **${prediction:.2f} USD**")
st.info("Prediction is based on engagement, audience, and content features using a trained ML model.")

# -------------------------------
# Visualization: Prediction Context
# -------------------------------
st.subheader("📊 Predicted Revenue in Context")

if not original_df.empty and 'ad_revenue_usd' in original_df.columns:
    fig, ax = plt.subplots()
    ax.hist(original_df['ad_revenue_usd'], bins=50)
    ax.axvline(prediction, linestyle='--', linewidth=2, label="Your Prediction")
    ax.set_title("Revenue Distribution")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Dataset not available for visualization.")

# -------------------------------
# Category Distribution
# -------------------------------
st.subheader("📊 Category Distribution (%)")

if not original_df.empty and 'category' in original_df.columns:
    category_counts = original_df['category'].value_counts(normalize=True) * 100

    fig2, ax2 = plt.subplots()
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    for i, v in enumerate(category_counts.values):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center')

    st.pyplot(fig2)
else:
    st.warning("Category data not available.")

# -------------------------------
# Correlation Heatmap
# -------------------------------
st.subheader("📊 Feature Correlation Heatmap")

if not original_df.empty:
    numeric_df = original_df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        fig3, ax3 = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("No numeric data for correlation.")
else:
    st.warning("Dataset not available.")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.write("⚠️ Note: This is an estimation. Actual revenue may vary.")
