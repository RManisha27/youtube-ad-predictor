import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="📹",
    layout="wide"
)

# -------------------------------
# File Paths
# -------------------------------
MODEL_FILE = "youtube_model.pkl"
DATA_FILE = "YouTube_Monetization_Modeler.csv"

# -------------------------------
# Debug: Show Files
# -------------------------------
st.write("📁 Files available in app directory:")

try:
    files = os.listdir(".")
    st.write(files)
except Exception as e:
    st.error(f"Unable to list files: {e}")

# -------------------------------
# Check Model File
# -------------------------------
if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Model file not found: {MODEL_FILE}")
    st.stop()

# -------------------------------
# Load Model
# -------------------------------
try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# Load Dataset
# -------------------------------
original_df = pd.DataFrame()

if os.path.exists(DATA_FILE):

    try:
        file_size = os.path.getsize(DATA_FILE) / (1024 * 1024)

        st.info(
            f"📄 Dataset found: {DATA_FILE} "
            f"({file_size:.2f} MB)"
        )

        original_df = pd.read_csv(
            DATA_FILE,
            encoding="utf-8",
            low_memory=False
        )

        st.success(
            f"✅ Dataset loaded successfully: "
            f"{original_df.shape[0]:,} rows × "
            f"{original_df.shape[1]} columns"
        )

    except UnicodeDecodeError:
        try:
            original_df = pd.read_csv(
                DATA_FILE,
                encoding="latin1",
                low_memory=False
            )

            st.success(
                f"✅ Dataset loaded successfully using latin1 encoding: "
                f"{original_df.shape[0]:,} rows × "
                f"{original_df.shape[1]} columns"
            )

        except Exception as e:
            st.error(f"❌ CSV reading error: {e}")

    except Exception as e:
        st.error(f"❌ Error reading dataset: {e}")

else:
    st.warning(
        f"⚠️ Dataset file not found: {DATA_FILE}"
    )

# -------------------------------
# App Title
# -------------------------------
st.title("📹 YouTube Ad Revenue Predictor")

st.write(
    "Estimate your potential YouTube ad revenue using "
    "Machine Learning 📊"
)

st.write("🤖 Model Type: Regression Model")

# -------------------------------
# User Inputs
# -------------------------------

views = st.number_input(
    "Views",
    min_value=0,
    value=10000
)

likes = st.number_input(
    "Likes",
    min_value=0,
    value=1100
)

comments = st.number_input(
    "Comments",
    min_value=0,
    value=274
)

watch_time = st.slider(
    "Watch Time (Minutes)",
    min_value=10000.0,
    max_value=70000.0,
    value=37500.0,
    step=100.0
)

vid_length = st.number_input(
    "Video Length (Minutes)",
    min_value=0.0,
    value=16.0
)

subs = st.number_input(
    "Subscribers",
    min_value=0,
    value=500000
)

cat = st.selectbox(
    "Category",
    [
        "Gaming",
        "Education",
        "Entertainment",
        "Tech",
        "Music",
        "Lifestyle"
    ]
)

dev = st.selectbox(
    "Device",
    [
        "Mobile",
        "Desktop",
        "Tablet"
    ]
)

country = st.selectbox(
    "Country",
    [
        "USA",
        "India",
        "UK",
        "Brazil",
        "CA",
        "DE",
        "AU"
    ]
)

# -------------------------------
# Feature Engineering
# -------------------------------

engagement_rate = (
    (likes + comments) / views
    if views > 0
    else 0
)

# -------------------------------
# Prediction Input
# -------------------------------

input_data = pd.DataFrame(
    [[
        cat,
        dev,
        country,
        views,
        likes,
        comments,
        watch_time,
        vid_length,
        subs,
        engagement_rate
    ]],
    columns=[
        "category",
        "device",
        "country",
        "views",
        "likes",
        "comments",
        "watch_time_minutes",
        "video_length_minutes",
        "subscribers",
        "engagement_rate"
    ]
)

# -------------------------------
# Prediction
# -------------------------------

try:

    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Estimated Ad Revenue: **${prediction:.2f} USD**"
    )

    st.info(
        "Prediction is based on engagement, audience, "
        "and content features using a trained ML model."
    )

except Exception as e:

    st.error(
        f"❌ Prediction Error: {e}"
    )

    st.stop()

# =========================================================
# VISUALIZATIONS
# =========================================================

# -------------------------------
# Prediction Context
# -------------------------------

st.subheader("📊 Predicted Revenue in Context")

if (
    not original_df.empty
    and "ad_revenue_usd" in original_df.columns
):

    fig, ax = plt.subplots()

    ax.hist(
        original_df["ad_revenue_usd"].dropna(),
        bins=50
    )

    ax.axvline(
        prediction,
        linestyle="--",
        linewidth=2,
        label="Your Prediction"
    )

    ax.set_title("Revenue Distribution")

    ax.set_xlabel("Ad Revenue (USD)")
    ax.set_ylabel("Number of Videos")

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

elif not original_df.empty:

    st.warning(
        "⚠️ Dataset loaded, but 'ad_revenue_usd' "
        "column was not found."
    )

else:

    st.warning(
        "⚠️ Dataset not available for this visualization."
    )

# -------------------------------
# Category Distribution
# -------------------------------

st.subheader("📊 Category Distribution (%)")

if (
    not original_df.empty
    and "category" in original_df.columns
):

    category_counts = (
        original_df["category"]
        .value_counts(normalize=True)
        * 100
    )

    fig2, ax2 = plt.subplots()

    sns.barplot(
        x=category_counts.index,
        y=category_counts.values,
        ax=ax2
    )

    ax2.set_title("Video Category Distribution")

    ax2.set_xlabel("Category")
    ax2.set_ylabel("Percentage (%)")

    ax2.tick_params(axis="x", rotation=45)

    for i, v in enumerate(category_counts.values):

        ax2.text(
            i,
            v + 0.5,
            f"{v:.1f}%",
            ha="center"
        )

    st.pyplot(fig2)

    plt.close(fig2)

elif not original_df.empty:

    st.warning(
        "⚠️ Dataset loaded, but 'category' "
        "column was not found."
    )

else:

    st.warning(
        "⚠️ Category data not available."
    )

# -------------------------------
# Correlation Heatmap
# -------------------------------

st.subheader("📊 Feature Correlation Heatmap")

if not original_df.empty:

    numeric_df = original_df.select_dtypes(
        include=np.number
    )

    if not numeric_df.empty:

        fig3, ax3 = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            fmt=".2f",
            ax=ax3
        )

        ax3.set_title(
            "Feature Correlation Heatmap"
        )

        st.pyplot(fig3)

        plt.close(fig3)

    else:

        st.warning(
            "⚠️ No numeric data available "
            "for correlation."
        )

else:

    st.warning(
        "⚠️ Dataset not available."
    )

# -------------------------------
# Dataset Information
# -------------------------------

if not original_df.empty:

    st.subheader("📋 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{original_df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            original_df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(original_df.isnull().sum().sum())
        )

# -------------------------------
# Footer
# -------------------------------

st.write("---")

st.write(
    "⚠️ Note: This is an estimation. "
    "Actual revenue may vary."
)
