import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="📹",
    layout="wide"
)

# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = "youtube_model.pkl"

# FIXED: This is the actual filename in your GitHub repository
DATA_FILE = "youtube_ad_revenue_dataset.csv"

# =========================================================
# APP TITLE
# =========================================================

st.title("📹 YouTube Ad Revenue Predictor")

st.write(
    "Estimate your potential YouTube ad revenue using "
    "Machine Learning 📊"
)

st.write("🤖 Model Type: Regression Model")

# =========================================================
# CHECK FILES
# =========================================================

with st.expander("📁 Application Files", expanded=False):

    try:
        files = os.listdir(".")
        st.write(files)

    except Exception as e:
        st.error(f"Unable to list files: {e}")

# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        f"❌ Model file not found: {MODEL_FILE}"
    )

    st.stop()

# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load(MODEL_FILE)

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()

# =========================================================
# LOAD DATASET
# =========================================================

original_df = pd.DataFrame()

if os.path.exists(DATA_FILE):

    try:

        file_size = (
            os.path.getsize(DATA_FILE)
            / (1024 * 1024)
        )

        st.success(
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
                f"✅ Dataset loaded successfully: "
                f"{original_df.shape[0]:,} rows × "
                f"{original_df.shape[1]} columns"
            )

        except Exception as e:

            st.error(
                f"❌ CSV reading error: {e}"
            )

    except Exception as e:

        st.error(
            f"❌ Error reading dataset: {e}"
        )

else:

    st.warning(
        f"⚠️ Dataset file not found: {DATA_FILE}"
    )

# =========================================================
# USER INPUT SECTION
# =========================================================

st.header("🎯 Enter Video Details")

col1, col2, col3 = st.columns(3)

# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    views = st.number_input(
        "👁️ Views",
        min_value=0,
        value=10000,
        step=1000
    )

    likes = st.number_input(
        "👍 Likes",
        min_value=0,
        value=1100,
        step=100
    )

    comments = st.number_input(
        "💬 Comments",
        min_value=0,
        value=274,
        step=10
    )

# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    watch_time = st.slider(
        "⏱️ Watch Time (Minutes)",
        min_value=10000.0,
        max_value=70000.0,
        value=37500.0,
        step=100.0
    )

    vid_length = st.number_input(
        "🎬 Video Length (Minutes)",
        min_value=0.0,
        value=16.0,
        step=1.0
    )

    subs = st.number_input(
        "👥 Subscribers",
        min_value=0,
        value=500000,
        step=1000
    )

# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

with col3:

    cat = st.selectbox(
        "🎮 Category",
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
        "📱 Device",
        [
            "Mobile",
            "Desktop",
            "Tablet"
        ]
    )

    country = st.selectbox(
        "🌎 Country",
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

# =========================================================
# FEATURE ENGINEERING
# =========================================================

engagement_rate = (
    (likes + comments) / views
    if views > 0
    else 0
)

# =========================================================
# SHOW ENGAGEMENT RATE
# =========================================================

st.metric(
    "📈 Engagement Rate",
    f"{engagement_rate:.2%}"
)

# =========================================================
# CREATE INPUT DATA
# =========================================================

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

# =========================================================
# PREDICTION
# =========================================================

st.header("💰 Revenue Prediction")

try:

    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Estimated Ad Revenue: **${prediction:.2f} USD**"
    )

    st.info(
        "The prediction is generated using the trained "
        "machine learning regression model."
    )

except Exception as e:

    st.error(
        f"❌ Prediction Error: {e}"
    )

    st.stop()

# =========================================================
# VISUALIZATION 1
# REVENUE DISTRIBUTION
# =========================================================

st.header("📊 Revenue Analysis")

if (
    not original_df.empty
    and "ad_revenue_usd" in original_df.columns
):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    revenue_data = (
        original_df["ad_revenue_usd"]
        .dropna()
    )

    ax.hist(
        revenue_data,
        bins=50
    )

    ax.axvline(
        prediction,
        linestyle="--",
        linewidth=2,
        label="Your Prediction"
    )

    ax.set_title(
        "YouTube Ad Revenue Distribution"
    )

    ax.set_xlabel(
        "Ad Revenue (USD)"
    )

    ax.set_ylabel(
        "Number of Videos"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

else:

    st.warning(
        "⚠️ Revenue column is not available."
    )

# =========================================================
# VISUALIZATION 2
# CATEGORY DISTRIBUTION
# =========================================================

if (
    not original_df.empty
    and "category" in original_df.columns
):

    st.subheader(
        "📊 Video Category Distribution"
    )

    category_counts = (
        original_df["category"]
        .value_counts(normalize=True)
        * 100
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x=category_counts.index,
        y=category_counts.values,
        ax=ax2
    )

    ax2.set_title(
        "Video Category Distribution"
    )

    ax2.set_xlabel(
        "Category"
    )

    ax2.set_ylabel(
        "Percentage (%)"
    )

    ax2.tick_params(
        axis="x",
        rotation=45
    )

    for i, value in enumerate(
        category_counts.values
    ):

        ax2.text(
            i,
            value + 0.5,
            f"{value:.1f}%",
            ha="center"
        )

    st.pyplot(fig2)

    plt.close(fig2)

# =========================================================
# VISUALIZATION 3
# CORRELATION HEATMAP
# =========================================================

if not original_df.empty:

    st.subheader(
        "🔥 Feature Correlation Heatmap"
    )

    numeric_df = (
        original_df
        .select_dtypes(
            include=np.number
        )
    )

    if not numeric_df.empty:

        correlation = numeric_df.corr()

        fig3, ax3 = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            ax=ax3
        )

        ax3.set_title(
            "Feature Correlation Heatmap"
        )

        st.pyplot(fig3)

        plt.close(fig3)

# =========================================================
# VISUALIZATION 4
# VIEWS VS REVENUE
# =========================================================

if (
    not original_df.empty
    and "views" in original_df.columns
    and "ad_revenue_usd" in original_df.columns
):

    st.subheader(
        "👁️ Views vs Ad Revenue"
    )

    fig4, ax4 = plt.subplots(
        figsize=(10, 5)
    )

    ax4.scatter(
        original_df["views"],
        original_df["ad_revenue_usd"],
        alpha=0.3
    )

    ax4.scatter(
        views,
        prediction,
        s=100,
        marker="*",
        label="Your Video"
    )

    ax4.set_title(
        "Views vs Ad Revenue"
    )

    ax4.set_xlabel(
        "Views"
    )

    ax4.set_ylabel(
        "Ad Revenue (USD)"
    )

    ax4.legend()

    st.pyplot(fig4)

    plt.close(fig4)

# =========================================================
# DATASET INFORMATION
# =========================================================

if not original_df.empty:

    st.header(
        "📋 Dataset Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📊 Rows",
            f"{original_df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "📁 Columns",
            original_df.shape[1]
        )

    with col3:

        st.metric(
            "❗ Missing Values",
            int(
                original_df
                .isnull()
                .sum()
                .sum()
            )
        )

    with col4:

        st.metric(
            "💾 Dataset Size",
            f"{os.path.getsize(DATA_FILE) / (1024 * 1024):.2f} MB"
        )

# =========================================================
# DATA PREVIEW
# =========================================================

if not original_df.empty:

    with st.expander(
        "🔍 View Dataset Preview"
    ):

        st.dataframe(
            original_df.head(10),
            use_container_width=True
        )

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.caption(
    "⚠️ This application provides an estimated "
    "YouTube ad revenue prediction. Actual revenue "
    "may vary depending on multiple factors."
)
