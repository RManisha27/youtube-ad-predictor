import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

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
DATA_FILE = "youtube_ad_revenue_dataset.csv"

# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Model file not found: {MODEL_FILE}")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# =========================================================
# LOAD DATASET
# =========================================================

original_df = pd.DataFrame()

if not os.path.exists(DATA_FILE):
    st.error(f"❌ Dataset file not found: {DATA_FILE}")
    st.stop()

try:
    original_df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8",
        low_memory=False
    )
except UnicodeDecodeError:
    try:
        original_df = pd.read_csv(
            DATA_FILE,
            encoding="latin1",
            low_memory=False
        )
    except Exception as e:
        st.error(f"❌ CSV reading error: {e}")
        st.stop()
except Exception as e:
    st.error(f"❌ Error reading dataset: {e}")
    st.stop()

# =========================================================
# HEADER
# =========================================================

st.title("📹 YouTube Ad Revenue Predictor")

st.markdown(
    """
    ### 🤖 Machine Learning Dashboard
    Predict estimated YouTube ad revenue and explore
    audience, engagement, category, device and revenue patterns.
    """
)

st.success(
    f"✅ Dataset loaded: {len(original_df):,} rows × "
    f"{len(original_df.columns)} columns"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🎛️ Dashboard Controls")

# ---------------------------------------------------------
# Dataset filters
# ---------------------------------------------------------

filtered_df = original_df.copy()

if "category" in original_df.columns:

    categories = sorted(
        original_df["category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "🎮 Category",
        categories,
        default=categories
    )

    if selected_categories:
        filtered_df = filtered_df[
            filtered_df["category"].astype(str).isin(
                selected_categories
            )
        ]

if "country" in original_df.columns:

    countries = sorted(
        original_df["country"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_countries = st.sidebar.multiselect(
        "🌎 Country",
        countries,
        default=countries
    )

    if selected_countries:
        filtered_df = filtered_df[
            filtered_df["country"].astype(str).isin(
                selected_countries
            )
        ]

if "device" in original_df.columns:

    devices = sorted(
        original_df["device"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_devices = st.sidebar.multiselect(
        "📱 Device",
        devices,
        default=devices
    )

    if selected_devices:
        filtered_df = filtered_df[
            filtered_df["device"].astype(str).isin(
                selected_devices
            )
        ]

# =========================================================
# KPI SECTION
# =========================================================

st.header("📊 Dataset Overview")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "🎬 Videos",
        f"{len(filtered_df):,}"
    )

with k2:

    if "views" in filtered_df.columns:
        total_views = filtered_df["views"].sum()
        st.metric(
            "👁️ Total Views",
            f"{total_views:,.0f}"
        )
    else:
        st.metric("👁️ Total Views", "N/A")

with k3:

    if "ad_revenue_usd" in filtered_df.columns:

        avg_revenue = (
            filtered_df["ad_revenue_usd"]
            .mean()
        )

        st.metric(
            "💰 Avg Revenue",
            f"${avg_revenue:,.2f}"
        )

    else:
        st.metric("💰 Avg Revenue", "N/A")

with k4:

    if "likes" in filtered_df.columns:

        total_likes = filtered_df["likes"].sum()

        st.metric(
            "👍 Total Likes",
            f"{total_likes:,.0f}"
        )

    else:
        st.metric("👍 Total Likes", "N/A")

# =========================================================
# PREDICTION SECTION
# =========================================================

st.header("🎯 Predict Your Video Revenue")

p1, p2, p3 = st.columns(3)

with p1:

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

with p2:

    watch_time = st.number_input(
        "⏱️ Watch Time (Minutes)",
        min_value=0.0,
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

with p3:

    # Use dataset categories when available
    if "category" in original_df.columns:

        prediction_categories = sorted(
            original_df["category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        prediction_categories = [
            "Gaming",
            "Education",
            "Entertainment",
            "Tech",
            "Music",
            "Lifestyle"
        ]

    cat = st.selectbox(
        "🎮 Category",
        prediction_categories
    )

    if "device" in original_df.columns:

        prediction_devices = sorted(
            original_df["device"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        prediction_devices = [
            "Mobile",
            "Desktop",
            "Tablet"
        ]

    dev = st.selectbox(
        "📱 Device",
        prediction_devices
    )

    if "country" in original_df.columns:

        prediction_countries = sorted(
            original_df["country"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        prediction_countries = [
            "USA",
            "India",
            "UK",
            "Brazil",
            "CA",
            "DE",
            "AU"
        ]

    country = st.selectbox(
        "🌎 Country",
        prediction_countries
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
# MODEL INPUT
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

try:

    prediction = float(
        model.predict(input_data)[0]
    )

except Exception as e:

    st.error(
        f"❌ Prediction Error: {e}"
    )

    st.stop()

# =========================================================
# PREDICTION KPI CARDS
# =========================================================

st.subheader("💰 Prediction Result")

r1, r2, r3 = st.columns(3)

with r1:

    st.metric(
        "Estimated Revenue",
        f"${prediction:,.2f}"
    )

with r2:

    st.metric(
        "Engagement Rate",
        f"{engagement_rate:.2%}"
    )

with r3:

    if views > 0:

        revenue_per_1000_views = (
            prediction / views
        ) * 1000

        st.metric(
            "Estimated Revenue / 1K Views",
            f"${revenue_per_1000_views:.2f}"
        )

# =========================================================
# REVENUE GAUGE
# =========================================================

st.subheader("🎯 Predicted Revenue Gauge")

if "ad_revenue_usd" in original_df.columns:

    max_revenue = float(
        original_df["ad_revenue_usd"]
        .dropna()
        .quantile(0.99)
    )

    if max_revenue <= 0:
        max_revenue = max(
            prediction * 1.5,
            100
        )

else:

    max_revenue = max(
        prediction * 1.5,
        100
    )

gauge_max = max(
    max_revenue,
    prediction * 1.2,
    100
)

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={
            "text": "Estimated Ad Revenue (USD)"
        },
        number={
            "prefix": "$",
            "valueformat": ",.2f"
        },
        gauge={
            "axis": {
                "range": [0, gauge_max]
            },
            "bar": {
                "thickness": 0.7
            }
        }
    )
)

fig_gauge.update_layout(
    height=350
)

st.plotly_chart(
    fig_gauge,
    use_container_width=True
)

# =========================================================
# INTERACTIVE DASHBOARD TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💰 Revenue",
        "🎮 Categories",
        "👁️ Engagement",
        "🔥 Correlations"
    ]
)

# =========================================================
# TAB 1 - REVENUE
# =========================================================

with tab1:

    st.subheader(
        "💰 Revenue Distribution"
    )

    if "ad_revenue_usd" in filtered_df.columns:

        revenue_df = filtered_df[
            ["ad_revenue_usd"]
        ].dropna()

        fig = px.histogram(
            revenue_df,
            x="ad_revenue_usd",
            nbins=50,
            title="Interactive Revenue Distribution",
            labels={
                "ad_revenue_usd":
                "Ad Revenue (USD)"
            }
        )

        fig.add_vline(
            x=prediction,
            line_dash="dash",
            annotation_text="Your Prediction",
            annotation_position="top"
        )

        fig.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Revenue by category
    # -----------------------------------------------------

    if (
        "category" in filtered_df.columns
        and "ad_revenue_usd" in filtered_df.columns
    ):

        category_revenue = (
            filtered_df
            .groupby("category")[
                "ad_revenue_usd"
            ]
            .mean()
            .reset_index()
            .sort_values(
                "ad_revenue_usd",
                ascending=False
            )
        )

        fig_cat = px.bar(
            category_revenue,
            x="category",
            y="ad_revenue_usd",
            title="Average Revenue by Category",
            labels={
                "category": "Category",
                "ad_revenue_usd":
                "Average Revenue (USD)"
            },
            text_auto=".2f"
        )

        st.plotly_chart(
            fig_cat,
            use_container_width=True
        )

# =========================================================
# TAB 2 - CATEGORIES
# =========================================================

with tab2:

    st.subheader(
        "🎮 Category Analysis"
    )

    if "category" in filtered_df.columns:

        category_counts = (
            filtered_df["category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "category",
            "count"
        ]

        fig_category = px.pie(
            category_counts,
            names="category",
            values="count",
            hole=0.45,
            title="Video Category Distribution"
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

        # Category revenue box plot

        if "ad_revenue_usd" in filtered_df.columns:

            fig_box = px.box(
                filtered_df,
                x="category",
                y="ad_revenue_usd",
                color="category",
                title="Revenue Distribution by Category",
                labels={
                    "category": "Category",
                    "ad_revenue_usd":
                    "Ad Revenue (USD)"
                }
            )

            st.plotly_chart(
                fig_box,
                use_container_width=True
            )

# =========================================================
# TAB 3 - ENGAGEMENT
# =========================================================

with tab3:

    st.subheader(
        "👁️ Views vs Revenue"
    )

    if (
        "views" in filtered_df.columns
        and "ad_revenue_usd"
        in filtered_df.columns
    ):

        plot_df = filtered_df[
            [
                "views",
                "ad_revenue_usd"
            ]
        ].dropna()

        # Limit plotted rows for browser performance
        if len(plot_df) > 10000:
            plot_df = plot_df.sample(
                10000,
                random_state=42
            )

        fig_scatter = px.scatter(
            plot_df,
            x="views",
            y="ad_revenue_usd",
            opacity=0.5,
            title="Views vs Ad Revenue",
            labels={
                "views": "Views",
                "ad_revenue_usd":
                "Ad Revenue (USD)"
            }
        )

        fig_scatter.add_trace(
            go.Scatter(
                x=[views],
                y=[prediction],
                mode="markers",
                marker={
                    "size": 18,
                    "symbol": "star"
                },
                name="Your Video"
            )
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Likes vs comments
    # -----------------------------------------------------

    if (
        "likes" in filtered_df.columns
        and "comments" in filtered_df.columns
    ):

        engagement_df = filtered_df[
            [
                "likes",
                "comments"
            ]
        ].dropna()

        if len(engagement_df) > 10000:

            engagement_df = engagement_df.sample(
                10000,
                random_state=42
            )

        fig_engagement = px.scatter(
            engagement_df,
            x="likes",
            y="comments",
            title="Likes vs Comments",
            opacity=0.5,
            labels={
                "likes": "Likes",
                "comments": "Comments"
            }
        )

        st.plotly_chart(
            fig_engagement,
            use_container_width=True
        )

# =========================================================
# TAB 4 - CORRELATION
# =========================================================

with tab4:

    st.subheader(
        "🔥 Feature Correlation Heatmap"
    )

    numeric_df = filtered_df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        correlation = numeric_df.corr()

        fig_heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Interactive Feature Correlation"
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )

    else:

        st.warning(
            "Not enough numeric columns for correlation analysis."
        )

# =========================================================
# DEVICE ANALYSIS
# =========================================================

if (
    "device" in filtered_df.columns
    and "ad_revenue_usd" in filtered_df.columns
):

    st.header("📱 Device Analysis")

    device_revenue = (
        filtered_df
        .groupby("device")["ad_revenue_usd"]
        .mean()
        .reset_index()
    )

    fig_device = px.bar(
        device_revenue,
        x="device",
        y="ad_revenue_usd",
        color="device",
        title="Average Revenue by Device",
        labels={
            "device": "Device",
            "ad_revenue_usd":
            "Average Revenue (USD)"
        },
        text_auto=".2f"
    )

    st.plotly_chart(
        fig_device,
        use_container_width=True
    )

# =========================================================
# COUNTRY ANALYSIS
# =========================================================

if (
    "country" in filtered_df.columns
    and "ad_revenue_usd" in filtered_df.columns
):

    st.header("🌎 Country Analysis")

    country_revenue = (
        filtered_df
        .groupby("country")["ad_revenue_usd"]
        .mean()
        .reset_index()
        .sort_values(
            "ad_revenue_usd",
            ascending=False
        )
    )

    fig_country = px.bar(
        country_revenue,
        x="country",
        y="ad_revenue_usd",
        color="country",
        title="Average Revenue by Country",
        labels={
            "country": "Country",
            "ad_revenue_usd":
            "Average Revenue (USD)"
        },
        text_auto=".2f"
    )

    st.plotly_chart(
        fig_country,
        use_container_width=True
    )

# =========================================================
# DATASET INFORMATION
# =========================================================

st.header("📋 Dataset Information")

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric(
        "Rows",
        f"{len(filtered_df):,}"
    )

with d2:
    st.metric(
        "Columns",
        len(filtered_df.columns)
    )

with d3:
    st.metric(
        "Missing Values",
        int(
            filtered_df
            .isnull()
            .sum()
            .sum()
        )
    )

with d4:
    st.metric(
        "Dataset Size",
        f"{os.path.getsize(DATA_FILE) / (1024 * 1024):.2f} MB"
    )

# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("🔍 View Dataset"):

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application provides an estimated YouTube "
    "ad revenue prediction. Actual revenue may vary."
)
