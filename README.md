📹 YouTube Ad Revenue Predictor
This project predicts the potential ad revenue of a YouTube video based on metrics like views, likes, comments, watch time, subscribers, and video metadata (category, device, country).

It includes a pre-trained machine learning model, a Streamlit web app, and an interactive demo via ngrok.

🔹 Features

Predict YouTube ad revenue using a trained Random Forest Regressor model.

Input video metrics: views, likes, comments, watch time, video length, subscribers.

Select Category, Device, and Country for prediction context.

Interactive histogram showing your predicted revenue in context of real data.

Category distribution visualization (percentage of videos per category).

Correlation heatmap of numeric features.

Fully deployed on Google Colab with ngrok for a public URL.

🔹 Demo

You can try the live demo in your browser via ngrok:

https://nonroyal-maura-magisterial.ngrok-free.dev


Note: The URL is temporary and changes every time the Colab session is restarted
.

🔹 How It Works

Features used:

Numerical: views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, engagement_rate

Categorical: category, device, country

Target: ad_revenue_usd

Preprocessing:

Median imputation for missing numeric values

Standard scaling for numeric features

One-hot encoding for categorical features

Models trained: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Ridge Regression

Best model selected: Random Forest Regressor

🔹 Usage Notes

Ensure dataset and model are in the correct Google Drive paths.
Predictions are estimates; actual revenue may vary.

🔹 Requirements

Python 3.9+

Packages:

streamlit
pyngrok
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib


Install via:

pip install -r requirements.txt

