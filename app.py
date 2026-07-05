import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import io
import base64
from datetime import datetime, timedelta

try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet


st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    /* Main background - Premium gradient */
    .stAppViewContainer { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    .main { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Block containers and general spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }
    
    /* Metrics styling - Premium cards with glow */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        border: 2px solid #e0e7ff;
        border-left: 5px solid #6366f1;
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15), 0 0 1px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.25), 0 0 20px rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling - Dark premium */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1a1f3a 50%, #2d1b4e 100%) !important;
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton"] > button {
        width: 100%;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] [data-testid="stBaseButton"] > button:hover {
        background: rgba(99, 102, 241, 0.4);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Radio button styling */
    [data-testid="stSidebar"] .stRadio > label > div {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label > input {
        accent-color: #6366f1;
    }
    
    /* Divider in sidebar */
    [data-testid="stSidebar"] hr {
        border-color: rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Page hero section - Modern gradient */
    .page-hero {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e0e7ff;
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.12);
    }
    
    .page-hero h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .page-hero p {
        margin: 0;
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Filter controls spacing */
    .filter-container {
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%);
        padding: 1.5rem;
        border-radius: 18px;
        border: 2px solid rgba(224, 231, 255, 0.5);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.08);
    }
    
    .filter-container [data-testid="stSelectbox"] {
        margin-bottom: 0.8rem;
    }
    
    /* Selectbox styling */
    [data-testid="stSelectbox"] {
        border-radius: 12px;
    }
    
    [data-testid="stSelectbox"] > div {
        border-radius: 12px;
        border: 2px solid #e0e7ff;
        background: white;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSelectbox"] > div:hover {
        border-color: #6366f1;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
    }
    
    /* Slider styling */
    [data-testid="stSlider"] > div > div > div > input {
        accent-color: #6366f1;
    }
    
    /* Info boxes */
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.08);
    }
    
    /* Section cards */
    .section-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e0e7ff;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.08);
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }
    
    /* Chart styling */
    [data-testid="plotly-graph"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e0e7ff;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.08);
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e0e7ff;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.08);
    }
    
    /* Button styling */
    button {
        background: linear-gradient(135deg, #6366f1 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e0e7ff;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.08);
    }
    
    /* Subheader styling */
    h2 {
        color: #1e293b !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
    }
    
    /* Subheader styling */
    h2 {
        color: #1e293b !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #334155 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar branding */
    .sidebar-brand {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #6366f1, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.5rem;
    }
    
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    /* Navigation styling */
    .nav-section-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        padding: 0.8rem 0.5rem 0.5rem 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] {
        padding: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
        background: rgba(99, 102, 241, 0.05);
        border: 2px solid transparent;
        border-radius: 12px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        transform: translateX(4px);
    }
    
    [data-testid="stSidebar"] .stRadio > div > input:checked + label {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(37, 99, 235, 0.2) 100%);
        border: 2px solid #6366f1;
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.5), inset 0 0 10px rgba(99, 102, 241, 0.2);
        border-left: 5px solid #6366f1;
    }
    
    [data-testid="stSidebar"] .stRadio > label > div {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_season_name(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("train.csv")

    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["Order Date"]).copy()
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Week_Number"] = df["Order Date"].dt.isocalendar().week
    df["Order_Day_of_Week"] = df["Order Date"].dt.dayofweek
    df["Order_Quarter"] = df["Order Date"].dt.quarter
    df["Order_Season"] = df["Order_Month"].apply(get_season_name)
    df["Order_Month_Name"] = df["Order Date"].dt.month_name()
    return df


@st.cache_data(show_spinner=False)
def prepare_clustering_data(df):
    df_filtered_dates = df.dropna(subset=["Order Date"]).copy()
    monthly_sub_category_sales = (
        df_filtered_dates.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="ME")])["Sales"]
        .sum()
        .unstack(fill_value=0)
    )

    total_sales_volume = df_filtered_dates.groupby("Sub-Category")["Sales"].sum()
    total_orders_per_subcategory = df_filtered_dates.groupby("Sub-Category")["Order ID"].nunique()
    average_order_value = total_sales_volume / total_orders_per_subcategory
    sales_volatility = monthly_sub_category_sales.std(axis=1).fillna(0)

    growth_rates = []
    for sub_cat in monthly_sub_category_sales.index:
        yearly_sales = monthly_sub_category_sales.loc[sub_cat].resample("YE").sum()
        yoy_growth = yearly_sales.pct_change() * 100
        mean_growth = yoy_growth[np.isfinite(yoy_growth)].mean()
        growth_rates.append({"Sub-Category": sub_cat, "Sales_Growth_Rate": mean_growth})

    sales_growth_rate_df = pd.DataFrame(growth_rates).set_index("Sub-Category")
    sales_growth_rate = sales_growth_rate_df["Sales_Growth_Rate"].fillna(0)

    clustering_df = pd.DataFrame(
        {
            "Total_Sales_Volume": total_sales_volume,
            "Average_Order_Value": average_order_value,
            "Sales_Volatility": sales_volatility,
            "Sales_Growth_Rate": sales_growth_rate,
        }
    ).fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=clustering_df.columns, index=clustering_df.index)

    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clustering_df["Cluster"] = kmeans.fit_predict(scaled_features)

    cluster_labels = {
        0: "Niche & Stable: Low Volume, Low Volatility, Moderate Growth",
        1: "Premium & Growing: High Value, High Volatility, High Growth",
        2: "Core & Mature: High Volume, Moderate Volatility, Low Growth",
        3: "Emerging Growth: Moderate Volume, Moderate Volatility, High Growth",
    }
    clustering_df["Cluster_Label"] = clustering_df["Cluster"].map(cluster_labels)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_features_df)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=clustering_df.index)
    pca_df["Cluster"] = clustering_df["Cluster"]
    pca_df["Cluster_Label"] = clustering_df["Cluster_Label"]

    return clustering_df, pca_df, pca


@st.cache_data(show_spinner=False)
def prepare_anomaly_data(df):
    weekly_sales = df.set_index("Order Date")["Sales"].resample("W").sum()
    weekly_sales_df = weekly_sales.to_frame(name="Sales").reset_index()
    weekly_sales_df.rename(columns={"Order Date": "Date"}, inplace=True)

    isolation_model = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales_df["anomaly_isolation"] = isolation_model.fit_predict(weekly_sales_df[["Sales"]])
    weekly_sales_df["isolation_flag"] = (weekly_sales_df["anomaly_isolation"] == -1).astype(int)

    weekly_sales_df["rolling_mean"] = weekly_sales_df["Sales"].rolling(window=4).mean()
    weekly_sales_df["rolling_std"] = weekly_sales_df["Sales"].rolling(window=4).std()
    weekly_sales_df["z_score"] = (weekly_sales_df["Sales"] - weekly_sales_df["rolling_mean"]) / weekly_sales_df["rolling_std"]
    weekly_sales_df["anomaly_zscore"] = (np.abs(weekly_sales_df["z_score"]) > 2).astype(int)

    combined = weekly_sales_df.copy()
    combined["Detection_Method"] = np.where(
        (combined["isolation_flag"] == 1) & (combined["anomaly_zscore"] == 1),
        "Isolation Forest + Z-Score",
        np.where(combined["isolation_flag"] == 1, "Isolation Forest", "Z-Score"),
    )
    combined = combined[(combined["isolation_flag"] == 1) | (combined["anomaly_zscore"] == 1)].copy()
    return weekly_sales_df, combined


def get_seasonal_decomposition(series):
    """Get seasonal decomposition of time series."""
    if len(series) < 24:
        return None
    try:
        result = seasonal_decompose(series, model='additive', period=12)
        return result
    except:
        return None


def get_data_quality_report(df):
    """Generate data quality report."""
    report = {}
    report['Total Records'] = len(df)
    report['Missing Values'] = df.isnull().sum().sum()
    report['Missing %'] = round((report['Missing Values'] / (len(df) * len(df.columns)) * 100), 2)
    report['Numeric Columns'] = len(df.select_dtypes(include=[np.number]).columns)
    report['Categorical Columns'] = len(df.select_dtypes(include=['object']).columns)
    if 'Sales' in df.columns:
        sales = df['Sales']
        report['Sales Mean'] = round(sales.mean(), 2)
        report['Sales Std Dev'] = round(sales.std(), 2)
        report['Sales Skewness'] = round(stats.skew(sales.dropna()), 2)
        report['Sales Kurtosis'] = round(stats.kurtosis(sales.dropna()), 2)
        report['Sales Min'] = round(sales.min(), 2)
        report['Sales Max'] = round(sales.max(), 2)
    return report


def get_xgboost_feature_importance(model, feature_names):
    """Extract feature importance from XGBoost model."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    return importance_df


def get_model_recommendation(mae_xgb, mae_sarima, mae_prophet):
    """Recommend best model based on MAE."""
    models = {'XGBoost': mae_xgb, 'SARIMA': mae_sarima, 'Prophet': mae_prophet}
    best_model = min(models, key=models.get)
    max_mae = max(mae_xgb, mae_sarima, mae_prophet)
    scores = {model: round(100 * (1 - metrics/max_mae), 1) for model, metrics in models.items()}
    return best_model, scores



def prepare_feature_frame(series):
    df_xgb = series.to_frame(name="Sales").copy()
    df_xgb["Lag_1"] = df_xgb["Sales"].shift(1)
    df_xgb["Lag_2"] = df_xgb["Sales"].shift(2)
    df_xgb["Lag_3"] = df_xgb["Sales"].shift(3)
    df_xgb["Rolling_Mean_3"] = df_xgb["Sales"].rolling(window=3).mean().shift(1)
    df_xgb["Month"] = df_xgb.index.month
    df_xgb["Quarter"] = df_xgb.index.quarter
    df_xgb["Season"] = df_xgb["Month"].apply(get_season_name)

    for season in ["Fall", "Spring", "Summer", "Winter"]:
        df_xgb[f"Season_{season}"] = (df_xgb["Season"] == season).astype(int)

    df_xgb = df_xgb.drop(columns=["Season"])
    return df_xgb.dropna().copy()


def train_sarima_forecast(series, horizon=3):
    """Train SARIMA model for forecasting."""
    monthly_sales = series.copy()
    if monthly_sales.index.freq is None:
        monthly_sales = monthly_sales.asfreq("ME")
    
    if len(monthly_sales) < 12:
        return None, None, None, None, None
    
    try:
        test_size = horizon
        train_data = monthly_sales.iloc[:-test_size]
        test_data = monthly_sales.iloc[-test_size:]
        
        model = SARIMAX(train_data, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        
        y_pred_test = results.get_prediction(start=len(train_data), end=len(monthly_sales) - 1).predicted_mean
        mae = np.mean(np.abs(test_data.values - y_pred_test.values))
        rmse = np.sqrt(np.mean((test_data.values - y_pred_test.values) ** 2))
        mape = np.mean(np.abs((test_data.values - y_pred_test.values) / test_data.values)) * 100
        
        forecast_result = results.get_forecast(steps=horizon)
        forecast_values = forecast_result.predicted_mean
        future_dates = pd.date_range(start=monthly_sales.index.max() + pd.DateOffset(months=1), periods=horizon, freq="ME")
        
        forecast_series = pd.Series(forecast_values.values, index=future_dates)
        return monthly_sales, forecast_series, mae, rmse, mape
    except:
        return None, None, None, None, None


def train_prophet_forecast(series, horizon=3):
    """Train Prophet model for forecasting."""
    monthly_sales = series.copy()
    if monthly_sales.index.freq is None:
        monthly_sales = monthly_sales.asfreq("ME")
    
    if len(monthly_sales) < 12:
        return None, None, None, None, None
    
    try:
        test_size = horizon
        train_data = monthly_sales.iloc[:-test_size]
        test_data = monthly_sales.iloc[-test_size:]
        
        df_prophet = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
        model = Prophet(yearly_seasonality=True, interval_width=0.95, yearly_seasonality_prior_scale=10)
        model.fit(df_prophet)
        
        test_forecast = model.make_future_dataframe(periods=0)
        test_pred = model.predict(test_forecast)
        y_pred_test = test_pred['yhat'].iloc[-test_size:].values
        
        mae = np.mean(np.abs(test_data.values - y_pred_test))
        rmse = np.sqrt(np.mean((test_data.values - y_pred_test) ** 2))
        mape = np.mean(np.abs((test_data.values - y_pred_test) / test_data.values)) * 100
        
        future_df = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future_df)
        forecast_values = forecast['yhat'].iloc[-horizon:].values
        future_dates = pd.date_range(start=monthly_sales.index.max() + pd.DateOffset(months=1), periods=horizon, freq="ME")
        
        forecast_series = pd.Series(forecast_values, index=future_dates)
        return monthly_sales, forecast_series, mae, rmse, mape
    except:
        return None, None, None, None, None


def train_and_forecast_segment(series, horizon=3):
    monthly_sales = series.copy()
    if monthly_sales.index.freq is None:
        monthly_sales = monthly_sales.asfreq("ME")

    df_xgb_cleaned = prepare_feature_frame(monthly_sales)
    if df_xgb_cleaned.empty or len(df_xgb_cleaned) < 8:
        return None, None, None, None, None

    X = df_xgb_cleaned.drop(columns=["Sales"])
    y = df_xgb_cleaned["Sales"]

    test_size = horizon
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred_test))
    rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    last_date = df_xgb_cleaned.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="ME")
    future_predictions = []

    last_known_sales = df_xgb_cleaned["Sales"].iloc[-1]
    last_n_sales = df_xgb_cleaned["Sales"].iloc[-min(3, len(df_xgb_cleaned)):]

    for i, date in enumerate(future_dates):
        current_features = pd.Series(index=X.columns, dtype=float)

        if i == 0:
            current_features["Lag_1"] = last_known_sales
            current_features["Lag_2"] = last_n_sales.iloc[-2] if len(last_n_sales) >= 2 else 0.0
            current_features["Lag_3"] = last_n_sales.iloc[-3] if len(last_n_sales) >= 3 else 0.0
        else:
            current_features["Lag_1"] = future_predictions[i - 1]
            current_features["Lag_2"] = future_predictions[i - 2] if i >= 2 else last_known_sales
            current_features["Lag_3"] = future_predictions[i - 3] if i >= 3 else (last_known_sales if i == 2 else (last_n_sales.iloc[-2] if i == 1 else 0.0))

        if i == 0:
            current_features["Rolling_Mean_3"] = df_xgb_cleaned["Sales"].iloc[-min(3, len(df_xgb_cleaned)):].mean()
        elif i == 1:
            rolling_vals = list(df_xgb_cleaned["Sales"].iloc[-min(2, len(df_xgb_cleaned)):].values) + [future_predictions[0]]
            current_features["Rolling_Mean_3"] = pd.Series(rolling_vals).mean()
        elif i == 2:
            rolling_vals = list(df_xgb_cleaned["Sales"].iloc[-min(1, len(df_xgb_cleaned)):].values) + future_predictions[0:2]
            current_features["Rolling_Mean_3"] = pd.Series(rolling_vals).mean()
        else:
            current_features["Rolling_Mean_3"] = pd.Series(future_predictions[i - 3:i]).mean()

        current_features["Month"] = date.month
        current_features["Quarter"] = date.quarter
        season_name = get_season_name(date.month)
        for col in [c for c in X.columns if "Season_" in c]:
            current_features[col] = 1 if col.split("_")[1] == season_name else 0

        feature_frame = pd.DataFrame([current_features[X.columns].values], columns=X.columns, index=[date])
        prediction = model.predict(feature_frame)[0]
        future_predictions.append(float(prediction))

    forecast = pd.Series(future_predictions, index=future_dates, name="Forecasted Sales")
    return model, forecast, mae, rmse, mape


def build_sales_metrics(df, filtered_df):
    return {
        "Total Sales": filtered_df["Sales"].sum(),
        "Total Orders": filtered_df["Order ID"].nunique(),
        "Total Customers": filtered_df["Customer ID"].nunique(),
        "Average Sales per Order": filtered_df["Sales"].mean(),
    }


def show_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sales_overview(df):
    show_page_header("Sales Overview", "Historical performance and business trends across the sales dataset.")

    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
    categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    years = ["All"] + sorted(df["Order_Year"].dropna().astype(int).unique().tolist())

    selected_region = col1.selectbox("Region", regions, label_visibility="collapsed")
    selected_category = col2.selectbox("Category", categories, label_visibility="collapsed")
    selected_year = col3.selectbox("Year", years, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["Order_Year"].astype(int) == int(selected_year)]

    metrics = build_sales_metrics(df, filtered_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"${metrics['Total Sales']:,.0f}")
    c2.metric("Total Orders", f"{metrics['Total Orders']:,}")
    c3.metric("Total Customers", f"{metrics['Total Customers']:,}")
    c4.metric("Average Sales per Order", f"${metrics['Average Sales per Order']:,.2f}")

    st.divider()

    yearly_sales = filtered_df.groupby("Order_Year")["Sales"].sum().reset_index()
    yearly_fig = px.bar(yearly_sales, x="Order_Year", y="Sales", text_auto=".2s", title="Total Sales by Year")
    yearly_fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))

    monthly_sales = filtered_df.set_index("Order Date")["Sales"].resample("ME").sum().reset_index()
    monthly_sales.columns = ["Date", "Sales"]
    monthly_fig = px.line(monthly_sales, x="Date", y="Sales", markers=True, title="Monthly Sales Trend")
    monthly_fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(yearly_fig, use_container_width=True)
    with col2:
        st.plotly_chart(monthly_fig, use_container_width=True)

    monthly_by_year = (
        filtered_df.groupby(["Order_Year", "Order_Month"])["Sales"]
        .sum()
        .reset_index()
    )
    monthly_by_year["Month_Name"] = monthly_by_year["Order_Month"].map(
        {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    )
    seasonality_fig = px.line(
        monthly_by_year,
        x="Month_Name",
        y="Sales",
        color="Order_Year",
        markers=True,
        title="Monthly Sales Across Years",
    )
    seasonality_fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(seasonality_fig, use_container_width=True)

    region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
    category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    top_products = filtered_df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()

    col3, col4 = st.columns(2)
    with col3:
        region_fig = px.bar(region_sales, x="Region", y="Sales", text_auto=".2s", title="Sales by Region")
        region_fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(region_fig, use_container_width=True)
    with col4:
        category_fig = px.bar(category_sales, x="Category", y="Sales", text_auto=".2s", title="Sales by Category")
        category_fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(category_fig, use_container_width=True)

    product_fig = px.bar(top_products, x="Sub-Category", y="Sales", text_auto=".2s", title="Top 10 Products")
    product_fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), xaxis_tickangle=-35)
    st.plotly_chart(product_fig, use_container_width=True)


def render_forecast_explorer(df):
    show_page_header("Forecast Explorer", "Multi-model sales forecasting aligned with the notebook implementation.")

    st.info("The dashboard compares three forecasting models: XGBoost (ML-based), SARIMA (Statistical), and Prophet (Time-series). Select a model to see future sales projections.")

    col1, col2 = st.columns(2)
    with col1:
        forecast_type = st.selectbox("Forecast Type", ["Overall Sales", "Category", "Region"])
    with col2:
        model_choice = st.selectbox("Select Forecasting Model", ["XGBoost", "SARIMA", "Prophet"])
    
    horizon = st.slider("Forecast Horizon", 1, 3, 3)

    if forecast_type == "Overall Sales":
        selected_series = df.set_index("Order Date")["Sales"].resample("ME").sum()
        selected_label = "Overall Sales"
    elif forecast_type == "Category":
        categories = sorted(df["Category"].dropna().unique().tolist())
        selected_category = st.selectbox("Select Category", categories)
        filtered_df = df[df["Category"] == selected_category]
        selected_series = filtered_df.set_index("Order Date")["Sales"].resample("ME").sum()
        selected_label = selected_category
    else:
        regions = sorted(df["Region"].dropna().unique().tolist())
        selected_region = st.selectbox("Select Region", regions)
        filtered_df = df[df["Region"] == selected_region]
        selected_series = filtered_df.set_index("Order Date")["Sales"].resample("ME").sum()
        selected_label = selected_region

    cache_key = f"{model_choice}:{forecast_type}:{selected_label}:{horizon}"
    if "forecast_cache" not in st.session_state:
        st.session_state.forecast_cache = {}

    if cache_key not in st.session_state.forecast_cache:
        with st.spinner(f"Training {model_choice} model for {selected_label}..."):
            if model_choice == "XGBoost":
                _, forecast, mae, rmse, mape = train_and_forecast_segment(selected_series, horizon=horizon)
            elif model_choice == "SARIMA":
                _, forecast, mae, rmse, mape = train_sarima_forecast(selected_series, horizon=horizon)
            else:  # Prophet
                _, forecast, mae, rmse, mape = train_prophet_forecast(selected_series, horizon=horizon)
            
            if forecast is None:
                st.error(f"Failed to train {model_choice} model. Insufficient data or model error.")
                return
            
            st.session_state.forecast_cache[cache_key] = {
                "forecast": forecast,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "history": selected_series,
                "model": model_choice,
            }

    cached = st.session_state.forecast_cache[cache_key]
    history = cached["history"]
    forecast = cached["forecast"]
    mae = cached["mae"]
    rmse = cached["rmse"]
    mape = cached["mape"]

    forecast_df = pd.DataFrame({"Date": forecast.index, "Forecasted Sales": forecast.values})
    history_df = pd.DataFrame({"Date": history.index, "Actual Sales": history.values})
    plot_df = history_df.merge(forecast_df, on="Date", how="outer")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["Date"], y=history_df["Actual Sales"], mode="lines+markers", name="Historical Sales", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecasted Sales"], mode="lines+markers", name=f"{model_choice} Forecast", line=dict(color="#dc2626", dash="dot")))
    fig.update_layout(title=f"{model_choice} Historical vs Forecasted Sales — {selected_label}", template="plotly_white", xaxis_title="Date", yaxis_title="Sales", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("MAPE", f"{mape:,.2f}%")

    forecast_table = forecast_df.copy()
    forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m-%d")
    forecast_table.columns = ["Date", "Forecasted Sales"]
    st.subheader("Forecast Table")
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    csv_data = forecast_table.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast as CSV", csv_data, file_name=f"forecast_{selected_label.lower().replace(' ', '_')}.csv", mime="text/csv")


def render_anomaly_report(df):
    show_page_header("Anomaly Report", "Weekly sales anomalies detected using the notebook's Isolation Forest and Z-Score approach.")

    weekly_sales_df, combined_anomalies = prepare_anomaly_data(df)

    anomaly_count = len(combined_anomalies)
    highest_spike = combined_anomalies["Sales"].max() if not combined_anomalies.empty else 0
    largest_drop = combined_anomalies["Sales"].min() if not combined_anomalies.empty else 0
    avg_weekly_sales = weekly_sales_df["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total anomalies detected", anomaly_count)
    c2.metric("Highest sales spike", f"${highest_spike:,.0f}")
    c3.metric("Largest sales drop", f"${largest_drop:,.0f}")
    c4.metric("Average weekly sales", f"${avg_weekly_sales:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_sales_df["Date"], y=weekly_sales_df["Sales"], mode="lines", name="Weekly Sales", line=dict(color="#2563eb")))
    if not combined_anomalies.empty:
        fig.add_trace(go.Scatter(x=combined_anomalies["Date"], y=combined_anomalies["Sales"], mode="markers", name="Anomaly", marker=dict(color="#dc2626", size=10, symbol="x")))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", template="plotly_white", xaxis_title="Date", yaxis_title="Sales", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    method_fig = go.Figure()
    method_fig.add_trace(go.Scatter(x=weekly_sales_df["Date"], y=weekly_sales_df["Sales"], mode="lines", name="Weekly Sales", line=dict(color="#4b5563")))
    if not weekly_sales_df.empty:
        isolation_points = weekly_sales_df[weekly_sales_df["isolation_flag"] == 1]
        zscore_points = weekly_sales_df[weekly_sales_df["anomaly_zscore"] == 1]
        method_fig.add_trace(go.Scatter(x=isolation_points["Date"], y=isolation_points["Sales"], mode="markers", name="Isolation Forest", marker=dict(color="#ef4444", size=8, symbol="circle")))
        method_fig.add_trace(go.Scatter(x=zscore_points["Date"], y=zscore_points["Sales"], mode="markers", name="Z-Score", marker=dict(color="#10b981", size=8, symbol="diamond")))
    method_fig.update_layout(title="Anomaly Detection Methods Comparison", template="plotly_white", xaxis_title="Date", yaxis_title="Sales", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(method_fig, use_container_width=True)

    anomaly_table = combined_anomalies[["Date", "Sales", "Detection_Method"]].copy()
    anomaly_table.columns = ["Date", "Sales", "Detection Method"]
    anomaly_table["Date"] = anomaly_table["Date"].dt.strftime("%Y-%m-%d")
    anomaly_table = anomaly_table.sort_values("Date")
    st.subheader("Anomaly Details")
    st.dataframe(anomaly_table, use_container_width=True, hide_index=True)


def render_demand_segments(df):
    show_page_header("Product Demand Segments", "Demand-based product clusters and recommended replenishment actions.")

    clustering_df, pca_df, _ = prepare_clustering_data(df)
    numeric_features = ["Total_Sales_Volume", "Average_Order_Value", "Sales_Volatility", "Sales_Growth_Rate"]
    cluster_summary = clustering_df.groupby("Cluster_Label")[numeric_features].mean().reset_index()
    cluster_summary["Number of Products"] = clustering_df.groupby("Cluster_Label").size().values
    cluster_summary["Average Sales"] = cluster_summary["Total_Sales_Volume"] / cluster_summary["Number of Products"]
    cluster_summary = cluster_summary.rename(columns={"Total_Sales_Volume": "Average Sales Volume"})

    stock_strategy_map = {
        "Niche & Stable: Low Volume, Low Volatility, Moderate Growth": "Maintain lean, predictable inventory with frequent review and just-in-time replenishment.",
        "Premium & Growing: High Value, High Volatility, High Growth": "Keep higher safety stock and strengthen supplier flexibility for rapid replenishment.",
        "Core & Mature: High Volume, Moderate Volatility, Low Growth": "Use stable planning cycles and efficient inventory controls to protect core revenue.",
        "Emerging Growth: Moderate Volume, Moderate Volatility, High Growth": "Increase stock cautiously while monitoring demand signals and promotional effect.",
    }

    cluster_summary["Recommended Stocking Strategy"] = cluster_summary["Cluster_Label"].map(stock_strategy_map)

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster_Label",
        hover_name=pca_df.index,
        title="Product Sub-Category Clusters (PCA)",
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")
    display_summary = cluster_summary[[
        "Cluster_Label",
        "Number of Products",
        "Average Sales",
        "Sales_Growth_Rate",
        "Recommended Stocking Strategy",
    ]].copy()
    display_summary = display_summary.rename(columns={"Cluster_Label": "Cluster Name", "Sales_Growth_Rate": "Growth Rate"})
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.subheader("Sub-Category Details")
    for _, row in cluster_summary.iterrows():
        cluster_name = row["Cluster_Label"]
        members = clustering_df[clustering_df["Cluster_Label"] == cluster_name].index.tolist()
        with st.expander(f"{cluster_name} ({len(members)} products)"):
            cluster_products = pd.DataFrame(
                {
                    "Sub-Category": members,
                    "Sales Volume": clustering_df.loc[members, "Total_Sales_Volume"],
                    "Average Order Value": clustering_df.loc[members, "Average_Order_Value"],
                    "Growth Rate": clustering_df.loc[members, "Sales_Growth_Rate"],
                }
            )
            cluster_products = cluster_products.sort_values("Sales Volume", ascending=False)
            st.dataframe(cluster_products, use_container_width=True, hide_index=True)
            st.caption(stock_strategy_map[cluster_name])


def render_data_quality(df):
    show_page_header("Data Quality Report", "Comprehensive data assessment and statistical analysis")
    
    report = get_data_quality_report(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Records", report['Total Records'])
    col2.metric("❌ Missing Values", f"{report['Missing Values']} ({report['Missing %']}%)")
    col3.metric("🔢 Numeric Columns", report['Numeric Columns'])
    col4.metric("📝 Categorical Columns", report['Categorical Columns'])
    
    st.subheader("Sales Distribution Statistics")
    stat_cols = st.columns(6)
    stat_cols[0].metric("Mean", f"${report['Sales Mean']:,.0f}")
    stat_cols[1].metric("Std Dev", f"${report['Sales Std Dev']:,.0f}")
    stat_cols[2].metric("Skewness", round(report['Sales Skewness'], 2))
    stat_cols[3].metric("Kurtosis", round(report['Sales Kurtosis'], 2))
    stat_cols[4].metric("Min", f"${report['Sales Min']:,.0f}")
    stat_cols[5].metric("Max", f"${report['Sales Max']:,.0f}")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['Sales'], nbinsx=50, name='Sales Distribution', marker_color='#6366f1'))
    fig.update_layout(title="Sales Distribution Histogram", template="plotly_white", xaxis_title="Sales Value", yaxis_title="Frequency", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison(df):
    show_page_header("Model Comparison", "Side-by-side comparison of XGBoost, SARIMA, and Prophet models")
    
    forecast_type = st.selectbox("Comparison Type", ["Overall Sales", "Category", "Region"])
    horizon = st.slider("Forecast Horizon", 1, 3, 3)
    
    if forecast_type == "Overall Sales":
        series = df.set_index("Order Date")["Sales"].resample("ME").sum()
        label = "Overall Sales"
    elif forecast_type == "Category":
        categories = sorted(df["Category"].dropna().unique().tolist())
        selected = st.selectbox("Select Category", categories)
        series = df[df["Category"] == selected].set_index("Order Date")["Sales"].resample("ME").sum()
        label = selected
    else:
        regions = sorted(df["Region"].dropna().unique().tolist())
        selected = st.selectbox("Select Region", regions)
        series = df[df["Region"] == selected].set_index("Order Date")["Sales"].resample("ME").sum()
        label = selected
    
    with st.spinner("Training all 3 models for comparison..."):
        _, xgb_forecast, xgb_mae, xgb_rmse, xgb_mape = train_and_forecast_segment(series, horizon=horizon)
        _, sarima_forecast, sarima_mae, sarima_rmse, sarima_mape = train_sarima_forecast(series, horizon=horizon)
        _, prophet_forecast, prophet_mae, prophet_rmse, prophet_mape = train_prophet_forecast(series, horizon=horizon)
    
    best_model, scores = get_model_recommendation(xgb_mae or float('inf'), sarima_mae or float('inf'), prophet_mae or float('inf'))
    
    st.markdown(f"### 🏆 Recommended Model: **{best_model}**")
    score_cols = st.columns(3)
    score_cols[0].metric("XGBoost Score", f"{scores['XGBoost']}%")
    score_cols[1].metric("SARIMA Score", f"{scores['SARIMA']}%")
    score_cols[2].metric("Prophet Score", f"{scores['Prophet']}%")
    
    st.subheader("Accuracy Metrics Comparison")
    metrics_df = pd.DataFrame({
        'Model': ['XGBoost', 'SARIMA', 'Prophet'],
        'MAE': [xgb_mae or 0, sarima_mae or 0, prophet_mae or 0],
        'RMSE': [xgb_rmse or 0, sarima_rmse or 0, prophet_rmse or 0],
        'MAPE': [xgb_mape or 0, sarima_mape or 0, prophet_mape or 0]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.subheader("Forecast Comparison Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name='Historical', line=dict(color='#2563eb', width=2)))
    if xgb_forecast is not None:
        fig.add_trace(go.Scatter(x=xgb_forecast.index, y=xgb_forecast.values, mode='lines+markers', name='XGBoost', line=dict(color='#6366f1', dash='dot')))
    if sarima_forecast is not None:
        fig.add_trace(go.Scatter(x=sarima_forecast.index, y=sarima_forecast.values, mode='lines+markers', name='SARIMA', line=dict(color='#dc2626', dash='dot')))
    if prophet_forecast is not None:
        fig.add_trace(go.Scatter(x=prophet_forecast.index, y=prophet_forecast.values, mode='lines+markers', name='Prophet', line=dict(color='#16a34a', dash='dot')))
    fig.update_layout(title=f"3-Model Forecast Comparison — {label}", template="plotly_white", xaxis_title="Date", yaxis_title="Sales", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_advanced_analytics(df):
    show_page_header("Advanced Analytics", "Seasonal decomposition, trends, and detailed insights")
    
    analysis_type = st.selectbox("Analysis Type", ["Seasonal Decomposition", "Feature Importance", "Forecast Sensitivity"])
    
    if analysis_type == "Seasonal Decomposition":
        monthly_sales = df.set_index("Order Date")["Sales"].resample("ME").sum()
        
        if len(monthly_sales) >= 24:
            decomposition = get_seasonal_decomposition(monthly_sales)
            if decomposition:
                from plotly.subplots import make_subplots
                fig = make_subplots(rows=4, cols=1, subplot_titles=("Trend", "Seasonal", "Residual", "Original"))
                fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend'), row=1, col=1)
                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal', line=dict(color='#dc2626')), row=2, col=1)
                fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual', line=dict(color='#16a34a')), row=3, col=1)
                fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales.values, name='Original', line=dict(color='#2563eb')), row=4, col=1)
                fig.update_layout(height=800, template="plotly_white", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 24 months of data for seasonal decomposition")
    
    elif analysis_type == "Feature Importance":
        monthly_sales = df.set_index("Order Date")["Sales"].resample("ME").sum()
        df_features = prepare_feature_frame(monthly_sales)
        
        if not df_features.empty:
            X = df_features.drop(columns=["Sales"])
            y = df_features["Sales"]
            
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            importance_df = get_xgboost_feature_importance(model, X.columns)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='XGBoost Feature Importance', color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Forecast Sensitivity Analysis: Adjusting key parameters to see forecast impact")
        monthly_sales = df.set_index("Order Date")["Sales"].resample("ME").sum()
        
        col1, col2 = st.columns(2)
        with col1:
            trend_adjustment = st.slider("Trend Multiplier", 0.5, 1.5, 1.0, 0.1)
        with col2:
            volatility_adjustment = st.slider("Volatility Multiplier", 0.5, 1.5, 1.0, 0.1)
        
        _, forecast, _, _, _ = train_and_forecast_segment(monthly_sales, horizon=3)
        
        if forecast is not None:
            adjusted_forecast = forecast * trend_adjustment
            volatility = monthly_sales.std()
            upper_band = adjusted_forecast + (volatility * volatility_adjustment)
            lower_band = adjusted_forecast - (volatility * volatility_adjustment)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales.values, name='Historical', line=dict(color='#2563eb')))
            fig.add_trace(go.Scatter(x=forecast.index, y=adjusted_forecast.values, name='Adjusted Forecast', line=dict(color='#dc2626', dash='dot')))
            fig.add_trace(go.Scatter(x=forecast.index, y=upper_band.values, fill=None, mode='lines', line_color='rgba(0,0,0,0)', name='Upper Band'))
            fig.add_trace(go.Scatter(x=forecast.index, y=lower_band.values, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Lower Band', fillcolor='rgba(99, 102, 241, 0.2)'))
            fig.update_layout(title="Sensitivity Analysis", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


def main():
    df = load_data()

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">Sales Forecasting Dashboard</div>
        <div class="sidebar-subtitle">End-to-end forecasting and demand intelligence</div>
        <div class="nav-section-title">📍 Navigation</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Sales Overview",
            "🔮 Forecast Explorer",
            "⚠️  Anomaly Report",
            "📈 Product Demand Segments",
            "📋 Data Quality",
            "🤖 Model Comparison",
            "📊 Advanced Analytics"
        ],
        label_visibility="collapsed"
    )
    st.sidebar.divider()
    st.sidebar.caption("🎓 Internship Project")

    if "Sales Overview" in page:
        render_sales_overview(df)
    elif "Forecast Explorer" in page:
        render_forecast_explorer(df)
    elif "Anomaly Report" in page:
        render_anomaly_report(df)
    elif "Product Demand" in page:
        render_demand_segments(df)
    elif "Data Quality" in page:
        render_data_quality(df)
    elif "Model Comparison" in page:
        render_model_comparison(df)
    elif "Advanced Analytics" in page:
        render_advanced_analytics(df)


if __name__ == "__main__":
    main()
