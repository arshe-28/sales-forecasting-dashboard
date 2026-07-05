import os

# Read the current app.py
with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Find the position to insert (before def main():)
insert_pos = content.find('\ndef main():')

if insert_pos == -1:
    print("ERROR: Could not find 'def main():' in file")
else:
    new_functions = '''
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

'''
    
    # Insert the new functions before main()
    new_content = content[:insert_pos] + new_functions + content[insert_pos:]
    
    # Write back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ New render functions added successfully!")
