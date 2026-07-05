import os

# Read the current app.py
with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Find the position to insert (before def prepare_feature_frame)
insert_pos = content.find('\ndef prepare_feature_frame(series):')

if insert_pos == -1:
    print("ERROR: Could not find 'def prepare_feature_frame' in file")
else:
    helper_functions = '''
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


'''
    
    # Insert the helper functions before prepare_feature_frame
    new_content = content[:insert_pos] + helper_functions + content[insert_pos:]
    
    # Write back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Helper functions added successfully!")
