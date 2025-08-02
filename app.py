import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import os
from sklearn.metrics import mean_absolute_error
from prophet.make_holidays import get_holiday_names

# --- 1. Configuration & Data Loading ---
st.set_page_config(layout="wide", page_title="Energy Forecasting Dashboard")
st.title("Energy Consumption Forecasting Dashboard")
st.write("This dashboard forecasts future energy usage based on historical data.")

# File path to the simulated data
file_path = "maize_mill_simulated_sensor_data.csv"

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: File not found at {file_path}. Please check the path.")
        st.stop()
    df = pd.read_csv(file_path, parse_dates=["Timestamp"])
    df = df.rename(columns={'Timestamp': 'ds', 'Power_kW': 'y'})
    return df

# Load the data
df = load_data(file_path)

# --- NEW: Updated holiday data setup for South Africa ---
@st.cache_data
def create_holidays():
    holidays_df = pd.DataFrame({
        'holiday': 'SA_Holiday',
        'ds': pd.to_datetime([
            '2025-01-01', '2025-03-21', '2025-04-18', '2025-04-21',
            '2025-04-27', '2025-04-28', '2025-05-01', '2025-06-16',
            '2025-08-09', '2025-09-24', '2025-12-16', '2025-12-25',
            '2025-12-26'
        ]),
        'lower_window': -1,
        'upper_window': 0
    })
    return holidays_df

holidays = create_holidays()

# --- 2. User Input ---
st.sidebar.header("Forecast Settings")

n_days_to_forecast = st.sidebar.slider(
    "Select number of days to forecast:",
    min_value=1,
    max_value=30,
    value=7
)

changepoint_prior_scale = st.sidebar.slider(
    "Model Flexibility:",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    help="Higher values allow the model to be more flexible to capture peaks and valleys."
)

include_holidays = st.sidebar.checkbox("Account for Holidays", value=True)

# --- 3. Forecasting Logic ---
@st.cache_data
def run_forecast(df, n_days, cps, include_holidays):
    train_df = df.iloc[:-n_days]
    test_df = df.iloc[-n_days:]

    if include_holidays:
        m = Prophet(
            changepoint_prior_scale=cps,
            seasonality_mode='multiplicative',
            holidays=holidays
        )
    else:
        m = Prophet(
            changepoint_prior_scale=cps,
            seasonality_mode='multiplicative'
        )
    
    m.fit(train_df)

    future = m.make_future_dataframe(periods=n_days, freq='D')
    forecast = m.predict(future)

    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
    forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(0, x))
    forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(0, x))
    
    test_forecast = forecast.iloc[-n_days:]
    mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])

    return forecast, mae, test_df

if st.sidebar.button("Run Forecast"):
    with st.spinner("Generating forecast..."):
        forecast, mae, test_df = run_forecast(df, n_days_to_forecast, changepoint_prior_scale, include_holidays)

    # NEW: Move the notice to below the button
    st.sidebar.info("💡 This dashboard is a simulation demo to showcase the potential of our forecasting service. The data is not live.")
    
    # --- 4. Dashboard Visualization ---
    st.header("Forecasted Energy Consumption")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[df['ds'] < test_df.iloc[0]['ds']]['ds'],
        y=df[df['ds'] < test_df.iloc[0]['ds']]['y'],
        mode='lines',
        name='Historical Power (kW)',
        line=dict(color='royalblue')
    ))

    fig.add_trace(go.Scatter(
        x=test_df['ds'],
        y=test_df['y'],
        mode='lines',
        name='Test Data (Actuals)',
        line=dict(color='darkgreen')
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecasted Power (kW)',
        line=dict(color='firebrick')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title='Historical and Forecasted Power Consumption',
        xaxis_title='Date',
        yaxis_title='Power (kW)',
        height=600,
        legend_x=0, legend_y=1
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. Key Metrics ---
    st.header("Forecast Summary")
    
    last_historical_day_value = df.iloc[-1]['y']
    first_forecast_day_value = forecast.iloc[0]['yhat']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Recorded Power", f"{last_historical_day_value:.2f} kW")
    col2.metric(
        f"Average Forecast for Next {n_days_to_forecast} Days",
        f"{forecast['yhat'].tail(n_days_to_forecast).mean():.2f} kW"
    )
    col3.metric(
        "Forecasted Change",
        f"{first_forecast_day_value - last_historical_day_value:.2f} kW"
    )
    col4.metric(
        "Model Accuracy (MAE)",
        f"{mae:.2f} kW",
        help="Mean Absolute Error on the test data."
    )
    
    st.markdown("---")
    st.header("What to Do with Your Predictions")
    st.markdown("""
    This dashboard is more than just a chart; it's a tool for real-life decisions. Here's how this service provides insights to help you manage your entire facility and generate value:
    
    - **Holistic Forecasts for Budgeting:** Our service connects to multiple machines and your main power meter to provide a **total energy consumption forecast** for your entire facility. You can use the **Average Forecast** to predict your total energy costs for the next few weeks, helping you plan your budget with confidence.
    - **Optimal Work Scheduling:** By identifying the predicted peaks and valleys in your facility's power usage, you can optimize your operations. Schedule heavy machinery use during times of lower consumption or plan maintenance during dips to avoid impacting peak production.
    - **Proven Efficiency Improvements:** Use our model's predictions as a baseline. After you make a change, such as installing new equipment or optimizing workflows, you can see if your actual usage drops below the prediction, providing **concrete proof that your changes are working.**
    - **Proactive Problem Detection:** Our service helps you move from reactive to proactive maintenance. We can set up an alert for when a specific machine's actual usage goes outside the **Confidence Zone**. This provides an **early warning** that a machine may be malfunctioning, allowing you to address issues before they cause costly downtime.
    """)