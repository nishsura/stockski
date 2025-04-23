import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor")
st.write("Developed by Nish Sura")

stocks = (
    "AAPL", "GOOG", "MSFT", "NVDA", "TSLA", "AMZN", "META", "NFLX", "BRK.A", "BRK.B",
    "JNJ", "V", "PG", "JPM", "UNH", "DIS", "MA", "NVDA", "PYPL", "HD", "VZ", "CMCSA",
    "INTC", "ADBE", "PFE", "NFLX", "KO", "PEP", "CSCO", "T", "ABT", "MRK", "XOM",
    "ORCL", "CRM", "NKE", "WMT", "LLY", "MCD", "MMM", "GE", "IBM", "TXN", "QCOM",
    "BA", "MDT", "HON", "AMGN", "COST", "TMO", "DHR", "UNP", "CVX", "CAT", "SPGI",
    "AXP", "USB", "NEE", "SBUX", "LMT", "MMM", "LOW", "GS", "PLD", "ISRG", "SYK",
    "BDX", "CI", "DUK", "EMR", "ETN", "FIS", "GD", "ITW", "MET", "PNC", "SO", "TRV",
    "WM", "AON", "CCI", "CHTR", "CME", "CTAS", "D", "DG", "ECL", "FDX", "FISV",
    "HCA", "ICE", "ILMN", "INTU", "KMB", "KR", "MMC", "MS", "NSC", "PEG", "PSA",
    "SHW", "STZ", "TGT", "TROW", "WBA", "ZTS", "AAP", "AES", "AIG", "AMT", "AIZ",
    "AMP", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB", "AVY", "AWK",
    "BAX", "BXP", "BWA", "BXP", "CDNS", "CERN", "CF", "CMS", "COP", "CPRT", "CSX",
    "CTSH", "CTL", "CTXS", "DOV", "DTE", "DVA", "DXC", "EA", "EIX", "EW", "EXPD",
    "FAST", "FE", "FLT", "FTI", "GILD", "GL", "HIG", "HII", "HPE", "HPQ", "HSY",
    "HST", "IP", "IR", "IQV", "J", "JBHT", "JCI", "JKHY", "JNPR", "K", "KEY", "KIM",
    "KLAC", "KMX", "KO", "KSS", "L", "LDOS", "LEN", "LIN", "LKQ", "LLY", "LNT",
    "LUV", "LYB", "MGM", "MHK", "MKC", "MLM", "MMC", "MOS", "MSCI", "MTD", "MUR",
    "MXIM", "NDAQ", "NI", "NLSN", "NOC", "NRG", "NUE", "NVR", "ODFL", "OKE", "OMC",
    "ORLY", "OTIS", "PAYX", "PAYC", "PBCT", "PCAR", "PFG", "PKG", "PKI", "PPL",
    "PRU", "PWR", "PXD", "QRVO", "RCL", "RE", "REG", "RF", "RHI", "RJF", "RMD",
    "ROK", "ROL", "ROP", "RSG", "RTX", "SBAC", "SJM", "SLB", "SNA", "SWK", "SYY",
    "TDG", "TEL", "TFC", "TJX", "TMO", "TROW", "TSCO", "TT", "TYL", "UHS", "ULTA",
    "UNH", "UNM", "UPS", "URI", "VFC", "VLO", "VMC", "VRSK", "VTR", "VZ", "WAB",
    "WEC", "WELL", "WMB", "WMT", "WRK", "WY", "XEL", "XLNX", "XYL", "YUM", "ZBH",
    "ZBRA", "ZION", "ZTS", "AAL", "ALK", "ALLE", "AME", "AMP", "ANTM", "AON", "AOS",
    "APA", "APD", "APH", "ARE", "ATO", "AVB", "AVY", "AWK", "BAX", "BXP", "BWA",
    "BXP", "CDNS", "CERN", "CF", "CMS", "COP", "CPRT", "CSX", "CTSH", "CTL", "CTXS",
    "DOV", "DTE", "DVA", "DXC", "EA", "EIX", "EW", "EXPD", "FAST", "FE", "FLT",
    "FTI", "GILD", "GL", "HIG", "HII", "HPE", "HPQ", "HSY", "HST", "IP", "IR",
    "IQV", "J", "JBHT", "JCI", "JKHY", "JNPR", "K", "KEY", "KIM", "KLAC", "KMX",
    "KO", "KSS", "L", "LDOS", "LEN", "LIN", "LKQ", "LLY", "LNT", "LUV", "LYB",
    "MGM", "MHK", "MKC", "MLM", "MMC", "MOS", "MSCI", "MTD", "MUR", "MXIM", "NDAQ",
    "NI", "NLSN", "NOC", "NRG", "NUE", "NVR", "ODFL", "OKE", "OMC", "ORLY", "OTIS",
    "PAYX", "PAYC", "PBCT", "PCAR", "PFG", "PKG", "PKI", "PPL", "PRU", "PWR", "PXD",
    "QRVO", "RCL", "RE", "REG", "RF", "RHI", "RJF", "RMD", "ROK", "ROL", "ROP",
    "RSG", "RTX", "SBAC", "SJM", "SLB", "SNA", "SWK", "SYY", "TDG", "TEL", "TFC",
    "TJX", "TMO", "TROW", "TSCO", "TT", "TYL", "UHS", "ULTA", "UNH", "UNM", "UPS",
    "URI", "VFC", "VLO", "VMC", "VRSK", "VTR", "VZ", "WAB", "WEC", "WELL", "WMB",
    "WMT", "WRK", "WY", "XEL", "XLNX", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
    "AAL", "ALK", "ALLE", "AME", "AMP", "ANTM", "AON", "AOS", "APA", "APD", "APH",
    "ARE", "ATO", "AVB", "AVY", "AWK", "BAX", "BXP", "BWA", "BXP", "CDNS", "CERN",
    "CF", "CMS", "COP", "CPRT", "CSX", "CTSH", "CTL", "CTXS", "DOV", "DTE", "DVA",
    "DXC", "EA", "EIX", "EW", "EXPD", "FAST", "FE", "FLT", "FTI", "GILD", "GL",
    "HIG", "HII", "HPE", "HPQ", "HSY", "HST", "IP", "IR", "IQV", "J", "JBHT", "JCI",
    "JKHY", "JNPR", "K", "KEY", "KIM", "KLAC", "KMX", "KO", "KSS", "L", "LDOS",
    "LEN", "LIN", "LKQ", "LLY", "LNT", "LUV", "LYB", "MGM", "MHK", "MKC", "MLM",
    "MMC", "MOS", "MSCI", "MTD", "MUR", "MXIM", "NDAQ", "NI", "NLSN", "NOC", "NRG",
    "NUE", "NVR", "ODFL", "OKE", "OMC", "ORLY", "OTIS", "PAYX", "PAYC", "PBCT",
    "PCAR", "PFG", "PKG", "PKI", "PPL", "PRU", "PWR", "PXD", "QRVO", "RCL", "RE",
    "REG", "RF", "RHI", "RJF", "RMD", "ROK", "ROL", "ROP", "RSG", "RTX", "SBAC",
    "SJM", "SLB", "SNA", "SWK", "SYY", "TDG", "TEL", "TFC", "TJX", "TMO", "TROW",
    "TSCO", "TT", "TYL", "UHS", "ULTA", "UNH", "UNM", "UPS", "URI", "VFC", "VLO",
    "VMC", "VRSK", "VTR", "VZ", "WAB", "WEC", "WELL", "WMB", "WMT", "WRK", "WY",
    "XEL", "XLNX", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS", "AAL", "ALK", "ALLE",
    "AME", "AMP", "ANTM", "AON", "AOS", "APA", "APD", "APH", "ARE", "ATO", "AVB",
    "AVY", "AWK", "BAX", "BXP", "BWA", "BXP", "CDNS", "CERN", "CF", "CMS", "COP",
    "CPRT", "CSX", "CTSH", "CTL", "CTXS", "DOV", "DTE", "DVA", "DXC", "EA", "EIX",
    "EW", "EXPD", "FAST", "FE", "FLT", "FTI", "GILD", "GL", "HIG", "HII", "HPE",
    "HPQ", "HSY", "HST", "IP", "IR", "IQV", "J", "JBHT", "JCI", "JKHY", "JNPR", "K",
    "KEY", "KIM", "KLAC", "KMX", "KO", "KSS", "L", "LDOS", "LEN", "LIN", "LKQ",
    "LLY", "LNT", "LUV", "LYB", "MGM", "MHK", "MKC", "MLM", "MMC", "MOS", "MSCI",
    "MTD", "MUR", "MXIM", "NDAQ", "NI", "NLSN", "NOC", "NRG", "NUE", "NVR", "ODFL",
    "OKE", "OMC", "ORLY", "OTIS", "PAYX", "PAYC", "PBCT", "PCAR", "PFG", "PKG",
    "PKI", "PPL", "PRU", "PWR", "PXD", "QRVO", "RCL", "RE", "REG", "RF", "RHI",
    "RJF", "RMD", "ROK", "ROL", "ROP", "RSG", "RTX", "SBAC", "SJM", "SLB", "SNA",
    "SWK", "SYY", "TDG", "TEL", "TFC", "TJX", "TMO", "TROW", "TSCO", "TT", "TYL",
    "UHS", "ULTA", "UNH", "UNM", "UPS", "URI", "VFC", "VLO", "VMC", "VRSK", "VTR",
    "VZ", "WAB", "WEC", "WELL", "WMB", "WMT", "WRK", "WY", "XEL", "XLNX", "XYL",
    "YUM", "ZBH", "ZBRA", "ZION", "ZTS"
)


st.sidebar.header('User Input Features')
selected_stock = st.sidebar.selectbox("Choose Stock to Predict", stocks)
start_date = st.sidebar.date_input("Start date", date(2015, 1, 1))
end_date = st.sidebar.date_input("End date", date.today())
n_years = st.sidebar.slider("Years of prediction: ", 1, 5)
period = n_years * 365

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data")
data = load_data(selected_stock, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
data_load_state.text("Data Loaded")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Descriptive Statistics
st.subheader('Descriptive Statistics')
st.write(data.describe())

# Moving Averages
def plot_moving_averages(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], name='MA20'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name='MA50'))
    fig.update_layout(title_text='Moving Averages', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_moving_averages(data)

# Prophet Forecasting
df_train = data[['Date', 'Close']].dropna()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

if df_train.shape[0] < 2:
    st.error("Not enough data to train Prophet model. Try a different stock or date range.")
else:
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.write(f'Forecast Plot for {n_years} Years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write(f'Forecast Plot for {n_years} Years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Machine Learning Model - RandomForestRegressor
st.subheader('RandomForestRegressor Prediction')

# Prepare data for ML model
data_ml = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data_ml['Date'] = pd.to_datetime(data_ml['Date'])
data_ml['Year'] = data_ml['Date'].dt.year
data_ml['Month'] = data_ml['Date'].dt.month
data_ml['Day'] = data_ml['Date'].dt.day
data_ml['DayOfWeek'] = data_ml['Date'].dt.dayofweek

# Features and target
X = data_ml[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek']]
y = data_ml['Close']

# Train-test split
train_size = int(0.8 * len(data_ml))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred)
mse_rf = mean_squared_error(y_test, y_pred)
rmse_rf = np.sqrt(mse_rf)



# Future predictions using RandomForestRegressor
last_date = data_ml['Date'].iloc[-1]
future_dates = pd.date_range(last_date, periods=period)

future_data = pd.DataFrame({
    'Date': future_dates,
    'Year': future_dates.year,
    'Month': future_dates.month,
    'Day': future_dates.day,
    'DayOfWeek': future_dates.dayofweek,
    'Open': data_ml['Open'].iloc[-1],
    'High': data_ml['High'].iloc[-1],
    'Low': data_ml['Low'].iloc[-1],
    'Volume': data_ml['Volume'].iloc[-1]
})

future_X = future_data[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek']]
future_data['Predicted_Close'] = rf_model.predict(future_X)

# Plot future predictions
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=future_data['Date'], y=future_data['Predicted_Close'], name='Predicted Close'))
fig3.update_layout(title_text='RandomForestRegressor Future Predictions', xaxis_rangeslider_visible=True)
st.plotly_chart(fig3)



# Add Performance Metrics
st.subheader('Model Performance Metrics')
actual = df_train['y'].values
predicted = forecast['yhat'][:len(actual)].values

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = mean_squared_error(actual, predicted, squared=False)

st.write(f'Prophet Model - MAE: {mae:.2f}')
st.write(f'Prophet Model - MSE: {mse:.2f}')
st.write(f'Prophet Model - RMSE: {rmse:.2f}')

st.write(f'RandomForestRegressor - MAE: {mae_rf:.2f}')
st.write(f'RandomForestRegressor - MSE: {mse_rf:.2f}')
st.write(f'RandomForestRegressor - RMSE: {rmse_rf:.2f}')


# Candlestick Chart
st.subheader('Candlestick Chart')
def plot_candlestick():
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_candlestick()

# Technical Indicators: Bollinger Bands
st.subheader('Bollinger Bands')
def plot_bollinger_bands(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['BB_upper'] = data['MA20'] + 2*data['Close'].rolling(window=20).std()
    data['BB_lower'] = data['MA20'] - 2*data['Close'].rolling(window=20).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_upper'], name='BB_upper', line=dict(color='rgba(255,0,0,0.2)')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_lower'], name='BB_lower', line=dict(color='rgba(255,0,0,0.2)')))
    fig.update_layout(title_text='Bollinger Bands', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_bollinger_bands(data)

# Download data
st.subheader('Download Data')
st.download_button('Download Raw Data', data.to_csv(), file_name='raw_data.csv')
st.download_button('Download Forecast Data', forecast.to_csv(), file_name='forecast_data.csv')
