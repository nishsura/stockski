# Stock Predictor

The "Stock Predictor" is a comprehensive project developed to predict stock prices using historical data and advanced forecasting techniques. This project leverages machine learning algorithms and statistical models to provide insightful predictions and visualizations for various stocks. The tool is designed to be user-friendly, incorporating an interactive web interface built using Streamlit, a popular Python library for creating web applications.

## User Interface

The user interface is powered by Streamlit, which allows for easy interaction with the tool. The application begins with a title and a brief description of the project. Users can:
- Select a stock from a predefined list of popular stocks.
- Specify a start and end date for historical data.
- Choose the number of years for future predictions.
- Configure the forecasting period and download data via the sidebar options.

## Data Loading and Preparation

The `load_data` function fetches historical stock data from Yahoo Finance using the `yfinance` library. This function:
- Downloads the stock data for the selected ticker symbol between the specified start and end dates.
- Processes and resets the downloaded data to ensure it is in a suitable format for further analysis and modeling.

## Data Visualization and Descriptive Statistics

- **Raw Data Visualization**: The application displays the raw stock data and its descriptive statistics, providing an overview of the data, including measures such as mean, standard deviation, and quartiles.
- **Time Series Plot**: A time series plot of the stock's opening and closing prices is generated using Plotly, allowing users to visualize the historical price movements.
- **Moving Averages**: Moving averages (MA20 and MA50) are calculated and plotted to show the stock's average price over different periods, helping to identify trends and smooth out short-term fluctuations.

## Forecasting with Prophet

Prophet, a forecasting tool developed by Meta, is used to predict future stock prices.

- **Model Training**: The historical data is split into training and testing sets. The training set is used to train the Prophet model, which learns the patterns in the data.
- **Forecasting**: The model makes future price predictions for the specified period (up to 5 years). These predictions are displayed in a table and plotted to visualize the forecasted stock prices.
- **Forecast Components**: Prophet decomposes the forecast into trend, weekly, and yearly components, helping users understand the underlying patterns and seasonality.

## Machine Learning with RandomForestRegressor

A RandomForestRegressor model is employed to further enhance prediction capabilities using various features from the historical data.

- **Feature Engineering**: Additional features are created from the date, and the data is split into training and testing sets.
- **Model Training and Prediction**: The RandomForestRegressor is trained on the training set, and predictions are made on the testing set. The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- **Future Predictions**: Using the trained RandomForestRegressor, future stock prices are predicted for the specified period and plotted to visualize future trends.

## Technical Analysis

- **Candlestick Chart**: A candlestick chart is created to visualize the stock's open, high, low, and close prices over time, commonly used in technical analysis to identify price patterns and trends.
- **Bollinger Bands**: Bollinger Bands are plotted to provide a visual representation of the stock's volatility, consisting of a moving average (MA20) and two standard deviation lines (upper and lower) to help identify overbought and oversold conditions.

## Data Download and User Interaction

The application includes features for downloading the raw and forecast data in CSV format, allowing users to perform further analysis or use the data for other purposes.

## Conclusion

The "Stock Predictor" project combines several advanced techniques to provide a robust tool for stock price prediction. By leveraging Prophet for statistical forecasting and RandomForestRegressor for machine learning predictions, the tool offers a comprehensive analysis of stock price movements. The inclusion of various visualizations, such as time series plots, moving averages, candlestick charts, and Bollinger Bands, enhances the user's ability to understand and interpret the stock data. Overall, this project demonstrates the power of combining different data analysis and machine learning techniques to create a valuable tool for stock market prediction and analysis.

---

## Getting Started

To run the "Stock Predictor" application locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd stock-predictor
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Prophet](https://facebook.github.io/prophet/)
- [scikit-learn](https://scikit-learn.org/)

For any issues or contributions, please open a pull request or issue on the GitHub repository.

