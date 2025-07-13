# 📈 StockSki - Advanced Stock Predictor

A comprehensive, AI-powered stock prediction and analysis application built with Streamlit, featuring multiple machine learning models, technical analysis, and sentiment analysis.

## 🚀 Features

### 🤖 AI & Machine Learning Models
- **Prophet Model**: Facebook's time series forecasting model
- **Random Forest Regressor**: Ensemble learning for price prediction
- **Gradient Boosting Regressor**: Advanced boosting algorithm
- **Linear Regression**: Traditional statistical model
- **Support Vector Regression (SVR)**: Non-linear regression with kernel methods
- **Model Comparison**: Automatic performance comparison and best model selection

### 📊 Technical Analysis
- **Moving Averages**: 5, 20, 50, and 200-day moving averages
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicators
- **Stochastic Oscillator**: Momentum indicator
- **ADX (Average Directional Index)**: Trend strength indicator
- **ATR (Average True Range)**: Volatility measure
- **Volume Analysis**: Volume moving averages and ratios

### 📰 News & Sentiment Analysis
- **Real-time News**: Latest stock-related news from financial sources
- **VADER Sentiment Analysis**: Automated sentiment scoring
- **News Impact Assessment**: Correlation between news sentiment and stock performance

### 📈 Performance Metrics
- **Return Analysis**: Total return, annualized return, volatility
- **Risk Metrics**: Maximum drawdown, Value at Risk (VaR), Conditional VaR
- **Sharpe Ratio**: Risk-adjusted return measure
- **Technical Signals**: Buy/sell signals based on technical indicators

### 🎨 Modern UI/UX
- **Responsive Design**: Works on desktop and mobile devices
- **Tabbed Interface**: Organized sections for different analyses
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Real-time Updates**: Live data loading and analysis
- **Professional Styling**: Modern gradient cards and clean layout

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd stockski
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Data Sources
- **Alpha Vantage API**: Built-in API key provides real-time stock data (500 requests/day)
- **Yahoo Finance**: Fallback data source
- **Finnhub**: Optional additional data source (requires free API key)

## 📋 Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **YFinance**: Yahoo Finance data API

### Machine Learning
- **Scikit-learn**: Machine learning algorithms
- **Prophet**: Time series forecasting
- **TA-Lib**: Technical analysis library

### Visualization
- **Plotly**: Interactive plotting library
- **Matplotlib**: Static plotting (if needed)

### Data Processing
- **BeautifulSoup**: Web scraping for news
- **Requests**: HTTP library for API calls
- **VADER Sentiment**: Sentiment analysis

## 🎯 How to Use

### 1. Stock Selection
- Choose from a curated list of popular stocks
- Select your desired date range for analysis
- Choose prediction period (1-5 years)

### 2. Data Analysis
- **Overview Tab**: Key metrics, price charts, and descriptive statistics
- **Predictions Tab**: AI model forecasts and performance comparison
- **Technical Analysis Tab**: Advanced technical indicators and signals
- **News & Sentiment Tab**: Latest news with sentiment analysis
- **Performance Metrics Tab**: Risk and return analysis
- **Data Export Tab**: Download data and forecasts

### 3. Model Configuration
- Enable/disable specific models (Prophet, ML models)
- Adjust technical analysis parameters
- Customize prediction periods

## 🔧 Technical Improvements

### Code Quality
- **Modular Design**: Separated functions for different features
- **Error Handling**: Comprehensive error handling and user feedback
- **Caching**: Optimized data loading with Streamlit caching
- **Code Documentation**: Detailed docstrings and comments

### Performance
- **Efficient Data Processing**: Optimized pandas operations
- **Smart Caching**: 1-hour cache for stock data
- **Parallel Processing**: Where applicable for model training
- **Memory Management**: Efficient data structures and cleanup

### User Experience
- **Loading States**: Progress indicators for long operations
- **Error Messages**: Clear, actionable error messages
- **Responsive Layout**: Adapts to different screen sizes
- **Intuitive Navigation**: Tabbed interface for easy exploration

## 📊 Model Performance

### Prophet Model
- **Strengths**: Handles seasonality and trends well
- **Best For**: Long-term forecasting with clear patterns
- **Limitations**: May struggle with sudden market changes

### Machine Learning Models
- **Random Forest**: Good for capturing non-linear relationships
- **Gradient Boosting**: Excellent for complex patterns
- **Linear Regression**: Baseline model for comparison
- **SVR**: Good for non-linear relationships with kernel methods

### Model Selection
The app automatically compares all models and highlights the best performer based on:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared Score (R²)

## 🚨 Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions. Always:

- Do your own research
- Consult with financial advisors
- Consider multiple data sources
- Understand the risks involved in stock investing
- Never invest more than you can afford to lose

## 🔮 Future Enhancements

### Planned Features
- **Portfolio Analysis**: Multi-stock portfolio optimization
- **Options Analysis**: Options pricing and strategies
- **Crypto Support**: Cryptocurrency analysis
- **Real-time Alerts**: Price and signal notifications
- **Advanced ML**: Deep learning models (LSTM, GRU)
- **Economic Indicators**: Integration with economic data
- **Social Sentiment**: Twitter and Reddit sentiment analysis

### Technical Improvements
- **API Integration**: Real-time data APIs
- **Database**: Persistent storage for historical data
- **Cloud Deployment**: AWS/Azure deployment options
- **Mobile App**: Native mobile application
- **API Endpoints**: REST API for external integrations

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Nish Sura**
- GitHub: [@nishsura](https://github.com/nishsura)
- LinkedIn: [Nish Sura](https://linkedin.com/in/nishsura)

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free stock data
- **Facebook Prophet**: For the time series forecasting model
- **TA-Lib**: For technical analysis indicators
- **Streamlit**: For the amazing web app framework
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning algorithms

---

**Made with ❤️ for the financial analysis community**

