# 📈 StockLens - Advanced Stock Intelligence Platform

StockLens is a state-of-the-art, AI-powered financial analysis and prediction platform. Built for investors who demand more than just static charts, StockLens leverages advanced machine learning models, comprehensive technical indicators, and real-time sentiment analysis to provide a 360-degree view of market dynamics.

---

## 🚀 Key Capabilities

### 🤖 Intelligent Forecasting
*   **Prophet-Driven Predictions**: Utilizing Meta's Prophet model with **logistic growth constraints** to ensure realistic, non-negative price forecasts even in volatile markets.
*   **Multi-Model ML Engine**: Parallel evaluation of Random Forest, Gradient Boosting, Linear Regression, and SVR models.
*   **Automated Model Selection**: Intelligent ranking of models based on RMSE, MAE, and R² scores to surface the most reliable forecast.

### 📊 Professional Technical Suite
*   **Trend Identification**: Multiple Moving Averages (5, 20, 50, 200-day) and ADX.
*   **Momentum & Volatility**: RSI, MACD, Bollinger Bands, and Stochastic Oscillators.
*   **Risk Assessment**: Advanced metrics including Value at Risk (VaR), Conditional VaR (CVaR), and Maximum Drawdown.
*   **Volume Dynamics**: Comprehensive volume-weighted analysis and moving averages.

### 📰 Sentiment & Alternative Data
*   **Real-time Financial News**: Aggregated news feeds for selected tickers.
*   **VADER Sentiment Scoring**: Automated NLP-based sentiment analysis to quantify market mood.
*   **Correlative Analysis**: Visualization of news sentiment trends alongside price movement.

### 🎨 Enterprise-Grade UI/UX
*   **Interactive Visualization**: Fully interactive Plotly charts with custom themes.
*   **Tabbed Analytical Framework**: Clean separation of Overview, Predictions, Technicals, News, and Risk metrics.
*   **Real-time Data Sync**: Seamless data fetching with multi-source fallback logic.

---

## 🛠️ Technical Architecture

### Resilient Data Ingestion
StockLens features a robust multi-tier data retrieval system:
1.  **Primary**: `yfinance` & direct Yahoo Finance API integration.
2.  **Secondary Fallback**: Alpha Vantage API (with automatic handling of rate limits and premium restrictions).
3.  **Buffer Management**: Intelligent data buffering to ensure technical indicators (like 200-day MA) are accurate from the very start of the requested period.

### Optimized Prediction Backend
*   **Non-Negative Forecasting**: Implements logistic growth floors to prevent mathematical artifacts like negative stock prices.
*   **Stan Backend Management**: Optimized CmdStan integration for high-performance Prophet calculations.
*   **Memory Efficient**: Designed to handle large historical datasets without performance degradation.

---

## 💻 Installation & Deployment

### Prerequisites
*   Python 3.10+
*   pip (or Poetry/Conda)

### Quick Start
1.  **Clone the Platform**:
    ```bash
    git clone <repository-url>
    cd stocklens
    ```

2.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch**:
    ```bash
    streamlit run app.py
    ```

---

## 🎯 Platform Navigation

*   **📈 Overview**: High-level metrics, real-time price action, and historical volatility.
*   **🤖 Predictions**: Deep-dive into AI forecasts with confidence intervals and model performance comparisons.
*   **📊 Technical Analysis**: Granular view of trend, momentum, and volatility indicators.
*   **📰 News & Sentiment**: Qualitative analysis powered by real-time NLP.
*   **📋 Performance Metrics**: Institutional-grade risk and return reporting.
*   **💾 Data Export**: Export analyzed datasets and forecasts for further research.

---

## 🚨 Disclaimer
StockLens is an analytical tool provided for educational and research purposes only. The financial markets involve significant risk. No prediction model is 100% accurate, and past performance is not indicative of future results. **Always consult with a certified financial advisor before making investment decisions.**

---

## 👨‍💻 Development & Support
Developed and maintained by **Nish Sura**.

*   **GitHub**: [@nishsura](https://github.com/nishsura)
*   **LinkedIn**: [Nish Sura](https://linkedin.com/in/nishsura)

---
**Crafted with precision for the modern investor.**
