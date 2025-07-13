"""
Configuration file for StockSki - Advanced Stock Predictor
"""

# App Configuration
APP_CONFIG = {
    'title': 'StockSki - Advanced Stock Predictor',
    'icon': '📈',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Stock Lists
POPULAR_STOCKS = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "BRK-B", 
    "JNJ", "V", "PG", "JPM", "UNH", "DIS", "MA", "PYPL", "HD", "VZ", "CMCSA",
    "INTC", "ADBE", "PFE", "KO", "PEP", "CSCO", "T", "ABT", "MRK", "XOM",
    "ORCL", "CRM", "NKE", "WMT", "LLY", "MCD", "MMM", "GE", "IBM", "TXN", 
    "QCOM", "BA", "MDT", "HON", "AMGN", "COST", "TMO", "DHR", "UNP", "CVX", 
    "CAT", "SPGI", "AXP", "USB", "NEE", "SBUX", "LMT", "LOW", "GS", "PLD", 
    "ISRG", "SYK", "BDX", "CI", "DUK", "EMR", "ETN", "FIS", "GD", "ITW", 
    "MET", "PNC", "SO", "TRV", "WM", "AON", "CCI", "CHTR", "CME", "CTAS", 
    "D", "DG", "ECL", "FDX", "FISV", "HCA", "ICE", "ILMN", "INTU", "KMB", 
    "KR", "MMC", "MS", "NSC", "PEG", "PSA", "SHW", "STZ", "TGT", "TROW", 
    "WBA", "ZTS"
]

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    'RSI': {
        'name': 'Relative Strength Index',
        'timeperiod': 14,
        'overbought': 70,
        'oversold': 30
    },
    'MACD': {
        'name': 'Moving Average Convergence Divergence',
        'fastperiod': 12,
        'slowperiod': 26,
        'signalperiod': 9
    },
    'BB': {
        'name': 'Bollinger Bands',
        'timeperiod': 20,
        'nbdevup': 2,
        'nbdevdn': 2
    },
    'STOCH': {
        'name': 'Stochastic Oscillator',
        'fastk_period': 5,
        'slowk_period': 3,
        'slowd_period': 3
    },
    'ADX': {
        'name': 'Average Directional Index',
        'timeperiod': 14
    },
    'ATR': {
        'name': 'Average True Range',
        'timeperiod': 14
    }
}

# Moving Averages Configuration
MOVING_AVERAGES = [5, 20, 50, 200]

# Model Configuration
ML_MODELS = {
    'Random Forest': {
        'class': 'RandomForestRegressor',
        'params': {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10
        }
    },
    'Gradient Boosting': {
        'class': 'GradientBoostingRegressor',
        'params': {
            'n_estimators': 100,
            'random_state': 42,
            'learning_rate': 0.1
        }
    },
    'Linear Regression': {
        'class': 'LinearRegression',
        'params': {}
    },
    'SVR': {
        'class': 'SVR',
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        }
    }
}

# Prophet Configuration
PROPHET_CONFIG = {
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'seasonality_mode': 'multiplicative',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0
}

# Cache Configuration
CACHE_CONFIG = {
    'ttl': 3600,  # 1 hour
    'max_entries': 100
}

# News Configuration
NEWS_CONFIG = {
    'max_articles': 10,
    'timeout': 10,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Performance Metrics Configuration
PERFORMANCE_CONFIG = {
    'risk_free_rate': 0.02,  # 2% annual risk-free rate
    'confidence_level': 0.95,  # 95% confidence for VaR
    'trading_days': 252  # Number of trading days per year
}

# UI Configuration
UI_CONFIG = {
    'chart_height': 500,
    'subplot_height': 800,
    'metric_columns': 4,
    'max_display_rows': 10
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'price_change_periods': [1, 5, 20],
    'volume_ma_period': 20,
    'technical_indicators': True,
    'time_features': True,
    'lag_features': False
}

# Error Messages
ERROR_MESSAGES = {
    'no_data': 'No data found for the selected stock and date range.',
    'insufficient_data': 'Not enough data for analysis. Please select a longer date range.',
    'model_error': 'Error training model: {error}',
    'data_load_error': 'Error loading data: {error}',
    'news_error': 'Could not fetch news: {error}'
}

# Success Messages
SUCCESS_MESSAGES = {
    'data_loaded': 'Data loaded successfully!',
    'model_trained': 'Models trained successfully!',
    'analysis_complete': 'Analysis completed successfully!'
}

# Disclaimer
DISCLAIMER = """
**IMPORTANT**: This application is for educational and research purposes only. 
Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions. 
Always do your own research, consult with financial advisors, and understand the risks involved in stock investing.
"""

# Color Schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'light': '#8c564b',
    'dark': '#e377c2',
    'muted': '#7f7f7f',
    'white': '#ffffff',
    'black': '#000000'
}

# Gradient Colors
GRADIENTS = {
    'primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'success': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'warning': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'info': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'
} 