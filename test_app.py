#!/usr/bin/env python3
"""
Test script for StockSki - Advanced Stock Predictor
This script tests the main functionality without running the Streamlit app
"""

import sys
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import app functions
try:
    from app import (
        load_stock_data, calculate_technical_indicators, 
        prepare_ml_features, train_ml_models, create_prophet_forecast
    )
    print("✅ Successfully imported app functions")
except ImportError as e:
    print(f"❌ Error importing app functions: {e}")
    sys.exit(1)

def test_data_loading():
    """Test stock data loading functionality"""
    print("\n🧪 Testing Data Loading...")
    
    try:
        # Test with a popular stock
        ticker = "AAPL"
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        data, error = load_stock_data(ticker, start_date, end_date)
        
        if error:
            print(f"❌ Data loading failed: {error}")
            return False
        
        if data is None or data.empty:
            print("❌ No data returned")
            return False
        
        print(f"✅ Successfully loaded {len(data)} rows of data for {ticker}")
        print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"   Columns: {list(data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\n🧪 Testing Technical Indicators...")
    
    try:
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        
        # Calculate technical indicators
        tech_data = calculate_technical_indicators(sample_data)
        
        # Check if indicators were calculated
        expected_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'MA20', 'MA50']
        missing_indicators = [ind for ind in expected_indicators if ind not in tech_data.columns]
        
        if missing_indicators:
            print(f"❌ Missing indicators: {missing_indicators}")
            return False
        
        print(f"✅ Successfully calculated technical indicators")
        print(f"   Available indicators: {[col for col in tech_data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def test_ml_features():
    """Test ML feature preparation"""
    print("\n🧪 Testing ML Feature Preparation...")
    
    try:
        # Create sample data with technical indicators
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100),
            'RSI': np.random.uniform(0, 100, 100),
            'MACD': np.random.uniform(-10, 10, 100),
            'MA20': np.random.uniform(100, 200, 100),
            'MA50': np.random.uniform(100, 200, 100)
        })
        
        # Prepare features
        X, y, error = prepare_ml_features(sample_data)
        
        if error:
            print(f"❌ Feature preparation failed: {error}")
            return False
        
        if X is None or y is None:
            print("❌ No features or target returned")
            return False
        
        print(f"✅ Successfully prepared ML features")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Feature columns: {list(X.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML feature preparation test failed: {e}")
        return False

def test_ml_models():
    """Test ML model training"""
    print("\n🧪 Testing ML Model Training...")
    
    try:
        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100),
            'RSI': np.random.uniform(0, 100, 100),
            'MACD': np.random.uniform(-10, 10, 100),
            'MA20': np.random.uniform(100, 200, 100),
            'MA50': np.random.uniform(100, 200, 100),
            'Year': np.random.randint(2020, 2024, 100),
            'Month': np.random.randint(1, 13, 100),
            'Day': np.random.randint(1, 29, 100),
            'DayOfWeek': np.random.randint(0, 7, 100)
        })
        y = np.random.uniform(100, 200, 100)
        
        # Train models
        results = train_ml_models(X, y)
        
        if not results:
            print("❌ No models were trained successfully")
            return False
        
        print(f"✅ Successfully trained {len(results)} models")
        for model_name, result in results.items():
            print(f"   {model_name}: RMSE = {result['rmse']:.4f}, R² = {result['r2']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML model training test failed: {e}")
        return False

def test_prophet_forecast():
    """Test Prophet forecasting"""
    print("\n🧪 Testing Prophet Forecasting...")
    
    try:
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Close': np.random.uniform(100, 200, 100) + np.sin(np.arange(100) * 0.1) * 20  # Add some trend
        })
        
        # Create forecast
        forecast, error = create_prophet_forecast(sample_data, periods=30)
        
        if error:
            print(f"❌ Prophet forecast failed: {error}")
            return False
        
        if forecast is None:
            print("❌ No forecast returned")
            return False
        
        print(f"✅ Successfully created Prophet forecast")
        print(f"   Forecast periods: {len(forecast)}")
        print(f"   Forecast columns: {list(forecast.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet forecast test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting StockSki App Tests...")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_technical_indicators,
        test_ml_features,
        test_ml_models,
        test_prophet_forecast
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 