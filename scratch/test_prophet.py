import pandas as pd
from prophet import Prophet
import numpy as np

# Create dummy data
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'y': np.random.randn(100).cumsum()
})

import cmdstanpy
import os

# Set the correct path
correct_path = '/Users/nishsura/.cmdstan/cmdstan-2.38.0'
cmdstanpy.set_cmdstan_path(correct_path)

# Monkeypatch to prevent Prophet from overwriting it
original_set_path = cmdstanpy.set_cmdstan_path
def mocked_set_path(path):
    # Only allow setting if it's the correct path
    if path != correct_path:
        return
    original_set_path(path)

cmdstanpy.set_cmdstan_path = mocked_set_path

try:
    print("Initializing Prophet...")
    model = Prophet(stan_backend='CMDSTANPY')
    print("Fitting model...")
    model.fit(df)
    print("Fit successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
