import yfinance as yf
import pandas as pd
from datetime import date, timedelta

ticker = "MSFT"
start_date = date(2020, 1, 1)
end_date = date.today()
buffer_start = pd.to_datetime(start_date) - timedelta(days=365)

print(f"Downloading {ticker} from {buffer_start} to {end_date}...")
try:
    data = yf.download(ticker, start=buffer_start, end=end_date, progress=False)
    print(f"Data received: {type(data)}")
    if data is not None:
        print(f"Columns: {data.columns}")
        print(f"Shape: {data.shape}")
        if not data.empty:
            print(f"Head:\n{data.head()}")
        else:
            print("Data is EMPTY!")
    else:
        print("Data is NONE!")
except Exception as e:
    print(f"Error: {e}")
