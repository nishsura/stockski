import requests
import time
from datetime import datetime

ticker = "AAPL"
end = int(time.time())
start = end - (365 * 24 * 60 * 60) # 1 year ago

url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print(f"Requesting {url}...")
response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("Success! Got JSON.")
    # print(data['chart']['result'][0]['indicators']['quote'][0].keys())
else:
    print(f"Failed: {response.text[:200]}")
