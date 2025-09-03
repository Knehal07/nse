import requests

def nse_session():
    """Create a session with headers that NSE accepts"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive"
    }
    session = requests.Session()
    session.headers.update(headers)

    # 👇 Warm up: visit NSE homepage to get cookies
    session.get("https://www.nseindia.com", timeout=5)

    return session
