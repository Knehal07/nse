📊 NSE Data Fetcher (Python)

Fetch real-time stock, index, and option chain data from the NSE India website using Python.

🚀 Features

✅ Fetch stock quote data (e.g., RELIANCE, TCS, INFY)

✅ Fetch index data (e.g., NIFTY 50, BANKNIFTY)

✅ Fetch option chain data (indices & stocks)

✅ Save results to CSV/JSON for analysis

✅ Ready-to-use Jupyter notebook demo

📂 Project Structure nse-data-fetcher/ │── README.md │── requirements.txt │── .gitignore │── src/ │ ├── utils.py # Session + headers for NSE API │ ├── nse_stocks.py # Fetch stock data │ ├── nse_indices.py # Fetch index data │ ├── nse_options.py # Fetch option chain │── notebooks/ │ ├── demo.ipynb # Example usage │── data/ │ ├── reliance.csv │ ├── nifty50.csv │ ├── nifty_options.csv

🔧 Installation

Clone the repo and install dependencies:

git clone https://github.com//nse-data-fetcher.git cd nse-data-fetcher pip install -r requirements.txt

📌 Usage

Fetch Stock Data python src/nse_stocks.py

Fetch Index Data python src/nse_indices.py

Fetch Option Chain python src/nse_options.py

Jupyter Notebook Demo jupyter notebook notebooks/demo.ipynb

📝 Requirements

Python 3.8+

Libraries:

requests

pandas

jupyter (optional, for notebooks)

Install with:

pip install -r requirements.txt

⚠️ Notes

NSE APIs sometimes block frequent requests → add delays if needed.

Always warm up the session by hitting https://www.nseindia.com before API calls.

Data is for educational and research purposes only (not for commercial trading).
