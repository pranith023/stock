# 📊 Stock Analyzer

A comprehensive stock analysis application built using Streamlit, `yfinance`, and `prophet`. This tool allows users to analyze individual stocks, screen multiple stocks, track a portfolio, and even forecast stock prices.

## ✨ Features

* **Dashboard**: Get a quick overview of a stock with its current price, market cap, P/E ratio, company information, and interactive historical and intraday charts.
* **Stock Screener**: Screen multiple stocks at once by entering a list of symbols to see key metrics like price, market cap, and sector.
* **Portfolio Tracker**: Track your investment portfolio by adding stocks with their buy price and quantity, and see your total profit or loss in real-time.
* **Forecasting**: Predict a stock's price for the next 30 days using the powerful `prophet` library.

## 🚀 Live Demo

You can view a live demo of this application here:

> **[https://stocksanalyzer.streamlit.app/]**

## ⚙️ Installation

To run this application, you need to have Python installed. You can install the required packages using `pip`.

1.  Clone this repository or download the `main.py` file.
2.  Install the dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

Once the dependencies are installed, you can run the Streamlit application from your terminal:

```bash
streamlit run main.py