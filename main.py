import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from textblob import TextBlob
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="📊 Stock Analyzer", layout="wide")

# ------------------------------
# Fetch Stock Data
# ------------------------------
@st.cache_data
def fetch_quote(symbol: str):
    ticker = yf.Ticker(symbol)
    info = dict(ticker.info) if hasattr(ticker, "info") else {}
    hist = ticker.history(period="6mo").reset_index()
    intraday = ticker.history(period="1d", interval="5m").reset_index()
    return info, hist, intraday


# ------------------------------
# Sentiment Analysis
# ------------------------------
def analyze_sentiment(text):
    if not text:
        return "Neutral"
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


# ------------------------------
# Stock Screener
# ------------------------------
@st.cache_data
def stock_screener(symbols):
    results = []
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            results.append({
                "Symbol": sym,
                "Price": info.get("currentPrice", None),
                "Market Cap": info.get("marketCap", None),
                "PE Ratio": info.get("trailingPE", None),
                "Sector": info.get("sector", None),
            })
        except Exception:
            continue
    return pd.DataFrame(results)


# ------------------------------
# Portfolio Tracker
# ------------------------------
def portfolio_summary(portfolio):
    df = []
    for sym, qty, buy_price in portfolio:
        try:
            info = yf.Ticker(sym).info
            current_price = info.get("currentPrice", 0)
            pnl = (current_price - buy_price) * qty
            df.append({
                "Symbol": sym,
                "Quantity": qty,
                "Buy Price": buy_price,
                "Current Price": current_price,
                "P/L": pnl
            })
        except Exception:
            continue
    return pd.DataFrame(df)


# ------------------------------
# Forecasting with Prophet
# ------------------------------
# ------------------------------
# Forecasting with Prophet
# ------------------------------
@st.cache_data
def forecast_stock(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="2y").reset_index()
    df = hist[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df['ds'] = df['ds'].dt.tz_localize(None) # Add this line to remove timezone

    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast


# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Stock Screener", "Portfolio Tracker", "Forecasting"])

# ------------------------------
# Dashboard Page
# ------------------------------
if page == "Dashboard":
    st.title("📊 Stock Analyzer Dashboard")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")

    if st.button("Analyze"):
        try:
            info, hist, intraday = fetch_quote(symbol)

            # Stock Info
            st.subheader(f"{info.get('longName', symbol)} ({symbol})")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            col2.metric("Market Cap", f"${info.get('marketCap', 'N/A')}")
            col3.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")

            st.markdown("---")

            # Historical Chart
            st.subheader("📈 6-Month Price History")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist["Date"],
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="Candlestick",
            ))
            fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Intraday Chart
            st.subheader("⏱ Intraday (5-min interval)")
            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(
                x=intraday["Datetime"],
                y=intraday["Close"],
                mode="lines+markers",
                name="Price",
            ))
            line_fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(line_fig, use_container_width=True)

            # Company Info
            st.subheader("🏢 Company Overview")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
            st.write(f"**Description:** {info.get('longBusinessSummary', 'N/A')[:500]}...")

        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")


# ------------------------------
# Stock Screener Page
# ------------------------------
elif page == "Stock Screener":
    st.title("🔎 Stock Screener")
    symbols = st.text_area("Enter symbols separated by commas (e.g., AAPL, TSLA, MSFT)",
                           "AAPL, TSLA, MSFT, AMZN, GOOGL").split(",")

    if st.button("Run Screener"):
        df = stock_screener([s.strip() for s in symbols])
        st.dataframe(df)


# ------------------------------
# Portfolio Tracker Page
# ------------------------------
elif page == "Portfolio Tracker":
    st.title("💼 Portfolio Tracker")

    portfolio = []
    st.sidebar.subheader("Add Stock to Portfolio")
    sym = st.sidebar.text_input("Symbol", "AAPL")
    qty = st.sidebar.number_input("Quantity", 1, step=1)
    buy_price = st.sidebar.number_input("Buy Price", 0.0, step=0.1)
    if st.sidebar.button("Add"):
        portfolio.append((sym, qty, buy_price))

    if portfolio:
        df = portfolio_summary(portfolio)
        st.dataframe(df)
        st.metric("Total P/L", f"${df['P/L'].sum():.2f}")


# ------------------------------
# Forecasting Page
# ------------------------------
elif page == "Forecasting":
    st.title("📈 Stock Price Forecasting (30 days)")
    symbol = st.text_input("Enter Stock Symbol for Forecasting", "AAPL")

    if st.button("Forecast"):
        try:
            model, forecast = forecast_stock(symbol)
            st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

            st.subheader("Forecast Plot")
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Forecasting error: {e}")
