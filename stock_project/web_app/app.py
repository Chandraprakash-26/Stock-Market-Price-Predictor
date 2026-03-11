from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime

app = Flask(__name__)


stock_dict = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "MARUTI.NS": "Maruti Suzuki",
    "HCLTECH.NS": "HCL Technologies",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ITC.NS": "ITC",
    "SBIN.NS": "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "TATAMOTORS.NS": "Tata Motors",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KO": "Coca-Cola",
    "TITAN.NS": "Titan Company",
    "AXISBANK.NS": "Axis Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "NESTLEIND.NS": "Nestle India",
    "INDIGO.NS": "InterGlobe Aviation",
    "BHEL.NS": "Bharat Heavy Electricals",
    "POLYCAB.NS": "Polycab India",
    "IRCTC.NS": "Indian Railway Catering and Tourism Corporation",
    "MRF.NS": "MRF",
    "HAVELLS.NS": "Havells India",
    "HONDA.NS": "Honda",
    "005930.KS": "Samsung Electronics",
    "NVDA": "NVIDIA",
    "FB": "Meta Platforms",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "PG": "Procter & Gamble",
    "DIS": "Walt Disney",
    "MA": "Mastercard",
    "PFE": "Pfizer",
    "NFLX": "Netflix",
    "ADBE": "Adobe",
    "PYPL": "PayPal",
    "CSCO": "Cisco Systems",
    "PEP": "PepsiCo",
    "INTC": "Intel",
    "MRNA": "Moderna",
    "AMD": "Advanced Micro Devices",
    "CRM": "Salesforce"
}

sorted_stock_dict = dict(sorted(stock_dict.items(), key=lambda item: item[1]))

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365*2)
    df = stock.history(start=start_date, end=end_date)
    return df

def prepare_data(df):
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df = df[['Days', 'Close']].dropna() 
    return df

def train_model(data):
    X = data['Days'].values.reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_price(model, days):
    prediction = model.predict(np.array([days]).reshape(-1, 1))
    return prediction[0]

def generate_recommendation(predicted_price, last_close_price, threshold=0.01):
    if predicted_price > last_close_price * (1 + threshold):
        return "Buy"
    elif predicted_price < last_close_price:
        return "Sell"
    else:
        return "Hold"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol')
        custom_ticker = request.form.get('custom_ticker')

        if custom_ticker:
            stock_symbol = custom_ticker

        if not stock_symbol:
            return render_template('index.html', error="Please select or enter a stock symbol", stock_dict=sorted_stock_dict)

        data = fetch_data(stock_symbol)
        if data.empty:
            return render_template('index.html', error="Failed to fetch data. Please try again.", stock_dict=sorted_stock_dict)

        prepared_data = prepare_data(data)
        if prepared_data.empty:
            return render_template('index.html', error="No valid data available. Please try another stock.", stock_dict=sorted_stock_dict)

        model = train_model(prepared_data)
        last_date = prepared_data['Days'].max() + 1
        last_close_price = prepared_data['Close'].iloc[-1]
        predicted_price = predict_price(model, last_date)
        predicted_price_formatted = f"₹{predicted_price:.2f}"
        last_close_price_formatted = f"₹{last_close_price:.2f}"
        recommendation = generate_recommendation(predicted_price, last_close_price)
        
        return render_template('index.html', stock_dict=sorted_stock_dict, stock_symbol=stock_symbol, 
                               last_close_price=last_close_price_formatted, predicted_price=predicted_price_formatted, 
                               recommendation=recommendation)
    return render_template('index.html', stock_dict=sorted_stock_dict)

if __name__ == '__main__':
    app.run(debug=True)
