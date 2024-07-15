from flask import Flask, render_template, request, redirect, url_for, session
from flask_bcrypt import Bcrypt
import pandas as pd
import numpy as np
import base64
from flask import send_from_directory
from io import BytesIO
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)

# Dummy user data for login
users = {'user': bcrypt.generate_password_hash('password').decode('utf-8')}

#ALPHA_VANTAGE_API_KEY = 'your_alpha_vantage_api_key'

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    error = None
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        frequency = request.form['frequency']
        model_type = request.form['model']

        # valid stock symbols
        valid_symbols = ['MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN','AAPL', 'IBM', 'QCOM', 'ADBE', 'INTC', 'RYAAY', 'JNJ', 'SHEL',
                        'BRK.A', 'KO', 'TM', 'CSCO', 'MCD', 'TSLA', 'DIS', 'META', 'UPS','ECL', 'MSFT', 'VZ', 'GEO', 'NVDA', 'WFC', 'GOOGL', 'ORCL', 'WMT','HSBC']



        if stock_symbol not in valid_symbols:
            error = 'Invalid stock symbol!'
        else:
            return redirect(url_for('predict', stock_symbol=stock_symbol, frequency=frequency, model_type=model_type))

    return render_template('index.html', error=error)

@app.route('/predict', methods=['GET'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    stock_symbol = request.args.get('stock_symbol')
    frequency = request.args.get('frequency')
    model_type = request.args.get('model_type')

    # 10 years of data
    dates = pd.date_range(start='1/1/2014', periods=3652)  # 3652 days for 10 years
    data = pd.DataFrame({'Date': dates, 'Price': np.random.randn(3652).cumsum()})
    data.set_index('Date', inplace=True)

    if frequency == 'weekly':
        data = data.resample('W').mean()
    elif frequency == 'monthly':
        data = data.resample('M').mean()

    plot_url = None
    best_params = None
    summary = None
    model_name = None

    if model_type == 'arima':
        model_name = 'ARIMA'
        model = ARIMA(data['Price'], order=(6, 3, 6))  # p=6, d=3, q=6
        model_fit = model.fit()
        best_params = model_fit.params.to_dict()
        summary = model_fit.summary().as_text()
        
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Price'], label='Actual Prices')
        plt.plot(data.index, model_fit.fittedvalues, label='ARIMA Predictions')
        plt.legend()
        plt.title(f'ARIMA Model - {stock_symbol} Prices')
    elif model_type == 'linear_regression':
        model_name = 'Linear Regression'
        data['Date_ordinal'] = data.index.map(pd.Timestamp.toordinal)
        X = data[['Date_ordinal']]
        y = data['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        best_params = {'coef': model.coef_.tolist(), 'intercept': model.intercept_}
        
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Price'], label='Actual Prices')
        plt.scatter(X_test.index, predictions, color='red', label='Predicted Prices')
        plt.legend()
        plt.title(f'Linear Regression Model - {stock_symbol} Prices')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return render_template('result.html', plot_url=plot_url, model_name=model_name, best_params=best_params, summary=summary)

import yfinance as yf

@app.route('/data', methods=['GET', 'POST'])
def data():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    error = None
    stock_data = None
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']

        # Fetch stock data from Yahoo Finance API
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="max")

        if data.empty:
            error = 'Invalid stock symbol or API error!'
        else:
            stock_data = data.reset_index().to_dict('records')

    return render_template('data.html', error=error, stock_data=stock_data)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and bcrypt.check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Invalid username or password!'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/help', methods=['GET'])
def help():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/download')
def download():
    return send_from_directory(directory='static', filename='stock_price_prediction_document.pdf', as_attachment=True)
