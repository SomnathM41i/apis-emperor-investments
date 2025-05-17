from flask import Flask, request, jsonify
import yfinance as yf
from flask_cors import CORS
import joblib
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the pre-trained model at startup
model = joblib.load('multi_stock_prediction_model_5y.pkl')

def get_next_trading_day(current_date):
    """Calculate the next trading day, skipping weekends."""
    next_date = current_date + timedelta(days=1)
    while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_date += timedelta(days=1)
    return next_date

@app.route('/stock', methods=['GET'])
def get_stock_data():
    symbols = request.args.get('symbol')
    if not symbols:
        return jsonify({"error": "Please provide at least one stock symbol"}), 400

    symbols_list = symbols.split(',')
    response_data = {}

    for symbol in symbols_list:
        try:
            symbol = symbol.strip().upper()
            stock = yf.Ticker(symbol)
            
            if not stock.info or stock.info.get('symbol') is None:
                response_data[symbol] = {"error": "Invalid stock symbol"}
                continue

            data = stock.history(period="5d")  # Fetch 5 days for historical trend
            latest_data = data.iloc[-1] if not data.empty else None

            response_data[symbol] = {
                "company_info": {
                    "name": stock.info.get('longName'),
                    "sector": stock.info.get('sector'),
                    "industry": stock.info.get('industry'),
                    "website": stock.info.get('website'),
                    "description": stock.info.get('longBusinessSummary'),
                    "country": stock.info.get('country'),
                    "employees": stock.info.get('fullTimeEmployees'),
                    "market_cap": stock.info.get('marketCap'),
                },
                "latest_trading_day": {
                    "date": str(latest_data.name.date()) if latest_data is not None else None,
                    "open": float(latest_data["Open"]) if latest_data is not None else None,
                    "high": float(latest_data["High"]) if latest_data is not None else None,
                    "low": float(latest_data["Low"]) if latest_data is not None else None,
                    "close": float(latest_data["Close"]) if latest_data is not None else None,
                    "volume": int(latest_data["Volume"]) if latest_data is not None else None,
                },
                "historical_data": [
                    {
                        "date": str(row.name.date()),
                        "close": float(row["Close"])
                    } for index, row in data.iterrows()
                ],
                "financials": {
                    "pe_ratio": stock.info.get('trailingPE'),
                    "dividend_yield": stock.info.get('dividendYield'),
                    "52_week_high": stock.info.get('fiftyTwoWeekHigh'),
                    "52_week_low": stock.info.get('fiftyTwoWeekLow'),
                    "average_volume": stock.info.get('averageVolume'),
                }
            }
        except Exception as e:
            response_data[symbol] = {"error": str(e)}

    return jsonify(response_data)

@app.route('/predict', methods=['GET'])
def predict_stock():
    symbols = request.args.get('symbol')
    if not symbols:
        return jsonify({"error": "Please provide at least one stock symbol"}), 400

    symbols_list = symbols.split(',')
    response_data = {}

    for symbol in symbols_list:
        try:
            symbol = symbol.strip().upper()
            stock = yf.Ticker(symbol)
            
            data = stock.history(period="1d")
            if data.empty:
                response_data[symbol] = {"error": "No data available for this symbol"}
                continue

            latest_data = data.iloc[-1]

            latest_ohlcv = [
                float(latest_data["Open"]),
                float(latest_data["High"]),
                float(latest_data["Low"]),
                float(latest_data["Close"]),
                float(latest_data["Volume"])
            ]

            predicted_ohlcv = model.predict([latest_ohlcv])[0]

            latest_date = latest_data.name.date()
            next_date = get_next_trading_day(latest_date)

            response_data[symbol] = {
                "latest_trading_day": {
                    "date": str(latest_date),
                    "open": float(latest_data["Open"]),
                    "high": float(latest_data["High"]),
                    "low": float(latest_data["Low"]),
                    "close": float(latest_data["Close"]),
                    "volume": int(latest_data["Volume"])
                },
                "prediction": {
                    "next_trading_day": {
                        "date": str(next_date),
                        "open": float(predicted_ohlcv[0]),
                        "high": float(predicted_ohlcv[1]),
                        "low": float(predicted_ohlcv[2]),
                        "close": float(predicted_ohlcv[3]),
                        "volume": int(predicted_ohlcv[4])
                    }
                }
            }
        except Exception as e:
            response_data[symbol] = {"error": str(e)}

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)