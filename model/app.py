import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("seaborn-v0_8-whitegrid")

app = Flask(__name__)

# Load the LSTM model (make sure your model is in the correct path)
model = load_model('stock_dl_model.h5')


# ── Technical Indicators ─────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def add_features(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df['Daily_Return'] = df['Close'].pct_change()
    df['RSI'] = compute_rsi(df['Close'])

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    rolling20 = df['Close'].rolling(window=20)
    df['BB_High'] = rolling20.mean() + 2 * rolling20.std()
    df['BB_Low'] = rolling20.mean() - 2 * rolling20.std()

    df['Volatility_5'] = df['Daily_Return'].rolling(5).std()
    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Spread'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].pct_change()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Replace any inf/-inf (e.g. from pct_change when previous value was 0) with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'POWERGRID.NS'  # Default stock if none is entered

        # Define the start and end dates for stock data
        start = dt.datetime(2012, 1, 1)
        end = dt.datetime(2026, 6, 1)

        # Download stock data
        df_raw = yf.download(stock, start=start, end=end)
        if df_raw.empty:
            return render_template('index.html', error="Invalid Stock Symbol")

        df_raw.reset_index(inplace=True)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [c[0] if c[1] == '' else c[0] for c in df_raw.columns]

        # Keep a clean copy for LSTM section (Close prices over full history)
        df_close = df_raw[['Date', 'Close']].copy()

        # Feature engineered dataframe (for EDA + RF)
        df = add_features(df_raw.copy())

        # Descriptive Data
        data_desc = df_raw.describe()

        # ── EMA SERIES (on full df_close, like original) ───────────────────
        ema20 = df_close['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df_close['Close'].ewm(span=50, adjust=False).mean()
        ema100 = df_close['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df_close['Close'].ewm(span=200, adjust=False).mean()

        # =====================================================================
        # PLOT 1: Closing Price vs Time with 20 & 50 Days EMA
        # =====================================================================
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df_close['Date'], df_close['Close'], color='#f2c94c', linewidth=1.2, label='Closing Price')
        ax1.plot(df_close['Date'], ema20, color='#27ae60', linewidth=1.4, label='EMA 20')
        ax1.plot(df_close['Date'], ema50, color='#e74c3c', linewidth=1.4, label='EMA 50')
        ax1.set_title(f"{stock} — Closing Price vs Time (20 & 50 Days EMA)", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.tick_params(axis='x', rotation=30)
        fig1.tight_layout()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path, dpi=150)
        plt.close(fig1)

        # =====================================================================
        # PLOT 2: Closing Price vs Time with 100 & 200 Days EMA
        # =====================================================================
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df_close['Date'], df_close['Close'], color='#f2c94c', linewidth=1.2, label='Closing Price')
        ax2.plot(df_close['Date'], ema100, color='#27ae60', linewidth=1.4, label='EMA 100')
        ax2.plot(df_close['Date'], ema200, color='#e74c3c', linewidth=1.4, label='EMA 200')
        ax2.set_title(f"{stock} — Closing Price vs Time (100 & 200 Days EMA)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.tick_params(axis='x', rotation=30)
        fig2.tight_layout()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200, dpi=150)
        plt.close(fig2)

        # =====================================================================
        # PLOT 3: Combined EDA — Moving Avg, Returns Dist, RSI, Correlation
        # =====================================================================
        fig_eda = plt.figure(figsize=(16, 11))
        gs = gridspec.GridSpec(2, 2, figure=fig_eda, hspace=0.35, wspace=0.3)

        ax_a = fig_eda.add_subplot(gs[0, 0])
        ax_a.plot(df['Date'], df['Close'], color='royalblue', linewidth=1.2, label='Close Price')
        ax_a.plot(df['Date'], df['SMA_10'], color='orange', linewidth=1, label='SMA 10')
        ax_a.plot(df['Date'], df['SMA_20'], color='green', linewidth=1, label='SMA 20')
        ax_a.set_title(f'{stock} Closing Price with Moving Averages', fontsize=11, fontweight='bold')
        ax_a.set_xlabel('Date')
        ax_a.set_ylabel('Price')
        ax_a.legend(fontsize=8)
        ax_a.tick_params(axis='x', rotation=30)

        ax_b = fig_eda.add_subplot(gs[0, 1])
        ax_b.hist(df['Daily_Return'], bins=60, color='mediumpurple', edgecolor='white', alpha=0.85)
        ax_b.set_title('Distribution of Daily Returns', fontsize=11, fontweight='bold')
        ax_b.set_xlabel('Daily Return')
        ax_b.set_ylabel('Count')

        ax_c = fig_eda.add_subplot(gs[1, 0])
        ax_c.plot(df['Date'], df['RSI'], color='red', linewidth=0.8)
        ax_c.axhline(70, color='gray', linestyle='--', linewidth=1, label='Overbought (70)')
        ax_c.axhline(30, color='gray', linestyle=':', linewidth=1, label='Oversold (30)')
        ax_c.set_title('Relative Strength Index (RSI)', fontsize=11, fontweight='bold')
        ax_c.set_xlabel('Date')
        ax_c.set_ylabel('RSI')
        ax_c.legend(fontsize=8)
        ax_c.tick_params(axis='x', rotation=30)

        ax_d = fig_eda.add_subplot(gs[1, 1])
        corr_cols = ['Close', 'Volume', 'Daily_Return', 'SMA_10', 'SMA_20', 'EMA_10', 'RSI', 'MACD', 'Volatility_5']
        corr = df[corr_cols].corr().round(2)
        sns.heatmap(corr, ax=ax_d, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, linewidths=0.5, annot_kws={'size': 6})
        ax_d.set_title('Feature Correlation Heatmap', fontsize=11, fontweight='bold')
        ax_d.tick_params(axis='x', rotation=45)

        fig_eda.suptitle(f"{stock} — Exploratory Data Analysis", fontsize=15, fontweight='bold', y=0.995)
        eda_chart_path = "static/eda_overview.png"
        fig_eda.savefig(eda_chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig_eda)

        # =====================================================================
        # RANDOM FOREST CLASSIFIER — Direction Prediction
        # =====================================================================
        feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Daily_Return',
                         'SMA_10', 'SMA_20', 'EMA_10', 'RSI', 'MACD', 'MACD_Signal',
                         'BB_High', 'BB_Low', 'Volatility_5', 'Price_Change',
                         'High_Low_Spread', 'Volume_Change']

        X = df[feature_cols].values
        y = df['Target'].values

        X_train, X_test, y_train, y_test_rf = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False)

        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        acc = accuracy_score(y_test_rf, y_pred_rf)
        report = classification_report(y_test_rf, y_pred_rf, target_names=['DOWN (0)', 'UP (1)'])

        # ── Confusion Matrix ─────────────────────────────────────────────
        cm = confusion_matrix(y_test_rf, y_pred_rf)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['0', '1'], yticklabels=['0', '1'])
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title(f'{stock} — Confusion Matrix (RF Direction Model)', fontsize=12, fontweight='bold')
        fig_cm.tight_layout()
        cm_chart_path = "static/confusion_matrix.png"
        fig_cm.savefig(cm_chart_path, dpi=150)
        plt.close(fig_cm)

        # ── Feature Importance ───────────────────────────────────────────
        importances = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 7))
        palette = sns.color_palette('viridis', len(importances))
        ax_fi.barh(importances['Feature'], importances['Importance'], color=palette[::-1])
        ax_fi.set_xlabel('Importance Score', fontsize=11)
        ax_fi.set_ylabel('Feature', fontsize=11)
        ax_fi.set_title(f'{stock} — Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
        ax_fi.invert_yaxis()
        fig_fi.tight_layout()
        fi_chart_path = "static/feature_importance.png"
        fig_fi.savefig(fi_chart_path, dpi=150)
        plt.close(fig_fi)

        # ── Next-day direction prediction ───────────────────────────────
        latest_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
        next_pred = rf_model.predict(latest_features)[0]
        next_proba = rf_model.predict_proba(latest_features)[0]
        next_direction = "UP" if next_pred == 1 else "DOWN"
        prob_down = round(next_proba[0] * 100, 2)
        prob_up = round(next_proba[1] * 100, 2)

        # =====================================================================
        # LSTM PRICE PREDICTION
        # =====================================================================
        data_training = pd.DataFrame(df_close['Close'][0:int(len(df_close) * 0.70)])
        data_testing = pd.DataFrame(df_close['Close'][int(len(df_close) * 0.70):])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = model.predict(x_test)

        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # =====================================================================
        # PLOT: Prediction vs Original Trend (LSTM)
        # =====================================================================
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, color='#27ae60', linewidth=1.2, label="Original Price")
        ax3.plot(y_predicted, color='#e74c3c', linewidth=1.2, label="Predicted Price")
        ax3.set_title(f"{stock} — LSTM Prediction vs Original Trend", fontsize=13, fontweight='bold')
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        fig3.tight_layout()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path, dpi=150)
        plt.close(fig3)

        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df_raw.to_csv(csv_file_path, index=False)

        # Return the rendered template with charts and dataset
        return render_template('index.html',
                               stock=stock,
                               plot_path_ema_20_50=ema_chart_path,
                               plot_path_ema_100_200=ema_chart_path_100_200,
                               plot_path_eda=eda_chart_path,
                               plot_path_confusion=cm_chart_path,
                               plot_path_feature_importance=fi_chart_path,
                               plot_path_prediction=prediction_chart_path,
                               data_desc=data_desc.to_html(classes='table table-bordered table-sm'),
                               dataset_link=csv_file_path,
                               rf_accuracy=f"{acc * 100:.2f}%",
                               rf_report=report,
                               next_direction=next_direction,
                               prob_up=prob_up,
                               prob_down=prob_down)

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)