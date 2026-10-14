import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from datetime import datetime

# --- GIAO DIỆN ỨNG DỤNG ---
st.set_page_config(page_title="Dự đoán Cổ phiếu LSTM", layout="wide")
st.title("📈 Công cụ Dự đoán Giá Cổ phiếu bằng Mô hình LSTM")
st.write("Ứng dụng lấy dữ liệu thời gian thực từ Yahoo Finance và dự đoán xu hướng giá đóng cửa.")

# Thanh bên (Sidebar) để người dùng nhập thông số
st.sidebar.header("Cấu hình Dự đoán")
ticker = st.sidebar.text_input("Nhập mã cổ phiếu (ví dụ: AAPL, MSFT, VNM.VN):", "AAPL").upper()
start_date = st.sidebar.date_input("Ngày bắt đầu dữ liệu quá khứ:", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc:", datetime.today())
prediction_days = st.sidebar.slider("Số ngày quá khứ dùng để dự đoán (Window Size):", 30, 90, 60)

# --- 1. TẢI DỮ LIỆU ---
@st.cache_data(show_spinner="Đang tải dữ liệu từ Yahoo Finance...")
def load_data(stock_ticker, start, end):
    data = yf.download(stock_ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu hoặc khoảng thời gian!")
else:
    st.subheader(f"📊 Dữ liệu lịch sử của mã {ticker}")
    st.write(data.tail())

    # --- 2. TIỀN XỬ LÝ DỮ LIỆU ---
    # Sử dụng giá Đóng cửa (Close) để dự đoán
    df_close = data[['Close']].dropna()
    dataset = df_close.values
    training_data_len = int(np.ceil(len(dataset) * 0.8)) # 80% train, 20% test

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Tạo tập dữ liệu huấn luyện (X_train, y_train)
    train_data = scaled_data[0:int(training_data_len), :]
    X_train, y_train = [], []

    for i in range(prediction_days, len(train_data)):
        X_train.append(train_data[i-prediction_days:i, 0])
        y_train.append(train_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Định dạng 3D cho LSTM

    # --- 3. XÂY DỰNG & HUẤN LUYỆN MÔ HÌNH ---
    st.subheader("🤖 Huấn luyện Mô hình Học sâu LSTM")
    
    if st.button("Bắt đầu Huấn luyện và Dự đoán"):
        with st.spinner("Mô hình đang học dữ liệu (Quá trình này có thể mất 1-2 phút)..."):
            # Thiết lập kiến trúc mạng mạng LSTM
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),Dense(units=25),
                Dense(units=1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0) # Epochs=10 để chạy demo nhanh
            
            # --- 4. DỰ ĐOÁN TRÊN TẬP TEST ---
            test_data = scaled_data[training_data_len - prediction_days:, :]
            X_test, y_test = [], dataset[training_data_len:, :]
            
            for i in range(prediction_days, len(test_data)):
                X_test.append(test_data[i-prediction_days:i, 0])
                
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Dự đoán và đưa giá trị về thang đo gốc (inverse_transform)
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)

            # --- 5. TRỰC QUAN HÓA KẾT QUẢ ---
            train = df_close.iloc[:training_data_len]
            valid = df_close.iloc[training_data_len:].copy()
            valid['Predictions'] = predictions

            # Tạo biểu đồ tương tác bằng Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train['Close'].values.flatten(), name='Dữ liệu Train'))
            fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'].values.flatten(), name='Giá Thực Tế (Test)'))
            fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'].values.flatten(), name='Giá Dự Đoán'))
            
            fig.update_layout(
                title=f"Kết quả dự đoán mã {ticker}",
                xaxis_title="Thời gian",
                yaxis_title="Giá (USD / VND)",
                legend_title="Chú thích",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Dự đoán ngày kế tiếp
            last_days = scaled_data[-prediction_days:]
            X_future = np.array([last_days])
            X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
            
            future_pred = model.predict(X_future)
            future_pred = scaler.inverse_transform(future_pred)
            st.success(f"🔮 Giá đóng cửa dự đoán cho ngày giao dịch tiếp theo: **{future_pred[0][0]:,.2f}**")
                Dropout(0.2),
