from flask import Blueprint, jsonify
from database import get_connection
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import warnings
from datetime import datetime, timedelta
import math

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configura la clave API desde variables de entorno
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
dashboard_bp = Blueprint('dashboard', __name__)

# Configuración de modelos
MODEL_CONFIG = {
    'min_data_points': {
        'ARIMA': 3,
        'SARIMA': 12,
        'Random_Forest': 6,
        'LSTM': 12
    },
    'weights': {
        'accuracy': 0.4,
        'recency': 0.3,
        'complexity': 0.2,
        'stability': 0.1
    }
}

def prepare_time_series_data(df):
    """Prepares time series data for modeling with improved validation"""
    if df.empty:
        return None
    
    try:
        df['mes'] = pd.to_datetime(df['mes'])
        df = df.set_index('mes')
        df = df.asfreq('MS')  # 'MS' = inicio de mes (mensual)
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        df = df.dropna()
        
        if len(df) < max(MODEL_CONFIG['min_data_points'].values()):
            return None
        
        return df
    except Exception as e:
        print(f"Error preparing time series data: {e}")
        return None

def predict_next_month_sales_arima(df):
    """Predicción con ARIMA mejorado con scoring"""
    df_clean = prepare_time_series_data(df)
    if df_clean is None:
        return {"prediction": 0, "model": "ARIMA", "error": "Insufficient data"}
    
    try:
        # Auto-select parameters based on AIC
        best_aic = np.inf
        best_order = (1,1,1)
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(df_clean['total'], order=(p,d,q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p,d,q)
                    except:
                        continue
        
        model = ARIMA(df_clean['total'], order=best_order)
        model_fit = model.fit()
        
        # Temporal cross-validation
        train_size = int(len(df_clean) * 0.8)
        train, test = df_clean.iloc[:train_size], df_clean.iloc[train_size:]
        model_val = ARIMA(train['total'], order=best_order)
        model_val_fit = model_val.fit()
        
        pred = model_fit.forecast(steps=1)
        
        # Calculate model score
        score = calculate_model_score(
            "ARIMA", 
            model_val_fit.aic, 
            len(df_clean)
        )
        
        return {
            'prediction': round(float(pred[0]), 2),
            'model': 'ARIMA',
            'parameters': str(best_order),
            'accuracy': round(model_val_fit.aic, 2),
            'score': score
        }
    except Exception as e:
        print("Error en ARIMA:", e)
        return {
            'prediction': 0,
            'model': 'ARIMA',
            'error': str(e),
            'score': 0
        }

def predict_next_month_sales_sarima(df):
    """
    Predice ventas usando SARIMA (ARIMA estacional)
    - Maneja patrones estacionales (ej. ventas navideñas)
    - Parámetros: (p,d,q)(P,D,Q,s) donde s=12 (para datos mensuales)
    """
    df_clean = prepare_time_series_data(df)
    if df_clean is None:
        return 0
    
    try:
        # Simple SARIMA configuration (can be enhanced with auto-selection)
        model = SARIMAX(df_clean['total'], 
                       order=(1,1,1), 
                       seasonal_order=(1,1,1,12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        
        pred = model_fit.forecast(steps=1)
        return {
            'prediction': round(float(pred[0]), 2),
            'model': 'SARIMA',
            'parameters': '(1,1,1)(1,1,1,12)'
        }
    except Exception as e:
        print("Error en SARIMA:", e)
        return {
            'prediction': 0,
            'model': 'SARIMA',
            'error': str(e)
        }

def predict_next_month_sales_random_forest(df):
    """
    Predice ventas usando Random Forest
    - Crea características temporales (mes, trimestre, año)
    - Usa ventas pasadas como características
    """
    df_clean = prepare_time_series_data(df)
    if df_clean is None or len(df_clean) < 6:
        return {
            'prediction': 0,
            'model': 'Random Forest',
            'error': 'Not enough data'
        }
    
    try:
        # Create features
        df_features = df_clean.copy()
        df_features['year'] = df_features.index.year
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        
        # Lag features (previous months sales)
        for i in range(1, 4):
            df_features[f'lag_{i}'] = df_features['total'].shift(i)
        
        df_features = df_features.dropna()
        
        if len(df_features) < 3:
            return {
                'prediction': 0,
                'model': 'Random Forest',
                'error': 'Not enough data after feature engineering'
            }
        
        # Prepare data
        X = df_features.drop('total', axis=1)
        y = df_features['total']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Prepare next month's features
        last_date = df_features.index[-1]
        next_month = last_date + timedelta(days=31)
        
        next_features = {
            'year': next_month.year,
            'month': next_month.month,
            'quarter': (next_month.month-1)//3 + 1,
            'lag_1': df_features['total'].iloc[-1],
            'lag_2': df_features['total'].iloc[-2],
            'lag_3': df_features['total'].iloc[-3]
        }
        
        next_features = pd.DataFrame([next_features], index=[next_month])
        next_features = next_features[X.columns]  # ensure same column order
        
        # Predict
        pred = model.predict(next_features)
        
        return {
            'prediction': round(float(pred[0]), 2),
            'model': 'Random Forest',
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
    except Exception as e:
        print("Error en Random Forest:", e)
        return {
            'prediction': 0,
            'model': 'Random Forest',
            'error': str(e)
        }

def predict_next_month_sales_lstm(df):
    """
    Predice ventas usando LSTM (Red Neuronal Recurrente)
    - Ideal para patrones temporales complejos
    - Requiere más datos que otros métodos
    """
    df_clean = prepare_time_series_data(df)
    if df_clean is None or len(df_clean) < 12:
        return {
            'prediction': 0,
            'model': 'LSTM',
            'error': 'Not enough data (minimum 12 months required)'
        }
    
    try:
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean[['total']])
        
        # Prepare sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data)-seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        seq_length = 3
        X, y = create_sequences(scaled_data, seq_length)
        
        # Train/test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train
        model.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_test, y_test))
        
        # Prepare last sequence for prediction
        last_sequence = scaled_data[-seq_length:]
        last_sequence = last_sequence.reshape((1, seq_length, 1))
        
        # Predict
        scaled_pred = model.predict(last_sequence)
        pred = scaler.inverse_transform(scaled_pred)
        
        return {
            'prediction': round(float(pred[0][0]), 2),
            'model': 'LSTM',
            'parameters': f'seq_length={seq_length}, epochs=50'
        }
    except Exception as e:
        print("Error en LSTM:", e)
        return {
            'prediction': 0,
            'model': 'LSTM',
            'error': str(e)
        }


def generate_ai_insights(data):
    """Genera insights usando la API de Gemini"""
    try:
        prompt = f"""
        Analiza los siguientes datos de ventas y predicciones:
        - Estadísticas: {data['estadisticas']}
        - Anomalías: {data['anomalias']}
        - Predicciones: {data['predicciones']}
        
        Proporciona:
        1. Un resumen ejecutivo de 2-3 frases
        2. 3 insights clave sobre patrones de ventas
        3. Recomendaciones de acción basadas en las predicciones
        4. Explicación breve de qué modelo de predicción parece más confiable y por qué
        
        Usa un tono profesional pero conciso.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generando insights con Gemini: {e}")
        return "No se pudieron generar insights en este momento."

def calcular_estadisticas_ventas(df):
    """
    Mejorado: Calcula estadísticas avanzadas de ventas
    - Ahora incluye desviación estándar, rango intercuartil (IQR)
    - Cálculo de tendencia usando regresión lineal simple
    """
    if df.empty:
        return {
            "mediana": 0,
            "moda": 0,
            "crecimiento": 0,
            "std": 0,
            "iqr": 0,
            "tendencia": "estable"
        }
    
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df = df.dropna()

    # Basic stats
    mediana = round(df['total'].median(), 2)
    moda = df['total'].mode().iloc[0] if not df['total'].mode().empty else 0
    std_dev = round(df['total'].std(), 2)
    
    if math.isnan(std_dev):
        std_dev = 0  # o None, si prefieres

    # IQR
    q75, q25 = np.percentile(df['total'], [75, 25])
    iqr = round(q75 - q25, 2)
    
    # Growth calculation
    crecimiento = 0
    if len(df) >= 2:
        crecimiento = ((df['total'].iloc[-1] - df['total'].iloc[-2]) / df['total'].iloc[-2]) * 100
    
    # Trend analysis (simple linear regression)
    trend = "estable"
    if len(df) >= 3:
        x = np.arange(len(df))
        y = df['total'].values
        coef = np.polyfit(x, y, 1)
        if coef[0] > 0.5:
            trend = "ascendente"
        elif coef[0] < -0.5:
            trend = "descendente"
    
    return {
        "mediana": mediana,
        "moda": moda,
        "crecimiento": round(crecimiento, 2),
        "std": std_dev,
        "iqr": iqr,
        "tendencia": trend
    }

def detectar_anomalias(df, threshold=2):
    """Detección de anomalías mejorada con múltiples métodos"""
    if df.empty or len(df) < 2:
        return []
    
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df = df.dropna()

    # Métodos de detección
    media = df['total'].mean()
    std = df['total'].std()
    q25 = df['total'].quantile(0.25)
    q75 = df['total'].quantile(0.75)
    iqr = q75 - q25
    
    # Rangos
    ranges = {
        'std_upper': media + threshold * std,
        'std_lower': media - threshold * std,
        'iqr_upper': q75 + 1.5 * iqr,
        'iqr_lower': q25 - 1.5 * iqr
    }
    
    # Detectar anomalías
    anomalias = []
    for _, row in df.iterrows():
        is_anomaly = False
        methods = []
        
        if row['total'] > ranges['std_upper'] or row['total'] < ranges['std_lower']:
            is_anomaly = True
            methods.append('std')
            
        if row['total'] > ranges['iqr_upper'] or row['total'] < ranges['iqr_lower']:
            is_anomaly = True
            methods.append('iqr')
        
        if is_anomaly:
            score = abs((row['total'] - media) / std) if std != 0 else 0
            anomalias.append({
                "mes": row['mes'],
                "total": row['total'],
                "score": round(score, 2),
                "methods": methods,
                "deviation": round((row['total'] - media)/media * 100, 2) if media != 0 else 0
            })
    
    return anomalias

def rank_models(predictions):
    """Rankea los modelos basado en sus scores"""
    ranked = sorted(
        [(k, v) for k, v in predictions.items() if v.get('score', 0) > 0],
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    # Asignar posiciones (manejar empates)
    result = []
    prev_score = None
    position = 0
    skip = 1
    
    for i, (model_name, data) in enumerate(ranked):
        if data['score'] != prev_score:
            position += skip
            skip = 1
        else:
            skip += 1
            
        result.append({
            'position': position,
            'model': model_name,
            'score': data['score'],
            'prediction': data['prediction'],
            'accuracy': data.get('accuracy', 'N/A')
        })
        
        prev_score = data['score']
    
    return result
  

@dashboard_bp.route('/api/dashboard/data')
def dashboard_data():
    """
    Endpoint API mejorado que ahora incluye:
    - Predicciones de múltiples modelos (ARIMA, SARIMA, Random Forest, LSTM)
    - Estadísticas avanzadas
    - Detección de anomalías mejorada
    - Explicaciones de modelos generadas por IA
    """
    conn = get_connection()
    cursor = conn.cursor()
    data = {}

    # Totales básicos (sin cambios)
    cursor.execute("SELECT COUNT(*) FROM pedidos")
    data['total_pedidos'] = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(total) FROM pedidos")
    data['total_ventas'] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM productos")
    data['total_productos'] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(total) FROM pedidos")
    data['ticket_promedio'] = round(cursor.fetchone()[0] or 0, 2)

    cursor.execute("""
        SELECT cliente_id, COUNT(*) as cantidad FROM pedidos
        GROUP BY cliente_id ORDER BY cantidad DESC LIMIT 1
    """)
    cliente_frecuente = cursor.fetchone()
    data['cliente_frecuente_id'] = cliente_frecuente[0] if cliente_frecuente else "N/A"
    data['cliente_frecuente_pedidos'] = cliente_frecuente[1] if cliente_frecuente else 0

    cursor.execute("""
        SELECT producto_id, SUM(cantidad) as cantidad FROM detalle_pedidos
        GROUP BY producto_id ORDER BY cantidad DESC LIMIT 1
    """)
    producto_top = cursor.fetchone()
    data['producto_top_id'] = producto_top[0] if producto_top else "N/A"
    data['producto_top_cantidad'] = producto_top[1] if producto_top else 0

    # Ventas mensuales
    cursor.execute("""
        SELECT DATE_FORMAT(fecha_pedido, '%Y-%m') mes, SUM(total) total
        FROM pedidos GROUP BY mes ORDER BY mes
    """)
    ventas_mes_raw = cursor.fetchall()
    ventas_df = pd.DataFrame(ventas_mes_raw, columns=['mes', 'total'])

    data['ventas_mes'] = ventas_mes_raw
    data['estadisticas'] = calcular_estadisticas_ventas(ventas_df)
    data['anomalias'] = detectar_anomalias(ventas_df, threshold=2)
    
    # Predicciones con múltiples modelos
    data['predicciones'] = {
        'ARIMA': predict_next_month_sales_arima(ventas_df),
        'SARIMA': predict_next_month_sales_sarima(ventas_df),
        'Random_Forest': predict_next_month_sales_random_forest(ventas_df),
        'LSTM': predict_next_month_sales_lstm(ventas_df)
    }
    
    # Consenso de predicciones
    try:
        predictions = [
            data['predicciones']['ARIMA']['prediction'],
            data['predicciones']['SARIMA']['prediction'],
            data['predicciones']['Random_Forest']['prediction'],
            data['predicciones']['LSTM']['prediction']
        ]
        valid_predictions = [p for p in predictions if p > 0]
        if valid_predictions:
            data['prediccion_consenso'] = round(sum(valid_predictions) / len(valid_predictions), 2)
        else:
            data['prediccion_consenso'] = 0
    except:
        data['prediccion_consenso'] = 0

    # Descripciones explicativas generadas por IA
    data['model_descriptions'] = {
        "ARIMA": {
            "description": "Modelo estadístico para series temporales que usa autocorrelación",
            "strengths": "Bueno para patrones a corto plazo, simple de implementar",
            "weaknesses": "No maneja bien patrones estacionales complejos"
        },
        "SARIMA": {
            "description": "ARIMA con componente estacional, ideal para patrones que se repiten",
            "strengths": "Excelente para patrones estacionales (ej. ventas navideñas)",
            "weaknesses": "Requiere más datos y es más complejo de configurar"
        },
        "Random_Forest": {
            "description": "Modelo de ensamble basado en árboles de decisión",
            "strengths": "Maneja bien relaciones no lineales, robusto a outliers",
            "weaknesses": "No captura inherentemente dependencias temporales"
        },
        "LSTM": {
            "description": "Red neuronal recurrente especializada en secuencias",
            "strengths": "Excelente para patrones temporales complejos a largo plazo",
            "weaknesses": "Requiere muchos datos y poder computacional"
        }
    }

    cursor.close()
    conn.close()

    return jsonify(data)