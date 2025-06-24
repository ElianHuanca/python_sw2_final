from flask import Blueprint, jsonify
from database import get_connection
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

dashboard_bp = Blueprint('dashboard', __name__)

def predict_next_month_sales_arima(df):
    """
    GEMINI: Predice la venta total para el próximo mes usando modelo ARIMA.
    - Requiere columna 'mes' en formato %Y-%m.
    - Ajusta modelo ARIMA a la serie temporal.
    - Retorna la predicción para el siguiente mes.
    """
    if df.empty or len(df) < 3:
        return 0
    
    df['mes'] = pd.to_datetime(df['mes'])
    df = df.set_index('mes')
    df = df.asfreq('MS')  # 'MS' = inicio de mes (mensual)
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df = df.dropna()

    try:
        model = ARIMA(df['total'], order=(1,1,1))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=1)  # predicción siguiente mes
        return round(float(pred[0]), 2)
    except Exception as e:
        print("Error en ARIMA:", e)
        return 0

def calcular_estadisticas_ventas(df):
    """
    GEMINI: Calcula estadísticas relevantes para ventas mensuales:
    - Mediana: Valor medio en la distribución.
    - Moda: Valor más frecuente.
    - Crecimiento porcentual respecto al mes anterior.
    """
    if df.empty:
        return {"mediana": 0, "moda": 0, "crecimiento": 0}
    
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df = df.dropna()

    mediana = round(df['total'].median(), 2)
    moda = df['total'].mode().iloc[0] if not df['total'].mode().empty else 0
    crecimiento = 0

    if len(df) >= 2:
        crecimiento = ((df['total'].iloc[-1] - df['total'].iloc[-2]) / df['total'].iloc[-2]) * 100

    return {"mediana": mediana, "moda": moda, "crecimiento": round(crecimiento, 2)}

def detectar_anomalias(df, threshold=2):
    """
    GEMINI: Detecta anomalías en las ventas mensuales usando desviación estándar.
    - threshold: número de desviaciones estándar para considerar anomalía.
    - Devuelve lista de meses y ventas donde hubo anomalías.
    """
    if df.empty or len(df) < 2:
        return []
    
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df = df.dropna()

    media = df['total'].mean()
    std = df['total'].std()

    limite_superior = media + threshold * std
    limite_inferior = media - threshold * std

    anomalias = df[(df['total'] > limite_superior) | (df['total'] < limite_inferior)]

    resultado = [{"mes": row['mes'], "total": row['total']} for _, row in anomalias.iterrows()]
    return resultado

@dashboard_bp.route('/api/dashboard/data')
def dashboard_data():
    """
    GEMINI: Endpoint API que devuelve KPIs clave para la gestión de ventas y análisis.
    Devuelve:
    - Totales generales (pedidos, ventas, productos).
    - Ticket promedio.
    - Cliente más frecuente y número de pedidos.
    - Producto más vendido y cantidad.
    - Ventas mensuales históricas.
    - Estadísticas (mediana, moda, crecimiento mensual).
    - Predicción de ventas para el próximo mes (ARIMA).
    - Detección de anomalías en ventas.
    """

    conn = get_connection()
    cursor = conn.cursor()
    data = {}

    # Totales
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
    data['prediccion_mes'] = predict_next_month_sales_arima(ventas_df)
    data['anomalias'] = detectar_anomalias(ventas_df, threshold=2)

    # Descripciones explicativas
    data['gemini_descripciones'] = {
        "predict_next_month_sales_arima": (
            "Predice la venta total para el próximo mes usando modelo ARIMA.\n"
            "- Requiere columna 'mes' en formato %Y-%m.\n"
            "- Ajusta modelo ARIMA a la serie temporal.\n"
            "- Retorna la predicción para el siguiente mes."
        ),
        "calcular_estadisticas_ventas": (
            "Calcula estadísticas relevantes para ventas mensuales:\n"
            "- Mediana: Valor medio en la distribución.\n"
            "- Moda: Valor más frecuente.\n"
            "- Crecimiento porcentual respecto al mes anterior."
        ),
        "detectar_anomalias": (
            "Detecta anomalías en las ventas mensuales usando desviación estándar.\n"
            "- threshold: número de desviaciones estándar para considerar anomalía.\n"
            "- Devuelve lista de meses y ventas donde hubo anomalías."
        ),
        "general": (
            "Este endpoint devuelve KPIs clave para la gestión de ventas y análisis:\n"
            "- Totales generales (pedidos, ventas, productos).\n"
            "- Ticket promedio.\n"
            "- Cliente más frecuente y número de pedidos.\n"
            "- Producto más vendido y cantidad.\n"
            "- Ventas mensuales históricas.\n"
            "- Estadísticas (mediana, moda, crecimiento mensual).\n"
            "- Predicción de ventas para el próximo mes.\n"
            "- Detección de anomalías en ventas."
        )
    }

    cursor.close()
    conn.close()

    return jsonify(data)
