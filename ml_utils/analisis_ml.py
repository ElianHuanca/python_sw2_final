import pandas as pd
from database import get_connection
from sklearn.cluster import KMeans
import joblib

def load_client_clusters():
    model = joblib.load('models/client_clusters.pkl')
    conn = get_connection()
    df = pd.read_sql("""
        SELECT cliente_id, SUM(total) total_spent, COUNT(*) num_orders 
        FROM pedidos GROUP BY cliente_id
    """, conn)
    df['cluster'] = model.predict(df[['total_spent', 'num_orders']])
    return df
