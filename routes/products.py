from flask import Blueprint, request, jsonify
from database import get_connection
import pandas as pd
import datetime

productos_bp = Blueprint('productos', __name__)

@productos_bp.route('/api/productos')
def get_productos():
    # Parámetros GET opcionales
    categoria = request.args.get('categoria')
    marca = request.args.get('marca')
    min_precio = request.args.get('min_precio', type=float)
    max_precio = request.args.get('max_precio', type=float)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Armar consulta SQL dinámica
    query = "SELECT p.*, c.nombre AS categoria_nombre, m.nombre AS marca_nombre FROM productos p " \
            "LEFT JOIN categorias c ON p.categoria_id = c.id " \
            "LEFT JOIN marcas m ON p.marca_id = m.id WHERE 1=1"
    
    filtros = []
    if categoria:
        query += " AND c.nombre = %s"
        filtros.append(categoria)
    if marca:
        query += " AND m.nombre = %s"
        filtros.append(marca)
    if min_precio is not None:
        query += " AND p.precio >= %s"
        filtros.append(min_precio)
    if max_precio is not None:
        query += " AND p.precio <= %s"
        filtros.append(max_precio)

    cursor.execute(query, filtros)
    productos = cursor.fetchall()

    # Convertir a DataFrame de pandas para análisis
    df = pd.DataFrame(productos)
    total_productos = len(df)
    promedio_precio = round(df["precio"].mean(), 2) if not df.empty else 0
    total_stock = int(df["stock"].sum()) if not df.empty else 0

    # Insertar bitácora
    bitacora_cursor = conn.cursor()
    bitacora_cursor.execute("""
        INSERT INTO bitacoras (accion, apartado, afectado, fecha_h, id_user, ip, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        "Consulta productos", "Productos", f"{total_productos} encontrados",
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        None,  # o request.user.id si usas autenticación
        request.remote_addr,
        datetime.datetime.now()
    ))
    conn.commit()

    # Cierre de cursores
    cursor.close()
    bitacora_cursor.close()
    conn.close()

    return jsonify({
        "total_productos": total_productos,
        "precio_promedio": promedio_precio,
        "stock_total": total_stock,
        "productos": productos
    })
