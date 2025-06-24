from flask import Flask
from flask_cors import CORS
from routes.products import productos_bp
from routes.dashboard import dashboard_bp
from routes.insights import insights_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(productos_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(insights_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Acepta conexiones externas si es necesario
    
