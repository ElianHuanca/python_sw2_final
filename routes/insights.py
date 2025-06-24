from flask import Blueprint, request, jsonify
import os
import google.generativeai as genai

# Configura la clave API desde variables de entorno
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define el blueprint
insights_bp = Blueprint('insights', __name__)

@insights_bp.route('/api/insights', methods=['POST'])
def generate_insights():
    try:
        # Validar tipo de solicitud
        if not request.is_json:
            return jsonify({'error': 'El cuerpo debe ser JSON'}), 400

        payload = request.get_json()
        text = payload.get('text', '')

        if not text:
            return jsonify({'error': 'El campo "text" es obligatorio'}), 400

        # Uso de Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")  # o gemini-1.5-pro
        response = model.generate_content(text)

        # Retorno
        return jsonify({'insights': response.text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
