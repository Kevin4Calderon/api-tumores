import os
from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename

from processing.tumor_detector import TumorDetector
from processing.image_processor import generate_visualizations

app = Flask(__name__)

# =========================
# CONFIGURACION
# =========================

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================
# VALIDAR ARCHIVO
# =========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# FRONTEND (WEB)
# =========================

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            detector = TumorDetector()
            result = detector.detect_tumor(filepath)

            base_name = os.path.splitext(filename)[0]

            result_images = generate_visualizations(
                filepath,
                app.config['RESULT_FOLDER'],
                base_name
            )

            return render_template(
                'index.html',
                original_image=filepath,
                result_images=result_images,
                has_tumor=result['has_tumor'],
                confidence=f"{result['confidence']:.2f}%"
            )

    return render_template('index.html')

# =========================
# API PARA N8N
# =========================

@app.route('/analyze', methods=['POST'])
def analyze_image():

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se envio archivo"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "Archivo vacio"}), 400

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            detector = TumorDetector()
            result = detector.detect_tumor(filepath)

            base_name = os.path.splitext(filename)[0]

            result_images = generate_visualizations(
                filepath,
                app.config['RESULT_FOLDER'],
                base_name
            )

            # 🔥 FIX IMPORTANTE: convertir bool a string
            return jsonify({
                "has_tumor": str(result['has_tumor']),
                "confidence": float(round(result['confidence'], 2)),
                "images": result_images
            })

        return jsonify({"error": "Formato no permitido"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN
# =========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)