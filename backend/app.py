from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow requests from frontend

@app.route("/api/tryon", methods=["POST"])
def tryon():
    if "person" not in request.files or "cloth" not in request.files:
        return jsonify({"error": "Please upload both 'person' and 'cloth' files."}), 400

    person = request.files["person"]
    cloth = request.files["cloth"]

    person_path = UPLOAD_DIR / "person.jpg"
    cloth_path = UPLOAD_DIR / "cloth.jpg"
    person.save(person_path)
    cloth.save(cloth_path)

    # placeholder create simple output
    # For now I just copy the cloth image to output (so frontend shows something)
    # Later I replace this block with actual model inference
    output_path = OUTPUT_DIR / "output.jpg"
    try:
        im = Image.open(cloth_path).convert("RGB")
        # optional: resize to sane size for fast transfer
        im.thumbnail((1024, 1024))
        im.save(output_path, format="JPEG", quality=90)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500
    
    # TODO: run neural network
    
    return send_file(str(output_path), mimetype="image/jpeg")

if __name__ == "__main__":
    # debug True ok for local demo
    app.run(host="127.0.0.1", port=5051, debug=True)
