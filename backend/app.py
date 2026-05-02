from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tryon_pipeline import run_virtual_tryon, TryOnPipelineError

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)


@app.route("/api/tryon", methods=["POST"])
def tryon():
    if "person" not in request.files or "cloth" not in request.files:
        return jsonify({"error": "Please upload both 'person' and 'cloth' files."}), 400

    person = request.files["person"]
    cloth  = request.files["cloth"]

    request_id  = uuid.uuid4().hex
    person_path = UPLOAD_DIR / f"{request_id}_person.jpg"
    cloth_path  = UPLOAD_DIR / f"{request_id}_cloth.png"
    person.save(person_path)
    cloth.save(cloth_path)

    output_path = OUTPUT_DIR / f"{request_id}_output.jpg"
    try:
        run_virtual_tryon(
            person_path,
            cloth_path,
            output_path,
            debug_dir=OUTPUT_DIR,
        )
    except TryOnPipelineError as e:
        return jsonify({"error": f"Try-on failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

    return send_file(str(output_path), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=True)
