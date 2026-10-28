from datetime import datetime
from pathlib import Path
import os
import sys
import uuid

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import certifi
except ImportError:
    certifi = None


# Certificate settings needed for downloading model files
if certifi is not None:
    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)


# Main project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tryon_pipeline import TryOnPipelineError, run_virtual_tryon


BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Flask application
app = Flask(__name__)


# Secret key is used for sessions
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "development-secret-key-change-later",
)


# Main database is SQLite.
# DATABASE_URL can be changed for testing.
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{BASE / 'virtual_tryon.db'}",
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Maximum upload size is 20 MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# Cookie settings for local development
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# Frontend and backend run on different ports,
# so CORS allows the frontend to use this API.
CORS(
    app,
    supports_credentials=True,
    origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    expose_headers=["X-Request-ID"],
)


# Database setup
db = SQLAlchemy(app)


# Login and session management
login_manager = LoginManager()
login_manager.init_app(app)


# User table
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    email = db.Column(
        db.String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    # Only password hash is stored, not the real password
    password_hash = db.Column(
        db.String(255),
        nullable=False,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # One user can have many saved results
    results = db.relationship(
        "TryOnResult",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )


# Saved try-on results table
class TryOnResult(db.Model):
    __tablename__ = "tryon_results"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    # Foreign key connecting the result to one user
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    person_image_path = db.Column(
        db.String(500),
        nullable=False,
    )

    cloth_image_path = db.Column(
        db.String(500),
        nullable=False,
    )

    result_image_path = db.Column(
        db.String(500),
        nullable=False,
    )

    store_url = db.Column(
        db.String(1000),
        nullable=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )


# Flask-Login loads the current user from the session
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(
            User,
            int(user_id),
        )
    except (TypeError, ValueError):
        return None


# Return JSON if an unauthenticated user opens a protected endpoint
@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({
        "error": "Authentication required."
    }), 401


# Normalize email before saving or searching
def normalize_email(value):
    return str(value or "").strip().lower()


# Convert User object to JSON-friendly data
def serialize_user(user):
    return {
        "id": user.id,
        "email": user.email,
    }


# Convert saved result to JSON-friendly data
def serialize_result(result):
    return {
        "id": result.id,
        "store_url": result.store_url,
        "created_at": result.created_at.isoformat(),
        "image_url": f"/api/results/{result.id}/image",
    }


# Check if request_id has expected UUID format
def valid_request_id(request_id):
    if len(request_id) != 32:
        return False

    return all(
        character in "0123456789abcdef"
        for character in request_id.lower()
    )


# ---------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------

@app.route("/api/register", methods=["POST"])
def register():

    # Read JSON from frontend
    data = request.get_json(silent=True) or {}

    email = normalize_email(data.get("email"))
    password = str(data.get("password") or "")

    # Basic validation
    if not email or not password:
        return jsonify({
            "error": "Email and password are required."
        }), 400

    if (
        "@" not in email
        or email.startswith("@")
        or email.endswith("@")
    ):
        return jsonify({
            "error": "Please enter a valid email address."
        }), 400

    if len(email) > 255:
        return jsonify({
            "error": "Email address is too long."
        }), 400

    if len(password) < 6:
        return jsonify({
            "error": "Password must contain at least 6 characters."
        }), 400


    # Check if this email already exists
    existing_user = User.query.filter_by(
        email=email
    ).first()

    if existing_user is not None:
        return jsonify({
            "error": "This email is already registered."
        }), 409


    # Create new user with hashed password
    user = User(
        email=email,
        password_hash=generate_password_hash(password),
    )

    try:
        db.session.add(user)
        db.session.commit()

    except IntegrityError:
        db.session.rollback()

        return jsonify({
            "error": "This email is already registered."
        }), 409

    except Exception:
        db.session.rollback()

        return jsonify({
            "error": "Registration failed due to a database error."
        }), 500


    # Log user in after registration
    login_user(user)

    return jsonify({
        "message": "Registration successful.",
        "user": serialize_user(user),
    }), 201


# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------

@app.route("/api/login", methods=["POST"])
def login():

    data = request.get_json(silent=True) or {}

    email = normalize_email(data.get("email"))
    password = str(data.get("password") or "")

    if not email or not password:
        return jsonify({
            "error": "Email and password are required."
        }), 400


    # Find user by email
    user = User.query.filter_by(
        email=email
    ).first()


    # Check password hash
    if (
        user is None
        or not check_password_hash(
            user.password_hash,
            password,
        )
    ):
        return jsonify({
            "error": "Invalid email or password."
        }), 401


    # Save user in session
    login_user(user)

    return jsonify({
        "message": "Login successful.",
        "user": serialize_user(user),
    }), 200


# ---------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------

@app.route("/api/logout", methods=["POST"])
@login_required
def logout():

    logout_user()

    return jsonify({
        "message": "Logout successful."
    }), 200


# ---------------------------------------------------------
# CURRENT USER
# ---------------------------------------------------------

# Frontend uses this endpoint to check current session
@app.route("/api/me", methods=["GET"])
def me():

    if not current_user.is_authenticated:
        return jsonify({
            "authenticated": False,
            "user": None,
        }), 200

    return jsonify({
        "authenticated": True,
        "user": serialize_user(current_user),
    }), 200


# ---------------------------------------------------------
# TRY-ON
# ---------------------------------------------------------

@app.route("/api/tryon", methods=["POST"])
def tryon():

    # Both files are required
    if (
        "person" not in request.files
        or "cloth" not in request.files
    ):
        return jsonify({
            "error": (
                "Please upload both "
                "'person' and 'cloth' files."
            )
        }), 400


    person = request.files["person"]
    cloth = request.files["cloth"]


    if not person.filename or not cloth.filename:
        return jsonify({
            "error": (
                "Both uploaded files "
                "must have a filename."
            )
        }), 400


    # Unique ID is used in file names
    request_id = uuid.uuid4().hex


    person_path = (
        UPLOAD_DIR / f"{request_id}_person.jpg"
    )

    cloth_path = (
        UPLOAD_DIR / f"{request_id}_cloth.png"
    )

    output_path = (
        OUTPUT_DIR / f"{request_id}_output.jpg"
    )


    try:
        # Save uploaded files on the server
        person.save(person_path)
        cloth.save(cloth_path)

        run_virtual_tryon(
            person_path,
            cloth_path,
            output_path,
            debug_dir=OUTPUT_DIR,
        )

    except TryOnPipelineError as error:
        return jsonify({
            "error": f"Try-on failed: {error}"
        }), 500

    except Exception as error:
        return jsonify({
            "error": f"Unexpected error: {error}"
        }), 500


    if not output_path.exists():
        return jsonify({
            "error": (
                "The model did not create "
                "an output image."
            )
        }), 500


    # Send generated image back to frontend
    response = send_file(
        str(output_path),
        mimetype="image/jpeg",
    )


    # Frontend keeps this ID for saving the result later
    response.headers["X-Request-ID"] = request_id

    return response


# ---------------------------------------------------------
# SAVE RESULT
# ---------------------------------------------------------

@app.route("/api/results", methods=["POST"])
@login_required
def save_result():

    data = request.get_json(silent=True) or {}

    request_id = str(
        data.get("request_id") or ""
    ).strip()

    store_url = str(
        data.get("store_url") or ""
    ).strip() or None


    if not request_id:
        return jsonify({
            "error": "request_id is required."
        }), 400


    if not valid_request_id(request_id):
        return jsonify({
            "error": "Invalid request_id."
        }), 400


    # Recreate file paths from request_id
    person_path = (
        UPLOAD_DIR / f"{request_id}_person.jpg"
    )

    cloth_path = (
        UPLOAD_DIR / f"{request_id}_cloth.png"
    )

    result_path = (
        OUTPUT_DIR / f"{request_id}_output.jpg"
    )


    # Files must exist before saving the database row
    if not person_path.exists():
        return jsonify({
            "error": "Person image was not found."
        }), 404

    if not cloth_path.exists():
        return jsonify({
            "error": "Garment image was not found."
        }), 404

    if not result_path.exists():
        return jsonify({
            "error": "Generated result was not found."
        }), 404


    # Do not save the same result twice
    existing_result = TryOnResult.query.filter_by(
        user_id=current_user.id,
        result_image_path=str(result_path),
    ).first()


    if existing_result is not None:
        return jsonify({
            "error": "This result has already been saved."
        }), 409


    # Create new database record
    result = TryOnResult(
        user_id=current_user.id,
        person_image_path=str(person_path),
        cloth_image_path=str(cloth_path),
        result_image_path=str(result_path),
        store_url=store_url,
    )


    try:
        db.session.add(result)
        db.session.commit()

    except Exception:
        db.session.rollback()

        return jsonify({
            "error": (
                "The result could not be saved "
                "to the database."
            )
        }), 500


    return jsonify({
        "message": "Result saved successfully.",
        "result": serialize_result(result),
    }), 201


# ---------------------------------------------------------
# LOAD WARDROBE
# ---------------------------------------------------------

@app.route("/api/results", methods=["GET"])
@login_required
def get_results():

    # Return only results belonging to the current user
    results = (
        TryOnResult.query
        .filter_by(user_id=current_user.id)
        .order_by(TryOnResult.created_at.desc())
        .all()
    )

    return jsonify({
        "results": [
            serialize_result(result)
            for result in results
        ]
    }), 200


# ---------------------------------------------------------
# GET SAVED IMAGE
# ---------------------------------------------------------

@app.route(
    "/api/results/<int:result_id>/image",
    methods=["GET"],
)
@login_required
def get_result_image(result_id):

    result = db.session.get(
        TryOnResult,
        result_id,
    )


    if result is None:
        return jsonify({
            "error": "Result not found."
        }), 404


    # User can only open their own result
    if result.user_id != current_user.id:
        return jsonify({
            "error": "Access denied."
        }), 403


    image_path = Path(result.result_image_path)


    if not image_path.exists():
        return jsonify({
            "error": "Image file not found."
        }), 404


    return send_file(
        str(image_path),
        mimetype="image/jpeg",
    )


# ---------------------------------------------------------
# DELETE RESULT
# ---------------------------------------------------------

@app.route(
    "/api/results/<int:result_id>",
    methods=["DELETE"],
)
@login_required
def delete_result(result_id):

    result = db.session.get(
        TryOnResult,
        result_id,
    )


    if result is None:
        return jsonify({
            "error": "Result not found."
        }), 404


    # User can only delete their own result
    if result.user_id != current_user.id:
        return jsonify({
            "error": "Access denied."
        }), 403


    try:
        db.session.delete(result)
        db.session.commit()

    except Exception:
        db.session.rollback()

        return jsonify({
            "error": (
                "The result could not be deleted "
                "from the database."
            )
        }), 500


    return jsonify({
        "message": "Result deleted successfully."
    }), 200


# Return JSON error for files larger than 20 MB
@app.errorhandler(413)
def file_too_large(_error):

    return jsonify({
        "error": "Uploaded files are too large."
    }), 413


# Create database tables if they do not exist yet
with app.app_context():
    db.create_all()


# Start local Flask server
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5051,
        debug=True,
        use_reloader=False,
    )