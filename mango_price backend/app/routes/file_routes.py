import os
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

file_bp = Blueprint("files", __name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'xlsx', 'xls', 'json', 'xml'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@file_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload a file to the server.
    Expected: multipart/form-data with 'file' field
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": f"File size exceeds maximum limit of {MAX_FILE_SIZE / (1024*1024):.0f}MB"}), 413

    try:
        # Secure the filename
        filename = secure_filename(file.filename)

        # Handle duplicate filenames
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(UPLOAD_FOLDER, f"{name}_{counter}{ext}")):
                counter += 1
            filename = f"{name}_{counter}{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save the file
        file.save(filepath)

        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "file_url": f"/api/v1/files/download/{filename}"
        }), 201

    except Exception as e:
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500


@file_bp.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """
    Download an uploaded file.
    Public endpoint - no authentication required.
    """
    try:
        # Secure the filename to prevent directory traversal
        filename = secure_filename(filename)

        # Check if file exists
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        # Serve the file
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)

    except Exception as e:
        return jsonify({"error": f"File download failed: {str(e)}"}), 500


@file_bp.route("/list", methods=["GET"])
def list_files():
    """
    List all uploaded files.
    """
    try:
        files = os.listdir(UPLOAD_FOLDER)
        file_list = []

        for filename in files:
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                file_size = os.path.getsize(filepath)
                file_list.append({
                    "filename": filename,
                    "size": file_size,
                    "size_mb": round(file_size / (1024*1024), 2),
                    "url": f"/api/v1/files/download/{filename}"
                })

        return jsonify({
            "total_files": len(file_list),
            "files": file_list
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to list files: {str(e)}"}), 500
