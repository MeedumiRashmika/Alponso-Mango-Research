from flask import Flask
from flask_cors import CORS
from .config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    from .routes.prediction_routes import prediction_bp
    from .routes.file_routes import file_bp
    app.register_blueprint(prediction_bp, url_prefix="/api/v1")
    app.register_blueprint(file_bp, url_prefix="/api/v1/files")

    return app