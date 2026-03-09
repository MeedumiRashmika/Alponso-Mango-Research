from app import create_app
from app.config import Config
from app.services.model_loader import load_models

app = create_app(Config)

# Load models at startup
with app.app_context():
    load_models(Config.MODEL_DIR)
    print("[Startup] Models loaded.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=Config.DEBUG)