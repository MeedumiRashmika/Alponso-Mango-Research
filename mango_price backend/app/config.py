import os


class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    MODEL_DIR = os.getenv("MODEL_DIR", "output/models")
    DATA_FILE = os.getenv("DATA_FILE", "dataset/mango_price_dataset.csv")
    SECRET_KEY = os.getenv("SECRET_KEY", "mango-secret-key")