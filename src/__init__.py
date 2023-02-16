from dotenv import load_dotenv
from flask import Flask
import os

load_dotenv

app = Flask(__name__)

app.secret_key = os.urandom(24)

from src.main.routes import main
app.register_blueprint(main.main)

