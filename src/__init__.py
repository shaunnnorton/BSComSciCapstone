from dotenv import load_dotenv
from flask import Flask
import os

# Load environmental Variables
load_dotenv()
# Create the flask app
app = Flask(__name__)
# Set the secret key
app.secret_key = os.urandom(24)

# Apply routes from main to the app
from src.main.routes import main
app.register_blueprint(main.main)

