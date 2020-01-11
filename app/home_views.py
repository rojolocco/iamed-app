
# Cargando librerias ##################################################
from app import app

import requests
import pyrebase

from flask import render_template, request, session
######################################################################


# Configurando Firebase ##############################################
config = {
    "apiKey": "AIzaSyDAAYooQSrfOwe-KOwYis27y7ZUSWtvzkw",
    "authDomain": "iamed-webapp-auth.firebaseapp.com",
    "databaseURL": "https://iamed-webapp-auth.firebaseio.com",
    "projectId": "iamed-webapp-auth",
    "storageBucket": "iamed-webapp-auth.appspot.com",
    "messagingSenderId": "591396173864",
    "appId": "1:591396173864:web:ba71b19ae187736bc91d1d",
    "measurementId": "G-T9H8C2VLHN"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
######################################################################


# Pagina Principal - Home ############################################
@app.route('/')
def index():
    return render_template('/home/index_home.html')
######################################################################


# Pagina de Acceso - LogIn01##########################################
@app.route('/login_home', methods=['GET', 'POST'])
def login_home():
    unsuccessful = True
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            response = auth.sign_in_with_email_and_password(email, password)
            if response['registered']:
                session["USERNAME"] = email
                return render_template('/dashboard/index_dashboard.html')
        except requests.exceptions.HTTPError:
            return render_template('/home/login_home.html', un=unsuccessful)
    return render_template('/home/login_home.html')
######################################################################
