
# Cargando librerias ##################################################
from app import app

import requests
import pyrebase

from flask import render_template, request, session, redirect, url_for
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
db = firebase.database()
storage = firebase.storage()
######################################################################


# Pagina Principal - Home ############################################
@app.route('/')
def index():
    return render_template('/home/index_home.html')
######################################################################


# Pagina de Acceso - Register #########################################
@app.route('/register_home', methods=['GET', 'POST'])
def register_home():
    unsuccessful = True
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form['username']
        try:
            new_user = auth.create_user_with_email_and_password(email, password)
            response = auth.sign_in_with_email_and_password(email, password)
            id_user = response['localId']
            if response['registered']:
                session["USERNAME"] = id_user
                data_user = {'userId': id_user,
                            'nombre': name,
                            'email': email,
                            'tipo': 'nuevo',
                            'avatar': 'nuevo.jpg',
                            'consultas_xusar': 0,
                            'consultas_totales': 0,
                            'telefono': 'NA',
                            'ciudad': 'NA',
                            'profesion': 'NA',
                            'especialidad': 'NA',
                            'terminos':'True',
                            'apps':'NA'
                        }
                db.child("users").child(response['localId']).set(data_user)
                session["USERDATA"] = db.child("users").get().val()[id_user]
                return render_template('/dashboard/index_dashboard.html', data_user=data_user)
        except requests.exceptions.HTTPError:
            return render_template('/home/register_home.html', un=unsuccessful)
    return render_template('/home/register_home.html')
######################################################################


# Pagina de Acceso - LogIn ###########################################
@app.route('/login_home', methods=['GET', 'POST'])
def login_home():
    unsuccessful = True
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            response = auth.sign_in_with_email_and_password(email, password)
            id_user = response['localId']
            if response['registered']:
                session["USERNAME"] = id_user
                session["USERDATA"] = db.child("users").get().val()[id_user]
                return redirect(url_for('dashboard')) #render_template('/dashboard/index_dashboard.html')
        except requests.exceptions.HTTPError:
            return render_template('/home/login_home.html', un=unsuccessful)
    return render_template('/home/login_home.html')
######################################################################
