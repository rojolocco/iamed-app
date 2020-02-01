
# Cargando librerias ##################################################
from app import app
from app.utils import findings1
from app.home_views import db

from flask import render_template, session
#######################################################################


#all_users = db.child("users").get()
#print(all_users.val())


# Pagina principal del Dashboard ######################################
@app.route('/dashboard')
def dashboard():
    entrar = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        data_user = session["USERDATA"]
        print(data_user)
        return render_template('/dashboard/index_dashboard.html', data_user=data_user)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Pagina principal de IA.Torax #######################################
@app.route('/chestXray', methods=['GET', 'POST'])
def chestXray():
    entrar = True
    main = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        data_user = session["USERDATA"]
        return render_template('/dashboard/chestXray_dashboard.html', main=main, data_user=data_user)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Pagina principal de IA.Triage #######################################
@app.route('/triage', methods=['GET', 'POST'])
def triage():
    entrar = True
    main = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        data_user = session["USERDATA"]
        return render_template('/dashboard/triage_dashboard.html', main=main, data_user=data_user)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Pagina principal de patologias #######################################
@app.route('/patologias', methods=['GET', 'POST'])
def patologias():
    entrar = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        data_user = session["USERDATA"]
        return render_template('/dashboard/patologias_dashboard.html', 
        findings1=findings1, data_user=data_user)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Pagina principal de pagos ##########################################
@app.route('/reportes', methods=['GET', 'POST'])
def reportes():
    entrar = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        data_user = session["USERDATA"]
        return render_template('/dashboard/reportes_dashboard.html', data_user=data_user)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Pagina principal de pagos ##########################################
@app.route('/pay', methods=['GET', 'POST'])
def pay():
    entrar = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        data_user = session["USERDATA"]
        return render_template('/dashboard/pay_dashboard.html', data_user=data_user)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Pagina principal de perfil de usuarios ###############################
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    entrar = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        data_user = session["USERDATA"]
        return render_template('/dashboard/profile_dashboard.html', data_user=data_user)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Salir de la aplicacion ##############################################
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop("USERNAME", None)
    session.pop("USERDATA", None)
    return render_template('/home/index_home.html')
#######################################################################
