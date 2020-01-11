
# Cargando librerias ##################################################
from app import app
from app.utils import findings1

from flask import render_template, session
#######################################################################


# Pagina principal del Dashboard ######################################
@app.route('/dashboard')
def dashboard():
    entrar = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        return render_template('/dashboard/index_dashboard.html')
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Pagina principal de chestXray #######################################
@app.route('/chestXray', methods=['GET', 'POST'])
def chestXray():
    entrar = True
    main = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        return render_template('/dashboard/chestXray_dashboard.html', main=main)
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
        return render_template('/dashboard/patologias_dashboard.html', findings1=findings1)
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
        return render_template('/dashboard/pay_dashboard.html')
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
        return render_template('/dashboard/profile_dashboard.html')
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################


# Salir de la aplicacion ##############################################
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop("USERNAME", None)
    return render_template('/home/index_home.html')
#######################################################################
