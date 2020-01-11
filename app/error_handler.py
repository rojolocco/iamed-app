
# Cargando librerias ##################################################
from app import app
from flask import render_template, session
#######################################################################


# Cargando Pagina 404 ##################################################
@app.errorhandler(404)
def page_not_found(error):
    entrar = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        return render_template('/home/404.html', title='404'), 404
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################
