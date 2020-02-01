# Cargando librerias ##################################################
from app import app
from app.home_views import db

from flask import render_template, request, session, jsonify
#######################################################################

all_users = db.child("users").get()

# Pagina Principal - Home ############################################
@app.route('/newprofile', methods=['GET', 'POST'])
def newprofile():
    id_user = session["USERNAME"]
    data_user = all_users.val()[id_user]

    if request.method == 'POST':
        name = request.form['new_name']
        email = request.form['new_email']
        telefono = request.form['new_phone']
        ciudad = request.form['new_city']
        profesion = request.form['new_job']
        especialidad = request.form['new_spe']
        try:
            data_user['nombre'] = name.title()
            data_user['email'] = email
            data_user['telefono'] = telefono
            data_user['ciudad'] = ciudad.title()
            data_user['profesion'] = profesion.title()
            data_user['especialidad'] = especialidad.title()
            db.child("users").child(id_user).update(data_user)
            return render_template('/dashboard/profile_dashboard.html', data_user=data_user)
        except requests.exceptions.HTTPError:
            return render_template('/home/login_home.html', un=unsuccessful)
    return render_template('/dashboard/profile_dashboard.html', data_user=data_user)
######################################################################
