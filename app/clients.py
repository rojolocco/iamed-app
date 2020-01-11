
# Cargando librerias ##################################################
from app import app
from flask import send_from_directory, abort
#######################################################################


# Cargando librerias ##################################################
@app.route('/get_image/<filename>')
def get_image(filename):
    try:
        return send_from_directory(app.config['CLIENTS_IMAGE'],
                                    filename=filename,
                                    as_attachment=True)
    except FileNotFoundError:
        abort(404)
#######################################################################


# Cargando librerias ##################################################
@app.route('/get_csv/<filename>')
def get_csv(filename):
    try:
        return send_from_directory(app.config['CLIENTS_CSV'],
                                    filename=filename,
                                    as_attachment=True)
    except FileNotFoundError:
        abort(404)
#######################################################################


# Cargando librerias ##################################################
@app.route('/get_pdf/<filename>')
def get_pdf(filename):
    try:
        return send_from_directory(app.config['CLIENTS_PDF'],
                                    filename=filename,
                                    as_attachment=True)
    except FileNotFoundError:
        abort(404)
#######################################################################


# Cargando librerias ##################################################
@app.route('/get_reports/<path:path>')
def get_reports(path):
    try:
        return send_from_directory(app.config['CLIENTS_IMAGE'],
                                    path=path,
                                    as_attachment=True)
    except FileNotFoundError:
        abort(404)
#######################################################################
