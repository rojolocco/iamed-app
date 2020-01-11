
# Cargando librerias ##################################################
from app import app
from app.utils import allowed_image, tresholds, classes, patologies_preds
from app.utils import base64_encode_image, check_image
from app import db
from app import settings
from app.utils import base64_encode_image, check_image

import os
from PIL import Image
from io import BytesIO
import numpy as np
import pyrebase

from flask import render_template, request, redirect, url_for, session, jsonify

from fastai import *
from fastai.vision import *

import sys
import uuid
import time
import json
import io
import glob
#########################################################################


# Configurando Firebase #################################################
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
########################################################################


# chestXray Api ########################################################
@app.route('/chestXray_api', methods=['GET', 'POST'])
def chestXray_api():

    data = {"success": False}

    if request.method == 'POST':

        # Verificando usuario autorizado ###############################
        email = request.authorization["username"]
        password = request.authorization["password"]

        try:
            response = auth.sign_in_with_email_and_password(email, password)
            if response['registered']:
                session["USERNAME"] = email
        except requests.exceptions.HTTPError:
            data["response"] = "Usuario no Autorizado"
            return jsonify(data)
        ##################################################################


        # Prediccion de imagen del api ###################################
        if request.files:

            # Inicializacion de variables ###############################
            image = request.files['image']
            root = 'app'
            k = str(uuid.uuid4())
            print(f'ID de la imagen:{k}')
            ############################################################

            # Borrar imagenes antiguas #################################
            print(f'\n1.) Borrando Imagenes Antiguas ...')
            files = glob.glob('app/static/uploads/chestXray/*.jpg')
            for f in files:
                os.remove(f)
            print(f'\nImagenes Borradas!')
            ############################################################

            # Guardar IDs de fotos en Redis #############################
            print(f'\n2.) Guardando Imagenes en Redis ...')
            path, filename = check_image(image, k)
            input_file = open_image(path).resize(256)
            img_original = open_image(path)
            img_original.save(os.path.join(root,
                                            app.config['IMAGE_TEMP'],
                                            'chestXray',
                                            f'{k}.jpg'))

            input_file_db = input_file.data.numpy()
            input_file_db = input_file_db.copy(order="C")
            input_file_db = base64_encode_image(input_file_db)
            d = {"id": k, "image": input_file_db}
            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
            print(f'\nImagenes Guardadas!')
            ############################################################

            # Busqueda continua de predicciones ########################
            print(f'\n3.) Buscando predicciones ...')
            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(settings.CLIENT_SLEEP)
            print(f'\nPredicciones encontradas!')
            ############################################################

            # Mostrar predicciones #####################################
            print(f'\n4.) Procesando las predicciones ...\n')
            data["success"] = True
            json_file = jsonify(data)

            outputs_list = []
            for i in range(len(data['predictions'])):
                outputs_list.append(data['predictions'][i]['probability'])
            print(outputs_list)

            pat_preds = patologies_preds(classes, tresholds, outputs_list)

            if len(pat_preds) == 0:
                pat_preds = 'No se encontro nada.'
            else:
                pat_preds = ', '.join(pat_preds)+'.'
            print(f'\nPredicciones procesadas!\n')
            ############################################################


        return jsonify(data)
#########################################################################
