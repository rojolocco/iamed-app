# Cargando librerias ##################################################
from app import app
from app import db
from app import settings
from app.utils import allowed_image, tresholds, classes, patologies_preds
from app.utils import base64_encode_image, check_image
from app.utils import findings1, pat_only, other_only

import os
import uuid
import time
import json
import glob
from PIL import Image

from flask import render_template, request, session, jsonify

from fastai import *
from fastai.vision import *
#######################################################################


# Prediccion: chestXray ################################################
@app.route('/chestXray_pred', methods=['GET', 'POST'])
def chestXray_pred():

    upload = True
    data = {"success": False}

    if request.method == 'POST':
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
            input_file = open_image(path).resize(256) #Redis
            img_original = open_image(path) #model_server
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
                output = db.get(k) #PREDICCIONES DEL MODEL SERVER
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
            print(tresholds)    
            
            pred_list = []
            for index, elem in enumerate(classes):
                pred_list.append(outputs_list[index]/(2*tresholds[elem]))
            print(pred_list)

            def filterByKey(keys, pred_dict): 
                return {x: pred_dict[x] for x in keys}
            
            pred_dict = dict(zip(classes, pred_list))
            pat_dict = filterByKey(pat_only, pred_dict)
            other_dict = filterByKey(other_only, pred_dict)
            pred_tuple = sorted(
                pat_dict.items(), key=lambda x: x[1], reverse=True)
            print(pred_tuple)

            pat_preds, war_preds = patologies_preds(classes, tresholds, outputs_list)
            print(pat_preds)

            if len(pat_preds) == 0:
                pat_preds = 'No indicios de patologias presentes.'
            else:
                pat_preds = ', '.join(pat_preds)+'.'
            
            if 'Sin Hallazgo' in war_preds:
                war_preds.remove('Sin Hallazgo')

            if len(war_preds) == 0:
                    war_preds = 'No indicios de patologias presentes.'
            else:
                war_preds = ', '.join(war_preds)+'.'

            print(f'\nPredicciones procesadas!\n')

            return render_template('/dashboard/chestXray_dashboard.html',
                                    outputs=outputs_list, filename=filename,
                                    classes=classes, pat_preds=pat_preds,
                                    war_preds=war_preds, tresholds=tresholds,
                                    pred_tuple=pred_tuple, other_dict=other_dict)

            ############################################################

    main = True
    if not session.get("USERNAME") is None:
        print("Username found in session")
        return render_template('/dashboard/chestXray_dashboard.html', main=main)
    else:
        print("No username found in session")
        return render_template('/home/login_home.html', entrar=entrar)
#######################################################################
