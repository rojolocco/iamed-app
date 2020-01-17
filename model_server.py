
# Cargando librerias ##########################################
from app import app
from app import settings
from app.utils import base64_decode_image, pred_mean
from app.arch.chestXray.misc.functions import get_all_models, get_all_data
from app.arch.chestXray.misc.functions import pre_single_model

import redis
import time
import json
import numpy as np
#from PIL import Image
import csv
import timeit
import os
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

print('\n######################')
print('MODEL SERVER CHESTXRAY')
print('######################')
########################################################################


# Conectando con Redis server ##########################################
print('\n1.) Iniciando Base de Datos de Redis ...')
db = redis.StrictRedis(host=settings.REDIS_HOST, 
                        port=settings.REDIS_PORT, db=settings.REDIS_DB)
print('\n  Base de Datos de Redis Iniciada!')
#######################################################################


# Iniciando Variables #################################################
cudnn.benchmark = True
patologies = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                'Pleural Other', 'Fracture', 'Support Devices']
#######################################################################


# Directorios de imagenes y modelos ###################################
cudnn.benchmark = True
TEST_IMAGE_LIST = "app/img_list/chestXray/image_path.csv"
models_dict = "app/all_models/chestXray"
#######################################################################


# Conectando con Redis server ##########################################
print('\n2.) Cargando modelos entrenados ...')
model, pNas_model, SA_model, pNasSA_model, XcepFPN_model, xception_gcn = get_all_models()
print('\n  Modelos Cargados!')
########################################################################


# Funcion Principal del Server #########################################
def classify_process():

    # Busqueda continua de imagenes para clasificar ####################
    while True:
        
        # Tomando batch de imagenes ###################################
        queue = db.lrange(settings.IMAGE_QUEUE, 0,
                            settings.BATCH_SIZE - 1)

        imageIDs = []
        batch = None
        print(f'\nEl batch actual es: {batch}. Esperando por fotos!')
        ########################################################################


        # Identificar IDs de las imagenes ######################################
        for q in queue:

            # Obteniendo la imagen de la base de datos de Redis
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"],
                                        settings.IMAGE_DTYPE,
                                        (1, settings.IMAGE_HEIGHT, 
                                            settings.IMAGE_WIDTH,
                                            settings.IMAGE_CHANS))

            # revisar si el batch es None
            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])

            # Actualizar la lista de imagenes
            imageIDs.append(q["id"])
            
            print(f'\nLos IDs de las imagenes actuales son: {imageIDs}')
        ########################################################################

        
        # Hacer las predicciones de las imagenes ###############################
        if batch is not None:
            
            # Guardando imagenes con sus IDs y Path en .csv ####################
            print(f'\n3.) Creando archivo csv con path y nombre de imagenes')
            row_list = [["Path"]]
            for i in imageIDs:
                row_list.append([f'app/img_temp/chestXray/{i}.jpg'])

            with open('app/img_list/chestXray/image_path.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
            print(f'\n  Archivo csv Creado!')
            ####################################################################

            # Creando Pytorch Dataloder con el archivo csv #####################
            print(f'\n4.) Creando Pytorch Dataloaders')
            valid_dataloder, Nas_dataloder, Atel_dataloader, dataset_loader = get_all_data(
                TEST_IMAGE_LIST)
            # valid_dataloder = get_all_data(TEST_IMAGE_LIST)
            print(f'\n  Dataloaders Creados!')
            ####################################################################

            # Creando Diccionario con los datos de modelos e imagenes ##########

            model_list = {'u2_file': [model,         valid_dataloder, '0.8990708733759138.pth'],
                            'u5_file': [pNas_model,    Nas_dataloder,   '0.8933122711473276.pth'],
                            'u6_file': [pNas_model,    Nas_dataloder,   '0.8918464376816251.pth'],
                            'u7_file': [pNas_model,    Nas_dataloder,   '0.8685262236170553.pth'],
                            'u8_file': [SA_model,      valid_dataloder, 'xcepdualexpsimat.pth'],
                            'u9_file': [SA_model,      valid_dataloder, 'xcepdualsumsimat.pth'],
                            'u10_file': [SA_model,      valid_dataloder, '0.8960423659716028.pth'],
                            'u11_file': [SA_model,      valid_dataloder, '0.8901343376820208.pth'],
                            'u12_file': [pNasSA_model,  Nas_dataloder,   '0.885028055543075.pth'],
                            'u13_file': [SA_model,      valid_dataloder, 'xcupdualesimscaleat.pth'],
                            'u14_file': [SA_model,      Atel_dataloader, 'xcupdualesimscaleat.pth'],
                            'u15_file': [model,         valid_dataloder, '0.8877043778295596.pth'],
                            'u16_file': [XcepFPN_model, valid_dataloder, '0.8743647271509183.pth'],
                            'u17_file': [XcepFPN_model, valid_dataloder, '0.8932593746041071.pth'],
                            'u18_file': [SA_model,      valid_dataloder, '0.8997462889772881.pth'],
                            'u19_file': [SA_model,      valid_dataloder, '0.8988146931955356.pth'],
                            'u20_file': [XcepFPN_model, valid_dataloder, '0.8952064212757506.pth'],
                            'u21_file': [SA_model,      dataset_loader,  'DR_cls_index_227.pth'],
                            'u22_file': [SA_model,      dataset_loader,  'DR_cls_index_113.pth'],
                            'u23_file': [xception_gcn,  dataset_loader,  'DR_cls_0.869_0.909.pth']
                            }

        ########################################################################


        # Revizar si es necesario clasificar imagenes ##########################
        if len(imageIDs) > 0:

            print(f"\nEl tamaño del Batch es: {batch.shape}")
            print(f"\nSe tiene(n) {batch.shape[0]} imagenes para clasificar!")

            # Clasificacion de imagenes nuevas #################################
            y_predU_list = {}
            print(f'\n5.) Haciendo Predicciones de las imagenes ...')

            start = timeit.default_timer()
            
            for k, v in model_list.items():
                value = k.split("_")[0][1:]
                model_file = os.path.join(models_dict, v[2])
                y_predU = pre_single_model(v[0], v[1], model_file)
                y_predU_list.update({f'y_predU{value}': y_predU})
            
            stop = timeit.default_timer()
            print(f"\nTiempo de Prediccion Inicial: {stop - start}\n")
            
            #print(y_predU_list.keys())

            results = pred_mean(y_predU_list)
            print(f'\nPredicciones Hechas!')
            ######################################################################


            # Insertas resultados en la base de datos de Redis ###################
            for (imageID, resultSet) in zip(imageIDs, results):
                
                print(f'\n6.) Enviando resultados a Redis DB ...')
                output = []

                for (label, prob) in zip(patologies,resultSet):
                    r = {"label": label, "probability": prob}
                    output.append(r)

                print(f'\nEnviando Resultados de la imagen: {imageID}\n')
                print(output)
                print(f'\nResultados Enviados!')
                db.set(imageID, json.dumps(output))
                
            ######################################################################


            # Borrar imagenes ya clasificadas ####################################
            print(f'\n7.) Borrar imagenes Clasificadas de Redis ...')
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
            

            files = glob.glob('app/img_temp/chestXray/*.jpg')
            for f in files:
                os.remove(f)
            print(f'\nImagenes Borradas de Redis!')
            print(f'\nPROCESO TERMINADO ESPERANDO NUEVAS IMAGENES!')
            ######################################################################

        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)
        
if __name__ == "__main__":
    classify_process()
