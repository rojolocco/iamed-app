
# Cargando librerias ##################################################
from flask import Flask
from app import settings

import secrets
import redis
#######################################################################


# Crear aplicacion Flask ##############################################
app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_urlsafe(16)
#######################################################################


# Crear Base de Datos Redis ###########################################
db = redis.StrictRedis(host=settings.REDIS_HOST,
                        port=settings.REDIS_PORT, db=settings.REDIS_DB)
#######################################################################


# Configurar ambiente de desarrollo ####################################
if app.config["ENV"] == "production":
    app.config.from_object("config.ProductionConfig")
else:
    app.config.from_object("config.DevelopmentConfig")
#######################################################################


# Importar archivos python de rutas ###################################
from app import home_views
from app import dashboard_views
from app import apis
from app import pred_apps
from app import error_handler
from app import profile
#######################################################################
