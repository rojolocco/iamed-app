
# Inicializar Configuraciones Conexion Redis ########################
#REDIS_HOST = "redis" #PARA DOCKER
REDIS_HOST = "127.0.0.1" #PARA FLASK
REDIS_PORT = 6379
REDIS_DB = 0
#####################################################################


# Inicializar Configuraciones de Imagesnes #########################
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
#####################################################################


# Inicializar Constantes Conexion Redis #############################
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 8
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25
#####################################################################
