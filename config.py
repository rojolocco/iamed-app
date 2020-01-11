class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'

    DB_NAME = "production-db"
    DB_USERNAME = "admin"
    DB_PASSWORD = "example"

    IMAGE_UPLOADS = "static/uploads"
    IMAGE_TEMP = "img_temp"
    ALLOW_IMAGES_EXTENSIONS = ["PNG", "JPG", "JPEG", "DICOM"]

    CLIENTS_IMAGE = "static\\clients\\image"
    CLIENTS_CSV = "static\\clients\\csv"
    CLIENTS_PDF = "static\\clients\\pdf"
    CLIENTS_REPORTS = "static\\clients\\reports"

    MODEL_DL = "static"

    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True

    DB_NAME = "development-db"
    DB_USERNAME = "admin"
    DB_PASSWORD = "example"

    IMAGE_UPLOADS = "static/uploads"
    ALLOW_IMAGES_EXTENSIONS = ["PNG", "JPG", "JPEG", "DICOM"]

    CLIENTS_IMAGE = "static\\clients\\image"
    CLIENTS_CSV = "static\\clients\\csv"
    CLIENTS_PDF = "static\\clients\\pdf"
    CLIENTS_REPORTS = "static\\clients\\reports"

    MODEL_DL = "static"

    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    TESTING = True

    DB_NAME = "development-db"
    DB_USERNAME = "admin"
    DB_PASSWORD = "example"

    SESSION_COOKIE_SECURE = False
