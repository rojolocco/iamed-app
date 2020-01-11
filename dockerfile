# Use the Python3.7.2 image
#FROM python:3.7.2-stretch
FROM tiangolo/python-machine-learning:cuda9.1-python3.7

# update packages
RUN apt-get update -y 

RUN conda install -c conda-forge/label/gcc7 uwsgi


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app 
ADD . /app

# Install the dependencies
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Modelos chestXray
# https://drive.google.com/file/d/1Rx5crF94lJUaPVL6RAXy9dM8Grpw1yCX/view?usp=sharing
# https://drive.google.com/file/d/1BzhbILTXA0YgEu6NlNsE3cF4_n7KMNWm/view?usp=sharing one_model.zip
# RUN export FILEID='1Rx5crF94lJUaPVL6RAXy9dM8Grpw1yCX'
# RUN export FILENAME='all_models.zip'
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
    --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
    --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Rx5crF94lJUaPVL6RAXy9dM8Grpw1yCX' \
    -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Rx5crF94lJUaPVL6RAXy9dM8Grpw1yCX" \
    -O all_models.zip && rm -rf /tmp/cookies.txt

RUN unzip one_model.zip -d /app/app/all_models/chestXray

RUN rm -r one_model.zip

RUN python model_server.py &

# run the command to start uWSGI
CMD ["uwsgi", "app.ini"]