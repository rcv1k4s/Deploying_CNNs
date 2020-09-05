From tensorflow/tensorflow:1.14.0-py3

RUN apt update&& apt install git libsm6 libxrender1 libxext6 -y --no-install-recommends
RUN pip3 install requests flask opencv-python==3.4.0.12 scikit-learn scikit-image 
RUN git clone https://github.com/rcv1k4s/Deploying_CNNs

RUN pip3 install numpy==1.18.1

EXPOSE 7788
ENV FLASK_PORT "7788"

CMD cd Deploying_CNNs/&&git pull origin develop&&cd SVHN_Classifier/&&python3 deploy_flask_server.py
