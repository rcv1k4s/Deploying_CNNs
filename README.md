# Deploying_CNNs

Contains modules to train a simple CNN model to classify on SVHN dataset using tensorflow and deploy using kubernetes through docker.

## Repo Setup

Dependencies Installation on Ubuntu-20.04

```
sudo apt install python3 python3-pip -y&&python3 -m pip install --upgrade pip&&python3 -m pip install poetry==1.3.0
```

Install packages

```
poetry install -E .dev
```

## Repository contains following modules:
- Train a simple CNN to classify SVHN dataset
- Create a frozen graph from trained model
- Use frozen graph to create a inference routine.
- Use frozen graph through Flask API server to be able to send image to API and get result.
- Use docker to package and create a executable that launches Flask inference server and create a docker run deployment.
- Use the docker deployment into kubernetes to create a scalable deployment

## References:
- Kubernetes: https://kubernetes.io/docs/tutorials/kubernetes-basics/
- Dockers: https://docs.docker.com/get-docker/
- Tensorflow: https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs
- SVHN Dataset: http://ufldl.stanford.edu/housenumbers/


Detailed steps are in individual directories SVHN_Classifier/Readme.MD and deploy_kubernetes/Readme.MD

