# Deploying_CNNs

## Repository contains following modules:
- Script to Train a simple CNN to classify SVHN dataset
- Create a frozen graph from trained model
- Use frozen graph to create a inference routing
- Use frozen graph through Flask API server to be able to send image to API and get result
- Use docker to package and create a executable that launches Flask inference server and create a docker run deployment.
- Use the docker file to integrate to kubernetes to create a scalable deployment

## References:
- Kubernetes: https://kubernetes.io/docs/tutorials/kubernetes-basics/
- Dockers: https://docs.docker.com/get-docker/
- Tensorflow: https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs
- SVHN Dataset: http://ufldl.stanford.edu/housenumbers/


Detailed steps are in individual directories SVHN_Classifier/Readme.MD and deploy_kubernetes/Readme.MD

