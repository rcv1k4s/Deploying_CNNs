# Deploy using Kubernetes
	- Deploying the app in ../SVHN_Classifier through docker in to kubernetes

## Dockerfile: File containing docker build configuration and to pull the code. App can be launches using following commands.
	- docker run -it -e FLASK_PORT=7788 -p 7788:7788 rcv1k4s/deploying_cnns:0.1 
	- This will start the app with flask port as 7788 and pipe the port outside of container to port 7788
