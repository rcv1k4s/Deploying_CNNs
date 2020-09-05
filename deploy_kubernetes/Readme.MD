# Deploy using Kubernetes
	- Deploying the app in ../SVHN_Classifier through docker in to kubernetes

## Dockerfile: File containing docker build configuration and to pull the code. App can be launches using docker following commands same docker will be used even in kubernetes
	- docker run -it -e FLASK_PORT=7788 -p 7788:7788 rcv1k4s/deploying_cnns:0.1 
	- This will start the app with flask port as 7788 and pipe the port outside of container to port 7788

## Install Kubernetes
	- https://kubernetes.io/docs/tasks/tools/install-kubectl/#install-kubectl-binary-with-curl-on-linux

	- After successfull installation of kubernetes
		- kubectl create namespace svhnclassifierflaskapp
		- kubectl get namespace # Verify the name space created
## Launch the Docker in to kubernetes
	- Create deployment manifest file
		- see deploy_classifier.yaml
	- Create a deployment using the manifest yaml file
		- kubectl apply -f deploy_classifier.yaml -n svhnclassifierflaskapp
	- Verify if deployment is up
		- kubectl get deploy -n svhnclassifierflaskapp
	- Forward kubenetes port to host port
		- kubectl port-forward deployment/svhnclassifier-deploy -n svhnclassifierflaskapp 7788:7788

# Send a job to kubernetes pod
	- cd SVHN_Classifier/
	- python3 send_image_to_flask_server.py
	- Make sure you use appropriate port inside send_image_to_flask_server.py
	
	
