# Deploy using Kubernetes

Deploying the app in ../SVHN_Classifier through docker in to kubernetes

In Real-World general deployments requirements are:
- Properly packaged executable
- Launch nodes with executable running
- A service to redirect and schedule requests in to all the nodes
	
There are different tools to acheive this:
- We use the dockers to created a executable
- We use containers orcherstration tool Kubernetes to launch nodes and also to handle request routing to various nodes 

While Deploying a application when we need to run them on multiple nodes to support scaling e.t.c each nodes would be assigned a different IP and as our application interface through a particular port to send job to particular node we would need to use ip of node and this would be hard to manage and schedule when multiple nodes are involved. Conveniently Kubernet API lets you create a service that works as a gateway and automatically sends the requests in to different nodes automatically. This can be done using port-wording or expose through load balancer service.
	
Psuedo code:
-Kubernetes cluster is launched using minikube or other ways. 
-Create a representative namespace for deployment [optional]
-Create a Docker container that host's the deployment engine and consumes input through port
-Create a manifest file and launch the pods running the docker
-Use port-forwarding or expose kubectl functionality services to create a gateway point to receive and schedule requests to each pods automatically

## Create Dockerfile: 

Install docker following instructions from: https://docs.docker.com/get-docker/

File containing docker build configuration and to pull the code. App can be launches using docker following command same docker will be used even in kubernetes:

Build Docker image. Register and create a account in https://hub.docker.com/ and create a repository to push the image to cloud this make the image accessible remote nodes.

Use following commands to build the image from Dockerfile and push it to your repository
```
docker build -t dockerim . # docker build -t <tagname> .
docker tag <tagname>:latest <username>/<imagename>:<version>
docker push <username>/<imagename>:<version>
```


Command to run a docker
```
 docker run -it -e FLASK_PORT=7788 -p 7788:7788 rcv1k4s/deploying_cnns:0.1
```

This will start the app with flask port as 7788 and pipe the port outside of container to port 7788

## Install Kubernetes

Use this link for installing kubernetes:
- https://kubernetes.io/docs/tasks/tools/install-kubectl/#install-kubectl-binary-with-curl-on-linux

Use this link for installing minikube:
- https://kubernetes.io/docs/tasks/tools/install-minikube/
	
minikube lets you create a kubernetes cluster and through kuberneters 'kubectl' API you control the cluster to launch, scale, stop the application.

- After successfull installation of kubernetes and creation of cluster using minikube		
```
kubectl create namespace svhnclassifierflaskapp
kubectl get namespace # Verify the name space created
```

## Launch the Docker in to kubernetes
- Create deployment manifest file
  - see deploy_classifier.yaml
  - Link to understand manifest yaml: https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/

- Create a deployment using the manifest yaml file
```
kubectl apply -f deploy_classifier.yaml -n svhnclassifierflaskapp
kubectl apply -f <path to manifest yaml> -n <namespace>
```
- Verify if deployment is up
```
kubectl get deployment -n svhnclassifierflaskapp
```
- Forward kubenetes deployment port to host port. This exposes a port specified on the host and request sent to this forwarded port would be automatically redirected to individual nodes.
```
kubectl port-forward deployment/svhnclassifier-deploy -n svhnclassifierflaskapp 7788:7788
kubectl port-forward <entity type/entity name> -n <namespace> <sourceport>:<destinationport>
```
  - Ref: https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/
	
- Use Loadbalancer service to create a publicly usable IP and service to redirect requests sent to public ip to be allocated to pod node ports
```
kubectl expose deployment svhnclassifier-deploy --type=LoadBalancer --name=svhn-loadbalancer
kubectl expose deployment <deployment name> --type=LoadBalancer --name=<Name of service>
```
 - Ref: https://kubernetes.io/docs/tasks/access-application-cluster/create-external-load-balancer/

- Difference between expose with LoadBalancer and port-forwarding is with LoadBalancer we get a PublicIP accessible from outside
```
kubectl get service <name of service>
```
  - list's the PublicIP or kubectl describe can also be used

# Send a job to kubernetes pod
```
cd SVHN_Classifier/&&python3 send_image_to_flask_server.py
```
- Make sure you use appropriate port inside send_image_to_flask_server.py
	
	
