ceo: Dockerfile
	cd ../; git archive --format=tar.gz HEAD > Docker/ceo.tar.gz
	sudo docker build -t gmto.im/ceo:devel_cuda10.1 .

cluster:
	eksctl create cluster --name DOS --region us-west-2 --nodegroup-name DOSlet --node-type p3.2xlarge --nodes 1 --nodes-min 0 --nodes-max 4 --node-ami ami-0ca5998dc2c88e64b
	kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta/nvidia-device-plugin.yml

test:
	kubectl apply -f ceo-test.yaml

clean:
	eksctl delete cluster --name DOS --region us-west-2

build: Dockerfile
	sudo docker build -t gmto.im/ceo:devel_cuda10.1 .

push_ohio:
	aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 378722409401.dkr.ecr.us-east-2.amazonaws.com
	docker tag gmto.im/ceo:devel_cuda10.1 378722409401.dkr.ecr.us-east-2.amazonaws.com/gmto.im/ceo:latest 
	docker push 378722409401.dkr.ecr.us-east-2.amazonaws.com/gmto.im/ceo:latest
