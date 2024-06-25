NAME:=hipie
VERSION:=24jun24
VERSION_TAG:=$(NAME):$(VERSION)

colon := :

docker-build:
		docker build -t "$(VERSION_TAG)" .

docker-run:
		docker run --expose 8887$(colon)8887 -v ./weights$(colon)/app/HIPIE/weights --name HIPIE --gpus '"device=3"' "$(VERSION_TAG)"

docker-run-interactive:
		docker run --expose 8887$(colon)8887 ./weights$(colon)/app/HIPIE/weights -it --name HIPIE --gpus '"device=3"' "$(VERSION_TAG)" /bin/bash

docker-clean:
		docker rm HIPIE
