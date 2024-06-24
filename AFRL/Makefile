NAME:=hipie
VERSION:=24jun24
VERSION_TAG:=$(NAME):$(VERSION)


docker-build:
		docker build -t "$(VERSION_TAG)" .

docker-run:
		docker run -it --name HIPIE --gpus '"device=3"' "$(VERSION_TAG)" /bin/bash

docker-clean:
		docker rm HIPIE