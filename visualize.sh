#!/bin/bash
docker run --rm --interactive --volume="$(pwd)":"$(pwd)" --workdir="$(pwd)" \
           --net=host \
           carlosgalvezp/fcnd_term1:latest python -m visdom.server

