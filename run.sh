docker run --rm --interactive --tty --volume="$(pwd)":"$(pwd)" --workdir="$(pwd)" \
           --user="$(id -u)":"$(id -g)" --net=host \
           carlosgalvezp/fcnd_term1:latest python controls_flyer.py
