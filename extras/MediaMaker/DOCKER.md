# Media Maker with Docker

> Use [Docker](https://docs.docker.com) to run Media Maker as a service and avoid painful tooling setup.


### Prerequisites
-----------------
Head over to [Docker](https://docs.docker.com/engine/installation/#supported-platforms) and install Docker for your own platform.


### Setup
---------
First build the image (*it's necessary to do this step if you want to run Brayns*):
```bash
docker build . -t mediaMaker
```


### Usage
---------
By default, the entrypoint when running the image is `braynsService`, but if you want to ssh into the container use:
```bash
# `-p 5000:5000` is used only to provide some port bindings (host:container) if you want to run and access Brayns from your host while in the container
docker run -ti --rm --entrypoint bash -p 5000:5000 mediaMaker
```

If you want to run Media Maker plugin use:
```bash
# Runs Media Maker plugin as a service with the HTTP interface binded on port 5000
docker run -ti --rm -p 5000:5000 mediaMaker
```
