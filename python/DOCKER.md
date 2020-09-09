# Blue Brain BioExplorer Python SDK with Docker

> Use [Docker](https://docs.docker.com) to run Blue Brain BioExplorer Python SDK as a service and avoid painful tooling setup.


### Prerequisites
-----------------
Head over to [Docker](https://docs.docker.com/engine/installation/#supported-platforms) and install Docker for your own platform.


### Setup
---------
First build the image (*it's necessary to do this step if you want to run Brayns*):
```bash
docker build . -t bioexplorer-python-sdk
```


### Usage
---------
If you want to run Blue Brain BioExplorer Python SDK use:
```bash
# Runs Blue Brain BioExplorer Python SDK as a service with the HTTP interface binded on port 5000
docker run -ti --rm -p 8888:8888 bioexplorer-python-sdk
```

**NOTE** If you are having trouble exiting the process after you run the container (with the above command), use `docker stop <container-id>` to stop the container.
`docker ps` will give you the current running process.
