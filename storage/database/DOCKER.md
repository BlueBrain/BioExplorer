# Blue Brain BioExplorer PostgreSQL Database with Docker

> Use [Docker](https://docs.docker.com) to run Blue Brain BioExplorer PostgreSQL Database as a service and avoid painful tooling setup.

### Prerequisites
-----------------
Head over to [Docker](https://docs.docker.com/engine/installation/#supported-platforms) and install Docker for your own platform.

### Setup
---------
First build the image (*it's necessary to do this step if you want to run BioExplorer*):
```bash
docker build -t bioexplorer-db .
```

### Usage
---------
If you want to run Blue Brain BioExplorer PostgreSQL Database use:
```bash
# Runs Blue Brain BioExplorer Database on port 5432
docker run -d -p 5432:5432 bioexplorer-db
```

### Populating the database

A [vasculature dataset](https://bbp.epfl.ch/ngv-portal/data/anatomy/experimental-data/vasculature-data/raw-vasculature-data.vtk) is available from the Blue Brain NGV portal. An example [Python Notebook](../../bioexplorer/pythonsdk/notebooks/vasculature/BioExplorer_import_vtk_to_db.ipynb) is available to import the VTK dataset into the database. The [vasculature Python Notebook](../../bioexplorer/pythonsdk/notebooks/vasculature/BioExplorer_vasculature.ipynb) can then be used to visualize the vasculature.

**NOTE** If you are having trouble exiting the process after you run the container (with the above command), use `docker stop <container-id>` to stop the container.
`docker ps` will give you the current running process.
