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
If you want to run Blue Brain BioExplorer PostgreSQL Database, use the following command to run the database instance:
```bash
# Runs Blue Brain BioExplorer Database on port 5432
docker run -d -p 5432:5432 bioexplorer-db
```

And run the Blue Brain BioExplorer with the following command:
```bash
# Runs the Blue Brain BioExplorer as a service with the HTTP interface bound to port 5000
docker run -ti --rm -p 5000:5000 bioexplorer --http-server :5000 --plugin "BioExplorer --db-name=bioexplorer --db-password=bioexplorer --db-user=postgres --db-host=<ip address> --db-port=5432" --plugin MediaMaker
```

Replace `<ip address>` with the IP address of the host where the database docker container is running.

### Populating the database with open datasets

A [full circuit dataset](https://zenodo.org/record/6906785#.Ywym7tVBxH6) is available in the [Model of Rat Non-barrel Somatosensory Cortex Anatomy](https://www.biorxiv.org/content/10.1101/2022.08.11.503144v1) publication. A [Python Notebook](../../bioexplorer/pythonsdk/notebooks/neurons/rat_non-barrel_somatosensory_cortex_anatomy/BioExplorer_import_sonata_to_db.ipynb) has been included with basic examples of how to explore the dataset.

A [vasculature dataset](https://bbp.epfl.ch/ngv-portal/data/anatomy/experimental-data/vasculature-data/raw-vasculature-data.vtk) is available from the Blue Brain NGV portal. An example [Python Notebook](../../bioexplorer/pythonsdk/notebooks/vasculature/BioExplorer_import_vtk_to_db.ipynb) is available to import the VTK dataset into the database. The [vasculature Python Notebook](../../bioexplorer/pythonsdk/notebooks/vasculature/BioExplorer_vasculature.ipynb) can then be used to visualize the vasculature.


**NOTE** If you are having trouble exiting the process after you run the container (with the above command), use `docker stop <container-id>` to stop the container.
`docker ps` will give you the current running process.
