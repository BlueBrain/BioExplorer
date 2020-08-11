# BioExplorer

The bioexplorer package provides an extended python API for the Blue Brain BioExplorer application

### 1. From the Python Package Index

```
(venv)$ pip install bioexplorer
```

### 2. From source

Clone the repository and install it:

```
(venv)$ git clone https://github.com/BlueBrain/BioExplorer.git
(venv)$ pip install -e ./bioexplorer
```

## Connect to running Brayns instance

```python
>>> from brayns import Client
>>> from bioexplorer import BioExplorer

>>> brayns = Client('localhost:8200')
>>> bio_explorer = BioExplorer(brayns)
```

# Upload to pypi

```bash
twine upload dist/*
```
