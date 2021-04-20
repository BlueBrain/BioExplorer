# Blue Brain BioExplorer Python SDK

The bioexplorer package provides an extended python API for the Blue Brain BioExplorer application

### 1. From the Python Package Index

```
(venv)$ pip install bioexplorer
```

### 2. From source

Clone the repository and install it:

```
(venv)$ git clone https://github.com/BlueBrain/BioExplorer.git
(venv)$ cd BioExplorer/bioexplorer/pythonsdk
(venv)$ python setup.py install
```

## Connect to running Blue Brain BioExplorer instance

```python
>>> from bioexplorer import BioExplorer
>>> bio_explorer = BioExplorer('localhost:8200')
```

# Upload to pypi

```bash
twine upload dist/*
```
