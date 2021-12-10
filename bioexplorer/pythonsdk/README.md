# Blue Brain BioExplorer Python SDK

The bioexplorer package provides an extended python API for the Blue Brain BioExplorer application

## Installation

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

## API

### Connect to running Blue Brain BioExplorer instance

```python
>>> from bioexplorer import BioExplorer
>>> bio_explorer = BioExplorer('localhost:8200')
```

### Snapshot

The following example illustrates how to connect to the Blue Brain BioExplorer and export a snapshot of the current view to disk. The snapshot is exported to the /tmp folder, with a resolution of 512x512, and with 16 samples per pixel.

```python
from bioexplorer import BioExplorer, MovieMaker

bio_explorer = BioExplorer('localhost:5000')
movie_maker = MovieMaker(bio_explorer)

movie_maker.create_snapshot(
    renderer='bio_explorer', path='/tmp', base_name='test', ,size=[512, 512], samples_per_pixel=16)
```

### Movie

The following example illustrates how to connect to the Blue Brain BioExplorer and generate a set of frames according to some given camera control points. Frames are exported to the /tmp folder.

```python
from bioexplorer import BioExplorer, MovieMaker

bio_explorer = BioExplorer('localhost:5000')
movie_maker = MovieMaker(bio_explorer)

control_points = [
    {
        'apertureRadius': 0.0,
        'direction': [0.0, 0.0, -1.0],
        'focusDistance': 0.0,
        'origin': [0.5, 0.5, 1.5],
        'up': [0.0, 1.0, 0.0]
    },
    {
        'apertureRadius': 0.0,
        'direction': [-0.482, -0.351, -0.802],
        'focusDistance': 0.0,
        'origin': [2.020, 1.606, 3.030],
        'up': [-0.199, 0.936, -0.289]
    }
]

movie_maker.build_camera_path(
    control_points=control_points, nb_steps_between_control_points=50,
    smoothing_size=50)

movie_maker.set_current_frame(10)
movie_maker.create_movie(
    path='/tmp', size=[512, 512], samples_per_pixel=16, start_frame=10, end_frame=20)
```

# Upload to pypi

```bash
twine upload dist/*
```
