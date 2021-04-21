# Media Maker Python SDK

## Description
The Media Maker plugin provides a media generation API for [Blue Brain BioExplorer](https://github.com/BlueBrain/BioExplorer).

## API

### Snapshot

The following example illustrates how to connect to the Blue Brain BioExplorer and export a snapshot of the current view to disk. The snapshot is exported to the /tmp folder, with a resolution of 512x512, and with 16 samples per pixel.

```python
from bioexplorer import BioExplorer
from mediamaker import MovieMaker

bio_explorer = BioExplorer('localhost:5000')
movie_maker = MovieMaker(bio_explorer)

movie_maker.create_snapshot(
    path='/tmp/test.png',size=[512, 512], samples_per_pixel=16)
```

### Movie

The following example illustrates how to connect to the Blue Brain BioExplorer and generate a set of frames according to some given camera control points. Frames are exported to the /tmp folder.

```python
from bioexplorer import BioExplorer
from mediamaker import MovieMaker

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

# License
This software is available to download and use under the GNU General Public License ([GPL](https://www.gnu.org/licenses/gpl.html), or “free software”). The code is open sourced with approval from the open sourcing committee and principal coordinators of the Blue Brain Project in February 2021.

## Contact
For more information on the Media Maker plugin, please contact:

__Cyrille Favreau__  
[cyrille.favreau@epfl.ch](cyrille.favreau@epfl.ch) 

# Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

COPYRIGHT 2020–2021, Blue Brain Project/EPFL
