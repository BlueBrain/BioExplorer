<link href="./bioexplorer/core/doc/extra.css" rel="stylesheet"></link>

# Blue Brain BioExplorer

<table border=0>
<tr>
  <td>
    <a href="https://github.com/BlueBrain/BioExplorer/tags">
    <img src="https://img.shields.io/github/v/tag/BlueBrain/BioExplorer?style=for-the-badge" alt="Latest release" />
    </a>
  </td>
  <td>
    <a href="https://github.com/BlueBrain/BioExplorer/blob/master/LICENSE.md">
    <img src="https://img.shields.io/github/license/BlueBrain/BioExplorer?style=for-the-badge" alt="License" />
    </a>
  </td>
	<td>
		<a href="https://github.com/BlueBrain/BioExplorer/forks">
		<img src="https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fbluebrain%2Fbioexplorer%2Fbadge%3Fref%3Dmaster&style=for-the-badge" alt="Actions">
		</a>
	</td>
  <td>
    <a href="https://github.com/BlueBrain/BioExplorer/stargazers">
    <img src="https://img.shields.io/github/stars/BlueBrain/BioExplorer?style=for-the-badge" alt="Stars" />
    </a>
  </td>
  <td>
    <a href="https://github.com/BlueBrain/BioExplorer/network/members">
    <img src="https://img.shields.io/github/forks/BlueBrain/BioExplorer?style=for-the-badge" alt="Forks" />
    </a>
  </td>
	<td>
		<a href="http://www.pydocstyle.org/">
		<img src="https://img.shields.io/badge/docstrings-pydocstyle-informational?style=for-the-badge" alt="Pydocstyle">
		</a>
	</td>
	<td>
		<a href="https://pypi.org/project/pycodestyle/">
		<img src="https://img.shields.io/badge/docstrings-pycodestyle-informational?style=for-the-badge" alt="Pycodestyle">
		</a>
	</td>
</tr>
</table>

![___](./bioexplorer/pythonsdk/doc/source/images/BBBE_banner.png)

## Description
In the context of the "[Elevated blood glucose levels as a primary risk factor for the severity of COVID-19](https://www.medrxiv.org/content/10.1101/2021.04.29.21256294v1)" study, the Blue Brain BioExplorer (_BBBE_) started as an internal project with the aim to answer key scientific questions related to the Coronavirus as a use case. This project aimed to deliver a visualization tool, the BioExplorer, to reconstruct, visualize, explore and describe in detail the structure and function of the Coronavirus.

<div align="center">
      <a href="https://www.youtube.com/watch?v=hkgqG0nzW9I">
         <img src="https://img.youtube.com/vi/hkgqG0nzW9I/0.jpg" style="width:100%;">
      </a>
</div>

## Architecture
The _BBBE_ application is built on top of [Blue Brain Brayns](https://github.com/BlueBrain/Brayns), the Blue Brain rendering platform. The _BBBE_ uses the underlying technical capabilities of the rendering platform to create large scale and accurate 3D scenes from Jupyter notebooks.

![___](./bioexplorer/pythonsdk/doc/source/images/architecture.png)

## General components

### Assemblies
Assemblies are groups of biological elements, such as proteins, membranes, glycans, etc. 
As an example, a virion is made of a lipid membrane, spikes proteins, an RNA sequence, etc, and all those elements belong to the same object. That’s why they need to belong to the same container, the assembly.
Assemblies can have different shapes: Sphere, Cube, etc, that are automatically generated according to the parameters of individual
components.

### Proteins
Proteins are loaded from PDB files. Atoms, non-polymer chemicals and bonds can be loaded and displayed in various colour schemes: chain id, atom, residue, etc.
Proteins also contain the amino acid sequences of the individual chains. Sequences that can be used to query glycosylation sites, or functional regions of the protein.

![___](./bioexplorer/pythonsdk/doc/source/images/6vyb_regions.gif)

### Glycans
Glycans are small proteins that are attached to an existing protein of the assembly. Individual glycan trees are loaded from PDB files and attached to the glycosylation sites of the specified protein. By default, glycans are attached to all available glycosylation sites, but a set of specific sites can be specified.

Glycan trees models located in the python sdk test folder were generated with [Glycam Builder](http://glycam.org).

![___](./bioexplorer/pythonsdk/doc/source/images/receptor_all_glycans.gif)

### RNA sequence (work in progress)
An RNA sequence can be loaded from a text sequence of codons.
Various shapes can be selected to represent the RNA sequence: Trefoil knot, torus, star, etc. This allows the sequence to be efficiently packed into a given volume. A different color is assigned per type of codon.

### Mesh-based membranes
Mesh-based membranes create membranes based on 3D meshes. This allows the construction of complex membranes where mesh faces are filled with proteins.

### Virus
A viral particle (= “virus”) is an assembly consisting of a membrane, an RNA sequence, and a given number of S, M and E proteins. The virus has a predefined spherical shape defined by its radius. The default parameters for the virus are a radius of 45 nanometers, 62 S proteins, 42 E proteins, and 50 M proteins. Dimensions and concentrations were retrieved from the literature.

![___](./bioexplorer/pythonsdk/doc/source/images/coronavirus.gif)

### Membrane

A membrane is an assembly of phospholipids. Phospholipids structures are created following the process described in the [VMD](https://www.ks.uiuc.edu/Research/vmd) Membrane Proteins [tutorial](http://www.ks.uiuc.edu/Training/Tutorials). The assembly itself is generated by the BioExplorer, for a given shape, and a number of instances of phospholipids.

![___](./bioexplorer/pythonsdk/doc/source/images/membrane.gif)

## Python SDK
A simple API if exposed via the _BBBE_ python library. The API allows scientists to easily create and modify assemblies, according the biological parameters. The _BBBE_ programming language is not necessarily reflecting the underlying implementation, but is meant to be as simple as close as possible to the language used by the scientists to describe biological assemblies.

The _BBBE_ Python SDK is available on [pypi](https://pypi.org/project/bioexplorer/).

A large number of examples (as python notebooks) are provided in the [notebooks](https://github.com/BlueBrain/BioExplorer/tree/master/bioexplorer/pythonsdk/notebooks) folder.

## Documentation

See [here](https://bluebrain.github.io/BioExplorer/) for detailed documentation of the source code.

## Deployment

_BBBE_ binaries are publicaly available as docker images. _BBBE_ is designed to run in distributed mode, and is composed of 3 modules: A [server](https://hub.docker.com/r/bluebrain/bioexplorer), a [python SDK](https://hub.docker.com/r/bluebrain/bioexplorer-python-sdk), and a [web user interface](https://hub.docker.com/r/bluebrain/bioexplorer-ui). This means that there are 3 docker images to be downloaded on run. Those images can of course run on different machines.

In this example, we will expose the server on port 5000, the python SDK jupyter notebooks on port 5001, and the user inferface on port 5002. One is free to change those ports at will.

### Server

```bash
docker run -ti --rm -p 5000:8200 bluebrain/bioexplorer
```

### Python SDK

```bash
docker run -ti --rm -p 5001:8888 bluebrain/bioexplorer-python-sdk
```

### Web User Interface

```bash
docker run -ti --rm -p 5002:8080 bluebrain/bioexplorer-ui
```

![___](./bioexplorer/pythonsdk/doc/source/images/BBBE_screenshot.png)

## Building from Source

### Blue Brain Brayns 1.0.0
In order to run the BioExplorer, it is necessary to build Blue Brain Brayns first. Please refer to the [documentation](https://github.com/favreau/Brayns/blob/master/README.md) and following the steps in the "Building from source" paragraph. Note that the BioExplorer is currently only supported with [version 1.0.0 (e12fa84)](https://github.com/favreau/Brayns/releases/tag/1.0.1) of [Blue Brain Brayns](https://github.com/favreau/Brayns/).

### BioExplorer

#### Compile
With [Blue Brain Brayns](https://github.com/favreau/Brayns/) compiled and installed in the <brayns_installation_folder>, run the statements to build the BioExplorer.

```bash
git clone https://github.com/BlueBrain/BioExplorer.git
mkdir build
cd build
CMAKE_PREFIX_PATH=<brayns_installation_folder> cmake .. -DCMAKE_INSTALL_PREFIX=<brayns_installation_folder>
make install
```

#### Run

The BioExplorer being a plug-in for [Blue Brain Brayns](https://github.com/favreau/Brayns/), the following commands can be used to start the backend:

```bash
export LD_LIBRARY_PATH=<brayns_installation_folder>/lib:${LD_LIBRARY_PATH}
export PATH=<brayns_installation_folder>/bin:${PATH}
braynsService --http-server localhost:5000 --plugin BioExplorer
```

## Simple example
Considering that the _BBBE_ server is running on the local host, on port 5000, the simplest example to visualize a coronavirus is:
```python
from bioexplorer import BioExplorer
be = BioExplorer('localhost:5000')
resource_folder = '../../tests/test_files/'
name='Coronavirus'
be.add_coronavirus(name=name, resource_folder=resource_folder)
```

# License
_BBBE_ is available to download and use under the GNU General Public License ([GPL](https://www.gnu.org/licenses/gpl.html), or “free software”). The code is open sourced with approval from the open sourcing committee and principal coordinators of the Blue Brain Project in February 2021.


# Contact
For more information on _BBBE_, please contact:

__Cyrille Favreau__  
Senior Scientific Visualization Engineer  
Blue Brain Project  
[cyrille.favreau@epfl.ch](cyrille.favreau@epfl.ch) 

# Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

COPYRIGHT 2020–2021, Blue Brain Project/EPFL
