**********************
Blue Brain BioExplorer
**********************

.. image:: ../../../pythonsdk/doc/source/images/BBBE_banner.png

Description
###########
The Blue Brain BioExplorer (*BBBE*) is a tool for scientists to extract and analyse scientific data from visualization. *BBBE* is built on top of `Blue Brain Brayns <https://github.com/favreau/Brayns>`_, the Blue Brain rendering platform.

Architecture
############
The *BBBE* application is built on top of Brayns, the Blue Brain rendering platform. The role of the application is to use the underlying technical capabilities of the rendering platform to create large scale and accurate 3D scenes from Jupyter notebooks.

General components
##################

Assemblies
**********
Assemblies are groups of biological elements, such as proteins, membranes, glycans, etc. 
As an example, a virion is made of a lipid membrane, spikes proteins, an RNA sequence, etc, and all those elements belong to the same object. That’s why they need to belong to the same container, the assembly.
Assemblies can have different shapes: Sphere, Cube, etc, that are automatically generated according to the parameters of individual
components.

Proteins
********
Proteins are loaded from PDB files. Atoms, non-polymer chemicals and bonds can be loaded and displayed in various colour schemes: chain id, atom, residue, etc.
Proteins also contain the amino acid sequences of the individual chains. Sequences that can be used to query glycosylation sites, or functional regions of the protein.

Glycans
*******
Glycans are small proteins that are attached to an existing protein of the assembly. Individual glycan trees are loaded from PDB files and attached to the glycosylation sites of the specified protein. By default, glycans are attached to all available glycosylation sites, but a set of specific sites can be specified.

RNA sequence
************
An RNA sequence can be loaded from a text sequence of codons.
Various shapes can be selected to represent the RNA sequence: Trefoil knot, torus, star, etc. This allows the sequence to be efficiently packed into a given volume. A different color is assigned per type of codon.

Mesh-based membranes
********************
Mesh-based membranes create membranes based on 3D meshes. This allows the construction of complex membranes where mesh faces are filled with proteins.

Documentation
*************
.. toctree::
   :maxdepth: 1

   Classes <api/class_view_hierarchy.rst>
   Files <api/file_view_hierarchy.rst>
   Full API <api/library_root.rst>

Python SDK
##########
A simple API if exposed via the *BBBE* python library. The API allows scientists to easily create and modify assemblies, according the biological parameters. The *BBBE* programming language is not necessarily reflecting the underlying implementation, but is meant to be as simple as close as possible to the language used by the scientists to describe biological assemblies.

The *BBBE* Python SDK is available on `pypi <https://pypi.org/project/bioexplorer>`_

Deployment
##########
*BBBE* binaries are publicaly available as docker images. *BBBE* is designed to run in distributed mode, and is composed of 3 modules: A `server <https://hub.docker.com/r/bluebrain/bioexplorer>`_, a `python SDK <https://hub.docker.com/r/bluebrain/bioexplorer-python-sdk>`_, and a `web user interface <https://hub.docker.com/r/bluebrain/bioexplorer-ui>`_. This means that there are 3 docker images to be downloaded on run. Those images can of course run on different machines.

In this example, we will expose the server on port 5000, the python SDK jupyter notebooks on port 5001, and the user inferface on port 5002. One is free to change those ports at will.

Server
******

.. code-block:: bash

   docker run -ti --rm -p 5000:8200 bluebrain/bioexplorer

Python SDK
**********

.. code-block:: bash

   docker run -ti --rm -p 5001:8888 bluebrain/bioexplorer-python-sdk

Web User Interface
******************

.. code-block:: bash

   docker run -ti --rm -p 5002:8080 bluebrain/bioexplorer-ui

.. image:: ../../../pythonsdk/doc/source/images/BBBE_screenshot.png

Building from Source
####################

Blue Brain Brayns 1.0.0
***********************
In order to run the BioExplorer, it is necessary to build Blue Brain Brayns first. Please refer to the `documentation <https://github.com/favreau/Brayns/blob/master/README.md>`_ and following the steps in the "Building from source" paragraph. Note that the BioExplorer is currently only supported with `version 1.0.0 (e12fa84) <https://github.com/favreau/Brayns/releases/tag/1.0.1>`_ of `Blue Brain Brayns <https://github.com/favreau/Brayns>`_.

BioExplorer
***********

With `Blue Brain Brayns <https://github.com/favreau/Brayns>`_ compiled and installed in the <brayns_installation_folder>, run the statements to build the BioExplorer.

.. code-block:: bash

   git clone https://github.com/BlueBrain/BioExplorer.git
   mkdir build
   cd build
   CMAKE_PREFIX_PATH=<brayns_installation_folder> cmake .. -DCMAKE_INSTALL_PREFIX=<brayns_installation_folder>
   make install

The BioExplorer being a plug-in for `Blue Brain Brayns <https://github.com/favreau/Brayns>`_, the following commands can be used to start the backend:

.. code-block:: bash

   export LD_LIBRARY_PATH=<brayns_installation_folder>/lib:${LD_LIBRARY_PATH}
   export PATH=<brayns_installation_folder>/bin:${PATH}
   braynsService --http-server localhost:5000 --plugin BioExplorer

Simple example
**************
Considering that the *BBBE* server is running on the local host, on port 5000, the simplest example to visualize a coronavirus is:

.. code-block:: python

   from bioexplorer import BioExplorer
   be = BioExplorer('localhost:5000')
   resource_folder = '../../tests/test_files/'
   name='Coronavirus'
   be.add_coronavirus(name=name, resource_folder=resource_folder)

License
#######
*BBBE* is available to download and use under the GNU General Public License (`GPL <https://www.gnu.org/licenses/gpl.html>`_, or “free software”). The code is open sourced with approval from the open sourcing committee and principal coordinators of the Blue Brain Project in February 2021.

Contact
#######
For more information on *BBBE*, please contact:

| Cyrille Favreau
| Senior Scientific Visualization Engineer
| Blue Brain Project
| `cyrille.favreau@epfl.ch <cyrille.favreau@epfl.ch>`_

Funding & Acknowledgment
########################
The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

COPYRIGHT 2020–2021, Blue Brain Project/EPFL
