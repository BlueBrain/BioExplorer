# !/usr/bin/env python
"""BioExplorer class"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2021 Blue BrainProject / EPFL
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

import math

from pyquaternion import Quaternion

import seaborn as sns

from brayns import Client
from .transfer_function import TransferFunction
from .version import VERSION as __version__

# pylint: disable=no-member
# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=inconsistent-return-statements
# pylint: disable=wrong-import-order
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=missing-return-type-doc
# pylint: disable=missing-return-doc
# pylint: disable=missing-raises-doc


class Vector3:
    """A Vector3 is an array of 3 floats representing a 3D vector"""

    def __init__(self, *args):
        """
        Define a simple 3D vector

        :args: 3 float values for x, y and z
        :raises: RuntimeError: Invalid number of floats
        """
        if len(args) not in [0, 3]:
            raise RuntimeError("Invalid number of floats (0 or 3 expected)")

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        if len(args) == 3:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]

    def to_list(self):
        """
        A list containing the values of x, y and z attributes

        :return: x, y and z attributes
        :rtype: list
        """
        return [self.x, self.y, self.z]


class Vector2:
    """A Vector2 is an array of 2 floats representing a 2D vector"""

    def __init__(self, *args):
        """
        Define a simple 2D vector

        :args: 2 float values for x and y
        :raises: RuntimeError: Invalid number of floats
        """
        if len(args) not in [0, 2]:
            raise RuntimeError("Invalid number of floats (0 or 2 expected)")

        self.x = 0.0
        self.y = 0.0
        if len(args) == 2:
            self.x = args[0]
            self.y = args[1]

    def to_list(self):
        """:return: A list containing the values of x and y attributes"""
        return [self.x, self.y]


class BioExplorer:
    """Blue Brain BioExplorer"""

    PLUGIN_API_PREFIX = "be-"

    POSITION_RANDOMIZATION_TYPE_CIRCULAR = 0
    POSITION_RANDOMIZATION_TYPE_RADIAL = 1

    COLOR_SCHEME_NONE = 0
    COLOR_SCHEME_ATOMS = 1
    COLOR_SCHEME_CHAINS = 2
    COLOR_SCHEME_RESIDUES = 3
    COLOR_SCHEME_AMINO_ACID_SEQUENCE = 4
    COLOR_SCHEME_GLYCOSYLATION_SITE = 5
    COLOR_SCHEME_REGION = 6

    SHADING_MODE_NONE = 0
    SHADING_MODE_BASIC = 1
    SHADING_MODE_DIFFUSE = 2
    SHADING_MODE_ELECTRON = 3
    SHADING_MODE_CARTOON = 4
    SHADING_MODE_ELECTRON_TRANSPARENCY = 5
    SHADING_MODE_PERLIN = 6
    SHADING_MODE_DIFFUSE_TRANSPARENCY = 7
    SHADING_MODE_CHECKER = 8
    SHADING_MODE_GOODSELL = 9

    SHADING_CHAMELEON_MODE_NONE = 0
    SHADING_CHAMELEON_MODE_EMITTER = 1
    SHADING_CHAMELEON_MODE_RECEIVER = 2

    CAMERA_PROJECTION_PERSPECTIVE = 0
    CAMERA_PROJECTION_FISHEYE = 1
    CAMERA_PROJECTION_PANORAMIC = 2
    CAMERA_PROJECTION_CYLINDRIC = 3

    RNA_SHAPE_TREFOIL_KNOT = 0
    RNA_SHAPE_TORUS = 1
    RNA_SHAPE_STAR = 2
    RNA_SHAPE_SPRING = 3
    RNA_SHAPE_HEART = 4
    RNA_SHAPE_THING = 5
    RNA_SHAPE_MOEBIUS = 6

    RENDERING_QUALITY_LOW = 0
    RENDERING_QUALITY_HIGH = 1

    REPRESENTATION_ATOMS = 0
    REPRESENTATION_ATOMS_AND_STICKS = 1
    REPRESENTATION_CONTOURS = 2
    REPRESENTATION_SURFACE = 3
    REPRESENTATION_UNION_OF_BALLS = 4
    REPRESENTATION_DEBUG = 5

    ASSEMBLY_SHAPE_SPHERICAL = 0
    ASSEMBLY_SHAPE_PLANAR = 1
    ASSEMBLY_SHAPE_SINUSOIDAL = 2
    ASSEMBLY_SHAPE_CUBIC = 3
    ASSEMBLY_SHAPE_FAN = 4
    ASSEMBLY_SHAPE_BEZIER = 5
    ASSEMBLY_SHAPE_SPHERICAL_TO_PLANAR = 6

    NAME_PROTEIN_S_OPEN = "Protein S (open)"
    NAME_PROTEIN_S_CLOSED = "Protein S (closed)"
    NAME_PROTEIN_M = "Protein M"
    NAME_PROTEIN_E = "Protein E"
    NAME_RNA_SEQUENCE = "RNA sequence"
    NAME_MEMBRANE = "Membrane"
    NAME_TRANS_MEMBRANE = "Trans-membrane"
    NAME_RECEPTOR = "Receptor"
    NAME_PROTEIN = "Protein"

    NAME_SURFACTANT_HEAD = "Head"
    NAME_COLLAGEN = "Collagen"
    NAME_GLUCOSE = "Glucose"

    NAME_LACTOFERRIN = "Lactoferrin"
    NAME_DEFENSIN = "Defensin"

    NAME_GLYCAN_HIGH_MANNOSE = "High-mannose"
    NAME_GLYCAN_O_GLYCAN = "O-glycan"
    NAME_GLYCAN_HYBRID = "Hybrid"
    NAME_GLYCAN_COMPLEX = "Complex"

    SURFACTANT_BRANCH = 0
    SURFACTANT_PROTEIN_A = 1
    SURFACTANT_PROTEIN_D = 2

    FILE_FORMAT_UNSPECIFIED = 0
    FILE_FORMAT_XYZ_BINARY = 1
    FILE_FORMAT_XYZR_BINARY = 2
    FILE_FORMAT_XYZRV_BINARY = 3
    FILE_FORMAT_XYZ_ASCII = 4
    FILE_FORMAT_XYZR_ASCII = 5
    FILE_FORMAT_XYZRV_ASCII = 6

    def __init__(self, url='localhost:5000'):
        """Create a new BioExplorer instance"""
        self._url = url
        self._client = Client(url)

        backend_version = self.version()
        if __version__ != backend_version:
            raise RuntimeError(
                "Wrong version of the back-end (" + backend_version +
                "). Use version " + __version__ +
                " for this version of the BioExplorer python library")

    def __str__(self):
        """
        A pretty-print of the class

        :rtype: string
        """
        return "Blue Brain BioExplorer"

    def core_api(self):
        """
        Access to underlying core API (Brayns core API)

        :rtype: Class
        """
        return self._client

    def version(self):
        """
        Version of the BioExplorer application

        :rtype: string
        """
        if self._client is None:
            return __version__

        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "get-version")
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result["contents"]

    def scene_information(self):
        """
        Metrics about the scene handled by the BioExplorer backend

        :rtype: Metrics
        """
        if self._client is None:
            return __version__

        return self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "get-scene-information")

    @staticmethod
    def authors():
        """
        List of authors

        :rtype: string
        """
        return "Cyrille Favreau (cyrille.favreau@epfl.ch)"

    @staticmethod
    def _check(command):
        result = command
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def reset(self):
        """
        Remove all assemblies

        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        return self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "reset")

    def export_to_file(self, filename, low_bounds=Vector3(-1e38, -1e38, -1e38),
                       high_bounds=Vector3(1e38, 1e38, 1e38)):
        """
        Export current scene to file as an optimized binary cache file

        :filename: Full path of the binary cache file
        :low_bounds: Brick low bounds
        :high_bounds: Brick high bounds
        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        params = dict()
        params["filename"] = filename
        params["lowBounds"] = low_bounds.to_list()
        params["highBounds"] = high_bounds.to_list()
        params["fileFormat"] = BioExplorer.FILE_FORMAT_UNSPECIFIED
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "export-to-file", params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def export_to_database(self, connection_string, schema, brick_id,
                           low_bounds=Vector3(-1e38, -1e38, -1e38),
                           high_bounds=Vector3(1e38, 1e38, 1e38)):
        """
        Export current scene to file as an optimized binary database entry

        :connection_string: Connection string to the database
        :schema: Database schema
        :schema: Id of the brick
        :low_bounds: Brick low bounds
        :high_bounds: Brick high bounds
        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        assert isinstance(low_bounds, Vector3)
        assert isinstance(high_bounds, Vector3)

        params = dict()
        params["connectionString"] = connection_string
        params["schema"] = schema
        params["brickId"] = brick_id
        params["lowBounds"] = low_bounds.to_list()
        params["highBounds"] = high_bounds.to_list()
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "export-to-database",
            params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def import_from_cache(self, filename):
        """
        Imports a 3D scene from an optimized binary cache file

        :filename: Full path of the binary cache file
        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        params = dict()
        params["filename"] = filename
        params["fileFormat"] = BioExplorer.FILE_FORMAT_UNSPECIFIED
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "import-from-cache", params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def export_to_xyz(self, filename, file_format):
        """
        Exports current scene to file as a XYZ file

        :filename: Full path of the XYZ file
        :file_format: Defines the format of the XYZ file
        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        params = dict()
        params["filename"] = filename
        params["fileFormat"] = file_format
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "export-to-xyz", params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def remove_assembly(self, name):
        """
        Removes the specified assembly

        :name: Name of the assembly
        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        params = dict()
        params["name"] = name
        params["position"] = Vector3().to_list()
        params["rotation"] = list(Quaternion())
        params["clippingPlanes"] = list()
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "remove-assembly", params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def add_coronavirus(self, name, resource_folder, assembly_params=[45.0, 1, 1.0, 2, 0.6, 0.0],
                        nb_protein_s=62, nb_protein_m=50, nb_protein_e=42,
                        open_protein_s_indices=list([0]), atom_radius_multiplier=1.0,
                        add_glycans=False, add_rna_sequence=False,
                        representation=REPRESENTATION_ATOMS, clipping_planes=list(),
                        position=Vector3(), rotation=Quaternion(), apply_colors=False):
        """
        Add a virus with the default coronavirus parameters

        :name: Name of the coronavirus
        :resource_folder: Folder containing the resources of the virus components (PDB and
                                 RNA files)
        :radius: Radius of the virus in nanometers
        :nb_protein_s: Number of S proteins
        :nb_protein_m: Number of M proteins
        :nb_protein_e: Number of E proteins
        :open_protein_s_indices: Indices for the open S proteins
        :add_glycans: Defines if glycans should be added
        :add_rna_sequence: Defines if RNA sequence should be added
        :atom_radius_multiplier: Multiplies atom radius by the specified value
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :clipping_planes: List of clipping planes to apply to the virus assembly
        :position: Position of the virus in the scene
        :rotation: rotation of the coronavirus in the scene
        :apply_colors: Applies default colors to the virus
        """
        pdb_folder = resource_folder + "pdb/"
        rna_folder = resource_folder + "rna/"
        glycan_folder = pdb_folder + "glycans/"
        membrane_folder = pdb_folder + "membrane/"

        open_conformation_indices = open_protein_s_indices
        closed_conformation_indices = list()
        for i in range(nb_protein_s):
            if i not in open_conformation_indices:
                closed_conformation_indices.append(i)

        # Protein S
        params = [11.5, 0, 0.0, assembly_params[3], 0.1, assembly_params[5]]
        virus_protein_s = Protein(
            sources=[
                pdb_folder + "6vyb.pdb", pdb_folder + "sars-cov-2-v1.pdb"
            ],
            occurences=nb_protein_s,
            assembly_params=params,
            rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            instance_indices=[
                open_conformation_indices, closed_conformation_indices
            ]
        )

        # Protein M (QHD43419)
        params = [
            2.5, 1, assembly_params[2], assembly_params[3] + 2, assembly_params[4],
            assembly_params[5]
        ]
        virus_protein_m = Protein(
            sources=[pdb_folder + "QHD43419a.pdb"],
            occurences=nb_protein_m,
            assembly_params=params,
            rotation=Quaternion(0.135, 0.99, 0.0, 0.0)
        )

        # Protein E (QHD43418 P0DTC4)
        params = [
            2.5, 3, assembly_params[2], assembly_params[3] + 4, assembly_params[4],
            assembly_params[5]
        ]
        virus_protein_e = Protein(
            sources=[pdb_folder + "QHD43418a.pdb"],
            occurences=nb_protein_e,
            assembly_params=params,
            rotation=Quaternion(-0.04, 0.705, 0.705, -0.04)
        )

        # Virus membrane
        virus_membrane = ParametricMembrane(
            sources=[
                membrane_folder + 'segA.pdb',
                membrane_folder + 'segB.pdb',
                membrane_folder + 'segC.pdb',
                membrane_folder + 'segD.pdb'
            ],
            occurences=6000
        )

        # RNA Sequence
        rna_sequence = None
        if add_rna_sequence:
            params = [11.0, 0.5]
            rna_sequence = RNASequence(
                source=rna_folder + "sars-cov-2.rna",
                assembly_params=params,
                t_range=Vector2(0, 30.5 * math.pi),
                shape=self.RNA_SHAPE_TREFOIL_KNOT,
                shape_params=Vector3(1.51, 1.12, 1.93),
                position=Vector3()
            )

        coronavirus = Virus(
            name=name,
            protein_s=virus_protein_s,
            protein_e=virus_protein_e,
            protein_m=virus_protein_m,
            membrane=virus_membrane,
            rna_sequence=rna_sequence,
            assembly_params=assembly_params
        )

        self.add_virus(
            virus=coronavirus,
            representation=representation,
            atom_radius_multiplier=atom_radius_multiplier,
            clipping_planes=clipping_planes,
            position=position,
            rotation=rotation
        )

        if add_glycans:
            complex_paths = [
                glycan_folder + "complex/5.pdb",
                glycan_folder + "complex/15.pdb",
                glycan_folder + "complex/25.pdb",
                glycan_folder + "complex/35.pdb",
            ]

            high_mannose_paths = [
                glycan_folder + "high-mannose/1.pdb",
                glycan_folder + "high-mannose/2.pdb",
                glycan_folder + "high-mannose/3.pdb",
                glycan_folder + "high-mannose/4.pdb",
            ]

            o_glycan_paths = [glycan_folder + "o-glycan/12.pdb"]

            # High-mannose
            indices_closed = [61, 122, 234, 603, 709, 717, 801, 1074]
            indices_open = [61, 122, 234, 709, 717, 801, 1074]
            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=self.NAME_PROTEIN_S_CLOSED,
                paths=high_mannose_paths,
                indices=indices_closed,
                representation=representation,
                atom_radius_multiplier=atom_radius_multiplier,
                assembly_params=[0, 0, 0.0, assembly_params[3] + 7, assembly_params[4]]
            )
            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=high_mannose_paths,
                indices=indices_open,
                representation=representation,
                atom_radius_multiplier=atom_radius_multiplier,
                assembly_params=[0, 0, 0.0, assembly_params[3] + 7, assembly_params[4]]
            )

            # Complex
            indices_closed = [17, 74, 149, 165, 282, 331,
                              343, 616, 657, 1098, 1134, 1158, 1173, 1194]
            indices_open = [17, 74, 149, 165, 282, 331, 343, 657, 1098, 1134, 1158, 1173, 1194]
            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_COMPLEX,
                protein_name=self.NAME_PROTEIN_S_CLOSED,
                paths=complex_paths,
                indices=indices_closed,
                representation=representation,
                atom_radius_multiplier=atom_radius_multiplier,
                assembly_params=[0, 0, 0.0, assembly_params[3] + 8, 2.0 * assembly_params[4]]
            )

            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_COMPLEX,
                protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=complex_paths,
                indices=indices_open,
                representation=representation,
                atom_radius_multiplier=atom_radius_multiplier,
                assembly_params=[0, 0, 0.0, assembly_params[3] + 8, 2.0 * assembly_params[4]]
            )

            # O-Glycans
            for index in [323, 325]:
                o_glycan_name = (name + "_" + self.NAME_GLYCAN_O_GLYCAN + "_" +
                                 str(index))
                o_glycan = Sugars(
                    assembly_name=name,
                    name=o_glycan_name,
                    source=o_glycan_paths[0],
                    protein_name=name + "_" + self.NAME_PROTEIN_S_CLOSED,
                    site_indices=[index],
                    representation=representation,
                    atom_radius_multiplier=atom_radius_multiplier,
                    assembly_params=[0, 0, 0.0, assembly_params[3] + 9, 2.0 * assembly_params[4]]
                )
                self.add_sugars(o_glycan)

            # High-mannose glycans on Protein M
            indices = [5]
            protein_name = name + "_" + self.NAME_PROTEIN_M
            high_mannose_glycans = Sugars(
                rotation=Quaternion(0.707, 0.0, 0.0, 0.707),
                assembly_name=name,
                name=protein_name + "_" + self.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=protein_name,
                source=high_mannose_paths[0],
                site_indices=indices,
                representation=representation,
                assembly_params=[0, 0, 0.0, assembly_params[3] + 10, 2.0 * assembly_params[4]]
            )
            self.add_glycans(high_mannose_glycans)

            # Complex glycans on Protein E
            indices = [48, 66]
            protein_name = name + "_" + self.NAME_PROTEIN_E
            complex_glycans = Sugars(
                rotation=Quaternion(0.707, 0.0, 0.0, 0.707),
                assembly_name=name,
                name=protein_name + "_" + self.NAME_GLYCAN_COMPLEX,
                protein_name=protein_name,
                source=complex_paths[0],
                site_indices=indices,
                representation=representation,
                assembly_params=[0, 0, 0.0, assembly_params[3] + 11, 2.0 * assembly_params[4]]
            )
            self.add_glycans(complex_glycans)

        if apply_colors:
            # Apply default materials
            self.apply_default_color_scheme(
                shading_mode=self.SHADING_MODE_BASIC)

    def add_virus(self, virus, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                  clipping_planes=list(), position=Vector3(), rotation=Quaternion()):
        """
        Adds a virus assembly to the scene

        :virus: Description of the virus
        :atom_radius_multiplier: Multiplies atom radius by the specified value
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :clipping_planes: List of clipping planes to apply to the virus assembly
        :position: Position of the virus in the scene
        :rotation: rotation of the protein in the scene
        """
        assert isinstance(virus, Virus)
        assert isinstance(clipping_planes, list)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)

        shape = self.ASSEMBLY_SHAPE_SPHERICAL_TO_PLANAR
        _protein_s = virus.protein_s

        self.remove_assembly(virus.name)
        self.add_assembly(
            name=virus.name,
            clipping_planes=clipping_planes,
            position=position,
            rotation=rotation
        )

        if virus.protein_s:
            params = virus.protein_s.assembly_params.copy()
            params[0] += virus.assembly_params[0]

            # S Protein
            if _protein_s.instance_indices[0]:
                # Open conformation
                _protein_s_open = AssemblyProtein(
                    assembly_name=virus.name,
                    name=virus.name + "_" + self.NAME_PROTEIN_S_OPEN,
                    source=_protein_s.sources[0],
                    shape=shape,
                    load_hydrogen=_protein_s.load_hydrogen,
                    occurrences=_protein_s.occurences,
                    assembly_params=params,
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=_protein_s.load_bonds,
                    load_non_polymer_chemicals=_protein_s.
                    load_non_polymer_chemicals,
                    representation=representation,
                    random_seed=1,
                    position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL,
                    rotation=_protein_s.rotation,
                    allowed_occurrences=_protein_s.instance_indices[0]
                )
                self.add_assembly_protein(_protein_s_open)

            if _protein_s.instance_indices[1]:
                # Closed conformation
                _protein_s_closed = AssemblyProtein(
                    assembly_name=virus.name,
                    name=virus.name + "_" + self.NAME_PROTEIN_S_CLOSED,
                    source=_protein_s.sources[1],
                    shape=shape,
                    load_hydrogen=_protein_s.load_hydrogen,
                    occurrences=_protein_s.occurences,
                    assembly_params=params,
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=_protein_s.load_bonds,
                    load_non_polymer_chemicals=_protein_s.
                    load_non_polymer_chemicals,
                    representation=representation,
                    random_seed=1,
                    position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL,
                    rotation=_protein_s.rotation,
                    allowed_occurrences=_protein_s.instance_indices[1]
                )
                self.add_assembly_protein(_protein_s_closed)

        if virus.protein_m:
            # M Protein
            params = virus.protein_m.assembly_params.copy()
            params[0] += virus.assembly_params[0]

            _protein_m = AssemblyProtein(
                assembly_name=virus.name,
                name=virus.name + "_" + self.NAME_PROTEIN_M,
                source=virus.protein_m.sources[0],
                shape=shape,
                assembly_params=params,
                load_hydrogen=virus.protein_m.load_hydrogen,
                occurrences=virus.protein_m.occurences,
                atom_radius_multiplier=atom_radius_multiplier,
                load_bonds=virus.protein_m.load_bonds,
                load_non_polymer_chemicals=virus.protein_m.
                load_non_polymer_chemicals,
                representation=representation,
                random_seed=2,
                position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL,
                rotation=virus.protein_m.rotation
            )
            self.add_assembly_protein(_protein_m)

        if virus.protein_e:
            # E Protein
            params = virus.protein_e.assembly_params.copy()
            params[0] += virus.assembly_params[0]

            _protein_e = AssemblyProtein(
                assembly_name=virus.name,
                name=virus.name + "_" + self.NAME_PROTEIN_E,
                source=virus.protein_e.sources[0],
                shape=shape,
                assembly_params=params,
                load_hydrogen=virus.protein_e.load_hydrogen,
                occurrences=virus.protein_e.occurences,
                atom_radius_multiplier=atom_radius_multiplier,
                load_bonds=virus.protein_e.load_bonds,
                load_non_polymer_chemicals=virus.protein_e.
                load_non_polymer_chemicals,
                representation=representation,
                random_seed=3,
                position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL,
                rotation=virus.protein_e.rotation
            )
            self.add_assembly_protein(_protein_e)

        if virus.membrane:
            # Parametric membrane
            virus.membrane.representation = representation
            virus.membrane.atom_radius_multiplier = atom_radius_multiplier
            self.add_parametric_membrane(
                assembly_name=virus.name,
                name=virus.name + "_" + self.NAME_MEMBRANE,
                membrane=virus.membrane,
                shape=BioExplorer.ASSEMBLY_SHAPE_SPHERICAL_TO_PLANAR,
                position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL,
                assembly_params=virus.assembly_params,
                rotation=Quaternion(0.0, 0.0, 1.0, 0.0),
                random_seed=4
            )

        if virus.rna_sequence:
            # RNA Sequence
            self.add_rna_sequence(
                assembly_name=virus.name,
                name=virus.name + "_" + self.NAME_RNA_SEQUENCE,
                rna_sequence=virus.rna_sequence
            )

    def add_cell(self, cell, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                 clipping_planes=list(), position=Vector3(), rotation=Quaternion(),
                 random_seed=0):
        """
        Add a cell assembly to the scene

        :cell: Description of the cell
        :atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        :representation: Multiplies atom radius by the specified value
        :clipping_planes: List of clipping planes to apply to the virus assembly
        :position: Position of the cell in the scene
        :rotation: rotation of the cell in the scene
        :random_seed: Seed used to randomise position the elements in the membrane
        """
        assert isinstance(cell, Cell)
        assert isinstance(clipping_planes, list)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert len(cell.receptor.sources) == 1

        self.remove_assembly(cell.name)
        self.add_assembly(
            name=cell.name,
            position=position,
            rotation=rotation,
            clipping_planes=clipping_planes
        )

        if cell.receptor.occurences != 0:
            receptor_params = cell.params.copy()
            receptor_params[4] /= 20.0  # Receptor rotates 20 times less than lipids

            _receptor = AssemblyProtein(
                assembly_name=cell.name,
                name=cell.name + "_" + self.NAME_RECEPTOR,
                shape=cell.shape,
                position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL,
                source=cell.receptor.sources[0],
                load_non_polymer_chemicals=cell.receptor.
                load_non_polymer_chemicals,
                occurrences=cell.receptor.occurences,
                assembly_params=receptor_params,
                atom_radius_multiplier=atom_radius_multiplier,
                load_bonds=True,
                representation=representation,
                random_seed=random_seed,
                position=cell.receptor.position,
                rotation=cell.receptor.rotation
            )
            self.add_assembly_protein(_receptor)

        cell.membrane.representation = representation
        cell.membrane.atom_radius_multiplier = atom_radius_multiplier
        return self.add_parametric_membrane(
            assembly_name=cell.name,
            name=cell.name + "_" + self.NAME_MEMBRANE,
            shape=cell.shape,
            assembly_params=cell.params,
            random_seed=random_seed,
            position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL,
            membrane=cell.membrane
        )

    def add_volume(self, volume, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                   position=Vector3(), random_seed=1):
        """
        Add a volume assembly to the scene

        :volume: Description of the volume
        :atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        :representation: Multiplies atom radius by the specified value
        :position: Position of the volume in the scene
        :random_seed: Random seed used to define the positions of the proteins in the volume
        """
        assert isinstance(volume, Volume)
        assert isinstance(position, Vector3)
        assert len(volume.protein.sources) == 1

        _protein = AssemblyProtein(
            assembly_name=volume.name,
            name=volume.name,
            shape=volume.shape,
            source=volume.protein.sources[0],
            load_non_polymer_chemicals=volume.protein.
            load_non_polymer_chemicals,
            occurrences=volume.protein.occurences,
            assembly_params=volume.params,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=True,
            representation=representation,
            random_seed=random_seed,
            position=volume.protein.position,
            rotation=volume.protein.rotation
        )

        self.remove_assembly(volume.name)
        result = self.add_assembly(name=volume.name, position=position)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        result = self.add_assembly_protein(_protein)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def add_surfactant(self, surfactant, atom_radius_multiplier=1.0,
                       representation=REPRESENTATION_ATOMS, position=Vector3(),
                       rotation=Quaternion(), random_seed=0):
        """
        Add a surfactant assembly to the scene

        :surfactant: Description of the surfactant
        :atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        :representation: Multiplies atom radius by the specified value
        :position: Position of the volume in the scene
        :rotation: rotation of the cell in the scene
        :random_seed: Random seed used to define the shape of the branches
        """
        assert isinstance(surfactant, Surfactant)
        assert isinstance(position, Vector3)

        shape = self.ASSEMBLY_SHAPE_SPHERICAL
        nb_branches = 1
        if surfactant.surfactant_protein == self.SURFACTANT_PROTEIN_A:
            shape = self.ASSEMBLY_SHAPE_FAN
            nb_branches = 6
        elif surfactant.surfactant_protein == self.SURFACTANT_PROTEIN_D:
            nb_branches = 4

        nb_collagens = 2
        collagen_size = 16.0

        head_name = surfactant.name + "_" + self.NAME_SURFACTANT_HEAD
        branch_name = surfactant.name + "_" + self.NAME_COLLAGEN + "_"

        protein_sp_d = AssemblyProtein(
            assembly_name=surfactant.name,
            name=head_name,
            shape=shape,
            source=surfactant.head_source,
            occurrences=nb_branches,
            assembly_params=[collagen_size * (nb_collagens + 1) - 9.0, 0, 0.0, 0, 0.0, 0.0],
            atom_radius_multiplier=atom_radius_multiplier,
            random_seed=random_seed,
            representation=representation,
            rotation=Quaternion(0.115, -0.297, 0.344, 0.883)
        )

        collagens = list()
        for i in range(nb_collagens):
            collagens.append(
                AssemblyProtein(
                    assembly_name=surfactant.name,
                    name=branch_name + str(i),
                    shape=shape,
                    atom_radius_multiplier=atom_radius_multiplier,
                    source=surfactant.branch_source,
                    occurrences=nb_branches,
                    assembly_params=[collagen_size * (i + 1) - 7.0, 0, 0.0, 0, 0.0, 0.0],
                    random_seed=random_seed,
                    representation=representation
                ))

        self.remove_assembly(surfactant.name)
        result = self.add_assembly(name=surfactant.name,
                                   position=position,
                                   rotation=rotation)
        if not result["status"]:
            raise RuntimeError(result["contents"])

        for collagen in collagens:
            result = self.add_assembly_protein(collagen)
            if not result["status"]:
                raise RuntimeError(result["contents"])
        result = self.add_assembly_protein(protein_sp_d)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    @staticmethod
    def get_mol():
        """
        Provide a hint to the meaning of life

        Provide a hint by solving a polynomial using Neville interpolation according to the given
        parameters provided by the BioExplorer. More information available at
        https://www.dcode.fr/function-equation-finder

        :return: A hint to the meaning of life
        """
        coefs = [
            [
                [32341, 39916800], [89063, 1814400], [937889, 725760], [234527, 12096],
                [221003071, 1209600], [97061099, 86400], [3289126079, 725760],
                [106494629, 9072], [16747876289, 907200], [8214593, 525], [73270357, 13860],
                [97, 1]
            ],
            [
                [2039, 13305600], [481, 43200], [251567, 725760], [6809, 1120],
                [80200787, 1209600], [2243617, 4800], [172704437, 80640], [13473253, 2160],
                [9901558223, 907200], [16059286, 1575], [10483043, 2772], [108, 1]
            ],
            [
                [13, 86400], [661, 72576], [269, 1120], [437929, 120960],
                [6933401, 201600], [3696647, 17280], [7570727, 8640], [839289373, 362880],
                [20844207, 5600], [32765923, 10080], [64229, 56], [100, 0]
            ]
        ]

        def mol(x, y):
            value = 0.0
            for i in range(len(coefs[y]) - 1):
                j = len(coefs[y]) - i - 1
                if i % 2 == coefs[y][len(coefs[y]) - 1][1]:
                    value -= coefs[y][i][0] * math.pow(x, j) / coefs[y][i][1]
                else:
                    value += coefs[y][i][0] * math.pow(x, j) / coefs[y][i][1]
            value += coefs[y][len(coefs[y]) - 1][0]
            return round(value)

        result = ''
        for x in range(33):
            result += chr(mol(x % 12, int(x / 12)))
        return 'You asked for the meaning of life and the answer is: ' + result

    def add_assembly(self, name, clipping_planes=list(), position=Vector3(), rotation=Quaternion()):
        """
        Add an assembly to the scene

        :name: Name of the assembly
        :clipping_planes: List of clipping planes to apply to the virus assembly
        :position: Position of the scene in the scene
        :rotation: rotation of the assembly in the scene
        """
        assert isinstance(clipping_planes, list)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)

        clipping_planes_values = list()
        for plane in clipping_planes:
            for i in range(4):
                clipping_planes_values.append(plane[i])

        params = dict()
        params["name"] = name
        params["position"] = position.to_list()
        params["rotation"] = list(rotation)
        params["clippingPlanes"] = clipping_planes_values
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "add-assembly", params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        self._client.set_renderer(accumulation=True)
        return result

    def set_protein_color_scheme(self, assembly_name, name, color_scheme, palette_name="",
                                 palette_size=256, palette=list(), chain_ids=list()):
        """
        Set a color scheme to a protein

        :assembly_name: Name of the assembly containing the protein
        :name: Name of the protein
        :color_scheme: Color scheme
        :palette_name: Name of the Seaborn color palette
        :palette_size: Size of the Seaborn color palette
        :palette: Seaborn palette (overrides the palette_name and palette size if specified)
        :chain_ids: Ids of the chains to which the color scheme should be applied
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(palette, list)
        assert isinstance(chain_ids, list)

        if not palette and palette_name != "":
            palette = sns.color_palette(palette_name, palette_size)

        local_palette = list()
        for color in palette:
            for i in range(3):
                local_palette.append(color[i])

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["colorScheme"] = color_scheme
        params["palette"] = local_palette
        params["chainIds"] = chain_ids
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "set-protein-color-scheme",
            params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        self._client.set_renderer(accumulation=True)
        return result

    def set_protein_amino_acid_sequence_as_string(self, assembly_name, name,
                                                  amino_acid_sequence):
        """
        Displays a specified amino acid sequence on the protein

        :assembly_name: Name of the assembly containing the protein
        :name: Name of the protein
        :amino_acid_sequence: String containing the amino acid sequence
        :return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["sequence"] = amino_acid_sequence
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX +
            "set-protein-amino-acid-sequence-as-string",
            params=params,
        )
        if not result["status"]:
            raise RuntimeError(result["contents"])
        self._client.set_renderer(accumulation=True)
        return result

    def set_protein_amino_acid_sequence_as_ranges(self, assembly_name, name,
                                                  amino_acid_ranges):
        """
        Displays a specified amino acid range on the protein

        :assembly_name: Name of the assembly containing the protein
        :name: Name of the protein
        :amino_acid_ranges: Tuples containing the amino acid range
        :return: Result of the call to the BioExplorer backend
        """
        assert len(amino_acid_ranges) % 2 == 0
        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["ranges"] = amino_acid_ranges
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX +
            "set-protein-amino-acid-sequence-as-ranges",
            params=params,
        )
        if not result["status"]:
            raise RuntimeError(result["contents"])
        self._client.set_renderer(accumulation=True)
        return result

    def get_protein_amino_acid_information(self, assembly_name, name):
        """
        Returns amino acid information of the protein

        :assembly_name: Name of the assembly containing the protein
        :name: Name of the protein
        :return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX +
            "get-protein-amino-acid-information",
            params=params,
        )
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result["contents"].split()

    def set_protein_amino_acid(self, assembly_name, name, index, amino_acid_short_name,
                               chain_ids=list()):
        """
        Displays a specified amino acid sequence on the protein

        :assembly_name: Name of the assembly containing the protein
        :name: Name of the protein
        :index: Index of the amino acid in the sequence
        :amino_acid_short_name: String containing the short name of the amino acid
        :chain_ids: Ids of the chains to which the color scheme should be applied
        :return: Result of the call to the BioExplorer backend
        """
        assert index >= 0
        assert len(amino_acid_short_name) == 3
        assert isinstance(chain_ids, list)
        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["index"] = index
        params["aminoAcidShortName"] = amino_acid_short_name
        params["chainIds"] = chain_ids
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "set-protein-amino-acid",
            params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        self._client.set_renderer(accumulation=True)
        return result

    def set_protein_instance_transformation(self, assembly_name, name, instance_index,
                                            position=Vector3(), rotation=Quaternion()):
        """
        Set a transformation to an instance of a protein in an assembly

        :assembly_name: Name of the assembly containing the protein
        :name: Name of the protein
        :instance_index: Index of the protein instance
        :position: Position of the instance
        :rotation: rotation of the instance
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(instance_index, int)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["instanceIndex"] = instance_index
        params["position"] = position.to_list()
        params["rotation"] = list(rotation)
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "set-protein-instance-transformation",
            params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        self._client.set_renderer(accumulation=True)
        return result

    def get_protein_instance_transformation(self, assembly_name, name, instance_index):
        """
        Get a transformation to an instance of a protein in an assembly

        :assembly_name: Name of the assembly containing the protein
        :name: Name of the protein
        :instance_index: Index of the protein instance
        :return: A dictionnary containing the position and rotation of the instance
        """
        assert isinstance(instance_index, int)

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["instanceIndex"] = instance_index
        params["position"] = Vector3().to_list()
        params["rotation"] = list(Quaternion())
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "get-protein-instance-transformation",
            params=params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        d = dict()
        for param in result["contents"].split("|"):
            s = param.split('=')
            d[s[0]] = s[1]
        return d

    def add_rna_sequence(self, assembly_name, name, rna_sequence):
        """
        Add an RNA sequence object to an assembly

        :assembly_name: Name of the assembly
        :name: Name of the RNA sequence
        :rna_sequence: Description of the RNA sequence
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(rna_sequence, RNASequence)
        t_range = Vector2(0.0, 2.0 * math.pi)
        if rna_sequence.t_range is None:
            # Defaults
            if rna_sequence.shape == self.RNA_SHAPE_TORUS:
                t_range = Vector2(0.0, 2.0 * math.pi)
            elif rna_sequence.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                t_range = Vector2(0.0, 4.0 * math.pi)
        else:
            t_range = rna_sequence.t_range

        shape_params = [1.0, 1.0, 1.0]
        if rna_sequence.shape_params is None:
            # Defaults
            if rna_sequence.shape == self.RNA_SHAPE_TORUS:
                shape_params = Vector3(0.5, 10.0, 0.0)
            elif rna_sequence.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                shape_params = Vector3(2.5, 2.0, 2.2)

        else:
            shape_params = rna_sequence.shape_params

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["contents"] = "".join(open(rna_sequence.source).readlines())
        params["shape"] = rna_sequence.shape
        params["assemblyParams"] = rna_sequence.assembly_params
        params["range"] = t_range.to_list()
        params["params"] = shape_params.to_list()
        params["position"] = rna_sequence.position.to_list()
        return self._check(self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "add-rna-sequence", params=params))

    def add_parametric_membrane(self, assembly_name, name, membrane, shape,
                                position_randomization_type, assembly_params, random_seed,
                                rotation=Quaternion()):
        """
        Add a parametric membrane to the scene

        :assembly_name: Name of the assembly
        :name: Name of the cell
        :membrane: Description of the parametric membrane
        :shape: Shape of the parametric membrane
        :position_randomization_type: Type of randomisation for the elements of the membrane
        :assembly_params: Size of the membrane
        :random_seed: Seed used to randomise position the elements in the parametric membrane
        :rotation: rotation of the proteins in the parametric membrane
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(membrane, ParametricMembrane)
        assert isinstance(assembly_params, list)

        contents = ["", "", "", ""]
        for i in range(len(membrane.sources)):
            contents[i] = "".join(open(membrane.sources[i]).readlines())

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["content1"] = contents[0]
        params["content2"] = contents[1]
        params["content3"] = contents[2]
        params["content4"] = contents[3]
        params["shape"] = shape
        params["assemblyParams"] = assembly_params
        params["atomRadiusMultiplier"] = membrane.atom_radius_multiplier
        params["loadBonds"] = membrane.load_bonds
        params["loadNonPolymerChemicals"] = membrane.load_non_polymer_chemicals
        params["representation"] = membrane.representation
        params["chainIds"] = membrane.chain_ids
        params["recenter"] = membrane.recenter
        params["occurrences"] = membrane.occurences
        params["randomSeed"] = random_seed
        params["positionRandomizationType"] = position_randomization_type
        params["rotation"] = list(rotation)
        return self._check(self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "add-membrane", params=params))

    def add_protein(self, name, protein, representation=REPRESENTATION_ATOMS, conformation_index=0,
                    atom_radius_multiplier=1.0, position=Vector3(), rotation=Quaternion()):
        """
        Add a protein to the scene

        :name: Name of the protein
        :protein: Description of the protein
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :conformation_index: Index of the source to be used for the protein conformation
        :atom_radius_multiplier: Multiplies atom radius by the specified value
        :position: Position of the protein in the scene
        :rotation: rotation of the protein in the scene
        :return: Result of the call to the BioExplorer backend
        """
        assert conformation_index < len(protein.sources)
        assert isinstance(protein, Protein)

        self.remove_assembly(name)
        self.add_assembly(name=name)
        _protein = AssemblyProtein(
            assembly_name=name,
            assembly_params=protein.assembly_params,
            name=name,
            source=protein.sources[conformation_index],
            shape=self.ASSEMBLY_SHAPE_PLANAR,
            load_hydrogen=protein.load_hydrogen,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=protein.load_bonds,
            load_non_polymer_chemicals=protein.load_non_polymer_chemicals,
            representation=representation,
            position=position,
            rotation=rotation
        )
        return self.add_assembly_protein(_protein)

    def add_assembly_protein(self, protein):
        """
        Add an protein to an assembly

        :protein: Description of the protein
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(protein, AssemblyProtein)

        params = dict()
        params["assemblyName"] = protein.assembly_name
        params["name"] = protein.name
        params["contents"] = protein.contents
        params["shape"] = protein.shape
        params["assemblyParams"] = protein.assembly_params
        params["atomRadiusMultiplier"] = protein.atom_radius_multiplier
        params["loadBonds"] = protein.load_bonds
        params["loadNonPolymerChemicals"] = protein.load_non_polymer_chemicals
        params["loadHydrogen"] = protein.load_hydrogen
        params["representation"] = protein.representation
        params["chainIds"] = protein.chain_ids
        params["recenter"] = protein.recenter
        params["occurrences"] = protein.occurrences
        params["allowedOccurrences"] = protein.allowed_occurrences
        params["randomSeed"] = protein.random_seed
        params[
            "positionRandomizationType"] = protein.position_randomization_type
        params["position"] = protein.position.to_list()
        params["rotation"] = list(protein.rotation)
        return self._check(self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "add-protein", params=params))

    def add_mesh_based_membrane(self, name, mesh_based_membrane, position=Vector3(),
                                rotation=Quaternion(), scale=Vector3(), clipping_planes=list()):
        """
        Add a mesh-based membrane to the scene

        :name: Name of the mesh-based membrane in the scene
        :mesh: Description of the mesh-based membrane
        :position: Position of the mesh-based membrane in the scene
        :rotation: Rotation of the mesh-based membrane in the scene
        :scale: Scale of the mesh-based membrane in the scene
        :clipping_planes: List of clipping planes to apply to the membrane
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(mesh_based_membrane, MeshBasedMembrane)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(scale, Vector3)
        assert isinstance(clipping_planes, list)

        self.remove_assembly(name)
        self.add_assembly(name=name, clipping_planes=clipping_planes)
        _mesh_based_membrane = AssemblyMeshBasedMembrane(
            assembly_name=name,
            name=name,
            mesh_source=mesh_based_membrane.mesh_source,
            protein_sources=mesh_based_membrane.protein_sources,
            density=mesh_based_membrane.density,
            surface_fixed_offset=mesh_based_membrane.surface_fixed_offset,
            surface_variable_offset=mesh_based_membrane.
            surface_variable_offset,
            atom_radius_multiplier=mesh_based_membrane.atom_radius_multiplier,
            assembly_params=mesh_based_membrane.assembly_params,
            representation=mesh_based_membrane.representation,
            random_seed=mesh_based_membrane.random_seed,
            position=position,
            rotation=rotation,
            scale=scale
        )
        return self.add_assembly_mesh_based_membrane(_mesh_based_membrane)

    def add_assembly_mesh_based_membrane(self, mesh_based_membrane):
        """
        Add an mesh-based membrane to an assembly

        :mesh_based_membrane: Description of the mesh-based membrane
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(mesh_based_membrane, AssemblyMeshBasedMembrane)

        contents = ["", "", "", ""]
        for i in range(len(mesh_based_membrane.protein_sources)):
            contents[i] = "".join(open(mesh_based_membrane.protein_sources[i]).readlines())

        params = dict()
        params["assemblyName"] = mesh_based_membrane.assembly_name
        params["name"] = mesh_based_membrane.name
        params["meshContents"] = "".join(open(mesh_based_membrane.mesh_source).readlines())
        params["proteinContents1"] = contents[0]
        params["proteinContents2"] = contents[1]
        params["proteinContents3"] = contents[2]
        params["proteinContents4"] = contents[3]
        params["recenter"] = mesh_based_membrane.recenter
        params["density"] = mesh_based_membrane.density
        params["surfaceFixedOffset"] = mesh_based_membrane.surface_fixed_offset
        params["surfaceVariableOffset"] = mesh_based_membrane.surface_variable_offset
        params["assemblyParams"] = mesh_based_membrane.assembly_params
        params["atomRadiusMultiplier"] = mesh_based_membrane.atom_radius_multiplier
        params["representation"] = mesh_based_membrane.representation
        params["randomSeed"] = mesh_based_membrane.random_seed
        params["position"] = mesh_based_membrane.position.to_list()
        params["rotation"] = list(mesh_based_membrane.rotation)
        params["scale"] = mesh_based_membrane.scale.to_list()
        return self._check(self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "add-mesh-based-membrane",
            params=params))

    def add_glycans(self, glycans):
        """
        Add glycans to an protein in an assembly

        :glycans: Description of the glycans
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(glycans, Sugars)

        params = dict()
        params["assemblyName"] = glycans.assembly_name
        params["name"] = glycans.name
        params["contents"] = glycans.contents
        params["proteinName"] = glycans.protein_name
        params["atomRadiusMultiplier"] = glycans.atom_radius_multiplier
        params["loadBonds"] = glycans.load_bonds
        params["representation"] = glycans.representation
        params["recenter"] = glycans.recenter
        params["chainIds"] = glycans.chain_ids
        params["siteIndices"] = glycans.site_indices
        params["rotation"] = list(glycans.rotation)
        params["assemblyParams"] = glycans.assembly_params
        return self._check(self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "add-glycans", params=params))

    def add_multiple_glycans(
            self, assembly_name, glycan_type, protein_name, paths, representation, chain_ids=list(),
            indices=list(), load_bonds=True, atom_radius_multiplier=1.0, rotation=Quaternion(),
            assembly_params=list()):
        """
        Add glycans to a protein in a assembly

        :assembly_name: Name of the assembly
        :glycan_type: Type of glycans
        :protein_name: Name of the protein
        :paths: Paths to PDB files with various glycan structures
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :chain_ids: IDs of the chains to be loaded
        :indices: Indices of the glycosylation sites where glycans should be added
        :load_bonds: Defines if bonds should be loaded
        :atom_radius_multiplier: Multiplies atom radius by the specified value
        :rotation: rotation applied to the glycan on the protein
        :assembly_params: Extra optional parameters for positioning on the protein
        """
        assert isinstance(chain_ids, list)
        assert isinstance(indices, list)

        path_index = 0
        for path in paths:
            site_indices = list()
            if indices is not None:
                for index in indices:
                    if index % len(paths) == path_index:
                        site_indices.append(index)

            if site_indices:
                _glycans = Sugars(
                    assembly_name=assembly_name,
                    name=assembly_name + "_" + protein_name + "_" +
                    glycan_type + "_" + str(path_index),
                    source=path,
                    protein_name=assembly_name + "_" + protein_name,
                    chain_ids=chain_ids,
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=load_bonds,
                    representation=representation,
                    recenter=True,
                    site_indices=site_indices,
                    rotation=rotation,
                    assembly_params=assembly_params
                )
                self.add_glycans(_glycans)
            path_index += 1

    def add_sugars(self, sugars):
        """
        Add sugars to a protein in an assembly

        :sugars: Description of the sugars
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(sugars, Sugars)

        params = dict()
        params["assemblyName"] = sugars.assembly_name
        params["name"] = sugars.name
        params["contents"] = sugars.contents
        params["proteinName"] = sugars.protein_name
        params["atomRadiusMultiplier"] = sugars.atom_radius_multiplier
        params["loadBonds"] = sugars.load_bonds
        params["representation"] = sugars.representation
        params["recenter"] = sugars.recenter
        params["chainIds"] = sugars.chain_ids
        params["siteIndices"] = sugars.site_indices
        params["rotation"] = list(sugars.rotation)
        params["assemblyParams"] = sugars.assembly_params
        return self._check(self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "add-sugars", params=params))

    def set_rendering_quality(self, image_quality):
        """
        Set rendering quality using presets

        :image_quality: Quality of the image (RENDERING_QUALITY_LOW or RENDERING_QUALITY_HIGH)
        :return: Result of the call to the BioExplorer backend
        """
        if image_quality == self.RENDERING_QUALITY_HIGH:
            self._client.clear_lights()
            self._client.add_light_directional(
                angularDiameter=0.5, color=[1, 1, 1], direction=[-0.7, -0.4, -1],
                intensity=1.0, is_visible=False
            )

            self._client.set_renderer(
                head_light=False, background_color=[96 / 255, 125 / 255, 139 / 255],
                current='bio_explorer', samples_per_pixel=1, subsampling=4,
                max_accum_frames=128)
            params = self._client.BioExplorerRendererParams()
            params.exposure = 1.0
            params.gi_samples = 1
            params.gi_weight = 0.3
            params.gi_distance = 5000
            params.shadows = 1.0
            params.soft_shadows = 0.1
            params.fog_start = 1200.0
            params.fog_thickness = 300.0
            params.max_bounces = 1
            params.use_hardware_randomizer = False
            return self._client.set_renderer_params(params)
        return self._client.set_renderer(
            background_color=Vector3(),
            current="basic",
            samples_per_pixel=1,
            subsampling=4,
            max_accum_frames=16
        )

    def get_model_name(self, model_id):
        """
        Return the list of model ids in the current scene

        :return: List of model Ids
        """
        params = dict()
        params["modelId"] = model_id
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "get-model-name", params)

    def get_model_ids(self):
        """
        Return the list of model ids in the current scene

        :return: List of model Ids
        """
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "get-model-ids")

    def get_material_ids(self, model_id):
        """
        Return the list of material Ids for a given model

        :model_id: Id of the model
        :return: List of material Ids
        """
        params = dict()
        params["modelId"] = model_id
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "get-material-ids", params)

    def set_materials(self, model_ids, material_ids, diffuse_colors, specular_colors,
                      specular_exponents=list(), opacities=list(), reflection_indices=list(),
                      refraction_indices=list(), glossinesses=list(), shading_modes=list(),
                      emissions=list(), user_parameters=list(), chameleon_modes=list()):
        """
        Set a list of material on a specified list of models

        :model_ids: IDs of the models
        :material_ids: IDs of the materials
        :diffuse_colors: List of diffuse colors (3 values between 0 and 1)
        :specular_colors: List of specular colors (3 values between 0 and 1)
        :specular_exponents: List of diffuse exponents
        :opacities: List of opacities
        :reflection_indices: List of reflection indices (value between 0 and 1)
        :refraction_indices: List of refraction indices
        :glossinesses: List of glossinesses (value between 0 and 1)
        :shading_modes: List of shading modes (SHADING_MODE_NONE, SHADING_MODE_BASIC,
                              SHADING_MODE_DIFFUSE, SHADING_MODE_ELECTRON, SHADING_MODE_CARTOON,
                              SHADING_MODE_ELECTRON_TRANSPARENCY, SHADING_MODE_PERLIN or
                              SHADING_MODE_DIFFUSE_TRANSPARENCY)
        :emissions: List of light emission intensities
        :user_parameters: List of convenience parameter used by some of the shaders
        :chameleon_modes: List of chameleon mode attributes. If receiver, material take the color of
        surrounding emitter geometry
        :return: Result of the request submission
        """
        if self._client is None:
            return

        params = dict()
        params["modelIds"] = model_ids
        params["materialIds"] = material_ids

        d_colors = list()
        for diffuse in diffuse_colors:
            for k in range(3):
                d_colors.append(diffuse[k])
        params["diffuseColors"] = d_colors

        s_colors = list()
        for specular in specular_colors:
            for k in range(3):
                s_colors.append(specular[k])
        params["specularColors"] = s_colors

        params["specularExponents"] = specular_exponents
        params["reflectionIndices"] = reflection_indices
        params["opacities"] = opacities
        params["refractionIndices"] = refraction_indices
        params["emissions"] = emissions
        params["glossinesses"] = glossinesses
        params["shadingModes"] = shading_modes
        params["userParameters"] = user_parameters
        params["chameleonModes"] = chameleon_modes
        return self._client.rockets_client.request(self.PLUGIN_API_PREFIX +
                                                   "set-materials",
                                                   params=params)

    def set_materials_from_palette(self, model_ids, material_ids, palette, shading_mode,
                                   specular_exponent, user_parameter=None, glossiness=None,
                                   emission=None, opacity=None, reflection_index=None,
                                   refraction_index=None, chameleon_mode=None):
        """
        Applies a palette of colors and attributes to specified materials

        :model_ids: Ids of the models
        :material_ids: Ids of the materials
        :palette: Palette of RGB colors
        :shading_mode: Shading mode (None, basic, diffuse, etc)
        :specular_exponent: Specular exponent for diffuse materials
        :user_parameter: User parameter specific to each shading mode
        :glossiness: Material glossiness
        :emission: Light emission
        :opacity: Opacity
        :reflection_index: Reflection index
        :refraction_index: Refraction index
        :chameleon_mode: Chameleon mode attributes. If receiver, material take the color of
        surrounding emitter geometry
        """
        colors = list()
        shading_modes = list()
        user_parameters = list()
        glossinesses = list()
        specular_exponents = list()
        emissions = list()
        opacities = list()
        reflection_indices = list()
        refraction_indices = list()
        chameleon_modes = list()
        for color in palette:
            colors.append(color)
            if shading_mode:
                shading_modes.append(shading_mode)
            if user_parameter:
                user_parameters.append(user_parameter)
            if specular_exponent:
                specular_exponents.append(specular_exponent)
            if glossiness:
                glossinesses.append(glossiness)
            if emission:
                emissions.append(emission)
            if opacity:
                opacities.append(opacity)
            if reflection_index:
                reflection_indices.append(reflection_index)
            if refraction_index:
                refraction_indices.append(refraction_index)
            if chameleon_mode:
                chameleon_modes.append(chameleon_mode)
        self.set_materials(
            model_ids=model_ids,
            material_ids=material_ids,
            diffuse_colors=colors,
            specular_colors=colors,
            specular_exponents=specular_exponents,
            user_parameters=user_parameters,
            glossinesses=glossinesses,
            shading_modes=shading_modes,
            emissions=emissions,
            opacities=opacities,
            reflection_indices=reflection_indices,
            refraction_indices=refraction_indices,
            chameleon_modes=chameleon_modes
        )

    def apply_default_color_scheme(
            self, shading_mode, user_parameter=3.0, specular_exponent=5.0, glossiness=1.0,
            glycans=True, proteins=True, membranes=True, collagen=True):
        """
        Apply a default color scheme to all components in the scene

        :shading_mode: Shading mode (None, basic, diffuse, electron, etc)
        :user_parameter: User parameter specific to each shading mode
        :specular_exponent: Specular exponent for diffuse shading modes
        :glossiness: Glossiness
        """
        glycans_colors = [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0.2, 0.2, 0.7]]

        model_ids = self.get_model_ids()

        for model_id in model_ids["ids"]:
            model_name = self.get_model_name(model_id)['name']
            material_ids = self.get_material_ids(model_id)["ids"]
            nb_materials = len(material_ids)

            if membranes and self.NAME_MEMBRANE in model_name:
                palette = sns.color_palette("gist_heat", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if proteins and self.NAME_RECEPTOR in model_name:
                palette = sns.color_palette("OrRd_r", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if proteins and (self.NAME_PROTEIN_S_CLOSED in model_name
                             or self.NAME_PROTEIN_S_OPEN in model_name
                             or self.NAME_PROTEIN_E in model_name
                             or self.NAME_PROTEIN_M in model_name):
                palette = sns.color_palette("Greens", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if proteins and self.NAME_GLUCOSE in model_name:
                palette = sns.color_palette("Blues", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if proteins and self.NAME_LACTOFERRIN in model_name:
                palette = sns.color_palette("afmhot", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if proteins and self.NAME_DEFENSIN in model_name:
                palette = sns.color_palette("plasma_r", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if glycans and self.NAME_GLYCAN_HIGH_MANNOSE in model_name:
                palette = list()
                for _ in range(nb_materials):
                    palette.append(glycans_colors[0])
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if glycans and self.NAME_GLYCAN_COMPLEX in model_name:
                palette = list()
                for _ in range(nb_materials):
                    palette.append(glycans_colors[1])
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if glycans and self.NAME_GLYCAN_HYBRID in model_name:
                palette = list()
                for _ in range(nb_materials):
                    palette.append(glycans_colors[2])
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if glycans and self.NAME_GLYCAN_O_GLYCAN in model_name:
                palette = list()
                for _ in range(nb_materials):
                    palette.append(glycans_colors[3])
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if proteins and (self.NAME_SURFACTANT_HEAD in model_name
                             or self.NAME_COLLAGEN in model_name):
                palette = sns.color_palette("OrRd_r", nb_materials)
                emission = 0
                if self.NAME_COLLAGEN in model_name:
                    emission = 0.1
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    emission=emission,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )

            if collagen and self.NAME_COLLAGEN in model_name:
                palette = list()
                emissions = list()
                for _ in range(nb_materials):
                    palette.append([1, 1, 1])
                    emissions.append(0.2)
                self.set_materials(
                    model_ids=[model_id], material_ids=material_ids,
                    diffuse_colors=palette, specular_colors=palette,
                    emissions=emissions
                )

    def go_magnetic(
            self, colormap_filename=None, colormap_range=[0, 0.008], voxel_size=0.01,
            density=1.0, samples_per_pixel=64):
        """
        Build fields from current scene and switch to default rendering settings

        :colormap_filename: Colormap full file name
        :colormap_range: Colormap value range
        :voxel_size: Voxel size
        :voxel_size: Density of atoms to consider (between 0 and 1)
        :samples_per_pixel: Samples per pixel
        """
        # Build fields acceleration structures
        result = self.build_fields(voxel_size=voxel_size, density=density)
        fields_model_id = int(result['contents'])

        # Rendering settings
        self._client.set_renderer(
            current='bio_explorer_fields',
            samples_per_pixel=1, subsampling=8,
            max_accum_frames=samples_per_pixel)
        params = self._client.BioExplorerFieldsRendererParams()
        params.cutoff = 5000
        params.exposure = 2.0
        params.alpha_correction = 0.1
        params.nb_ray_steps = 16
        params.nb_ray_refinement_steps = samples_per_pixel
        params.use_hardware_randomizer = True
        self._client.set_renderer_params(params)

        if colormap_filename:
            tf = TransferFunction(
                bioexplorer=self, model_id=fields_model_id,
                filename=colormap_filename)
            tf.set_range(colormap_range)
            return tf
        return None

    def build_fields(self, voxel_size, density=1.0):
        """
        Build fields acceleration structures and creates according data handler

        :voxel_size: Voxel size
        :voxel_size: Density of atoms to consider (between 0 and 1)
        :return: Result of the request submission
        """
        if self._client is None:
            return

        params = dict()
        params["voxelSize"] = voxel_size
        params["density"] = density
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "build-fields", params)

    def import_fields_from_file(self, filename):
        """
        Imports fields acceleration structures from file

        :filename: Octree filename
        :return: Result of the request submission
        """
        if self._client is None:
            return

        params = dict()
        params["filename"] = filename
        params["lowBounds"] = Vector3().to_list()
        params["highBounds"] = Vector3().to_list()
        params["fileFormat"] = BioExplorer.FILE_FORMAT_UNSPECIFIED
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "import-fields-from-file", params)

    def export_fields_to_file(self, model_id, filename):
        """
        Exports fields acceleration structures to file

        :model_id: id of the model containing the fields
        :filename: Octree filename
        :return: Result of the request submission
        """
        assert isinstance(model_id, int)
        if self._client is None:
            return

        params = dict()
        params["modelId"] = model_id
        params["filename"] = filename
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "export-fields-to-file", params)

    def add_grid(self, min_value, max_value, interval, radius=1.0, opacity=0.5, show_axis=True,
                 show_planes=True, show_full_grid=False, colored=True, position=Vector3()):
        """
        Add a reference grid to the scene

        :min_value: Minimum value for all axis
        :max_value: Maximum value for all axis
        :interval: Interval at which lines should appear on the grid
        :radius: Radius of grid lines
        :opacity: Opacity of the grid
        :show_axis: Shows axis if True
        :show_planes: Shows planes if True
        :show_full_grid: Shows full grid if True
        :colored: Colors the grid it True. X in red, Y in green, Z in blue
        :position: Position of the grid
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(position, Vector3)
        params = dict()
        params["minValue"] = min_value
        params["maxValue"] = max_value
        params["steps"] = interval
        params["radius"] = radius
        params["planeOpacity"] = opacity
        params["showAxis"] = show_axis
        params["showPlanes"] = show_planes
        params["showFullGrid"] = show_full_grid
        params["useColors"] = colored
        params["position"] = position.to_list()
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "add-grid", params)

    def add_sphere(self, name, position, radius, color=Vector3(1.0, 1.0, 1.0)):
        """
        Add a reference grid to the scene

        :name: Name of the sphere
        :position: Position of the sphere
        :radius: Radius of the sphere
        :color: RGB Color of the sphere (0..1)
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(position, Vector3)
        assert isinstance(color, Vector3)
        params = dict()
        params["name"] = name
        params["position"] = position.to_list()
        params["radius"] = radius
        params["color"] = color.to_list()
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "add-sphere", params)

    def set_general_settings(self, model_visibility_on_creation=True, off_folder='/tmp',
                             logging_enabled=False):
        """
        Set general settings for the plugin

        :model_visibility_on_creation: Visibility of the model on creation
        :off_folder: Folder where off files are stored (to avoid recomputation of molecular surface)
        :logging_enabled: Enable/Disable back-end logging
        :return: Result of the request submission
        """
        params = dict()
        params["modelVisibilityOnCreation"] = model_visibility_on_creation
        params["offFolder"] = off_folder
        params["loggingEnabled"] = logging_enabled
        response = self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "set-general-settings", params)
        return response

    def set_models_visibility(self, visible):
        """
        Set the visibility of all models in the scene

        :visible: Visibility of the models
        :return: Result of the request submission
        """
        params = dict()
        params["visible"] = visible
        response = self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + "set-models-visibility", params)
        self._client.set_renderer(accumulation=True)
        return response

    def get_out_of_core_configuration(self):
        """
        Returns the out-of-core configuration

        :rtype: string
        """
        result = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "get-out-of-core-configuration")
        if not result["status"]:
            raise RuntimeError(result["contents"])
        d = dict()
        for param in result["contents"].split("|"):
            s = param.split('=')
            d[s[0]] = s[1]
        return d

    def get_out_of_core_progress(self):
        """
        Returns the out-of-core loading progress

        :rtype: float
        """
        response = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX + "get-out-of-core-progress")
        return response

    def get_out_of_core_average_loading_time(self):
        """
        Returns the out-of-core average loading time (per brick, in milliseconds)

        :rtype: float
        """
        response = self._client.rockets_client.request(
            method=self.PLUGIN_API_PREFIX +
            "get-out-of-core-average-loading-time")
        return response


# Internal classes


class AssemblyProtein:
    """An AssemblyProtein is a Protein that belongs to an assembly"""

    def __init__(self, assembly_name, name, source, assembly_params=list(),
                 shape=BioExplorer.ASSEMBLY_SHAPE_PLANAR, atom_radius_multiplier=1.0,
                 load_bonds=True, representation=BioExplorer.REPRESENTATION_ATOMS,
                 load_non_polymer_chemicals=False, load_hydrogen=True, chain_ids=list(),
                 recenter=True, occurrences=1, random_seed=0,
                 position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_CIRCULAR,
                 position=Vector3(), rotation=Quaternion(), allowed_occurrences=list()):
        """
        An AssemblyProtein is a protein that belongs to an assembly

        :assembly_name: Name of the assembly
        :name: Name of the protein
        :sources: Full paths to the PDB files for the various protein conformations
        :assembly_params: Assembly parameters (Virus radius and maximum range for random
                                positions of membrane components)
        :shape: Shape of the membrane (Spherical, planar, sinusoidal, cubic, etc)
        :atom_radius_multiplier: Multiplier applied to atom radius
        :load_bonds: Loads bonds if True
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :load_non_polymer_chemicals: Loads non-polymer chemicals if True
        :load_hydrogen: Loads hydrogens if True
        :chain_ids: IDs of the protein chains to be loaded
        :recenter: Centers the protein if True
        :occurences: Number of occurences to be added to the assembly
        :random_seed: Seed for position randomization
        :position_randomization_type: Type of randomisation for the elements of the membrane
        :position: Relative position of the protein in the assembly
        :rotation: Relative rotation of the protein in the assembly
        :allowed_occurrences: Indices of protein occurences in the assembly for
                                    which proteins are added
        """
        self.assembly_name = assembly_name
        self.name = name
        self.contents = "".join(open(source).readlines())
        self.shape = shape
        self.assembly_params = assembly_params
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.load_hydrogen = load_hydrogen
        self.representation = representation
        self.chain_ids = chain_ids
        self.recenter = recenter
        self.occurrences = occurrences
        self.allowed_occurrences = allowed_occurrences
        self.random_seed = random_seed
        self.position_randomization_type = position_randomization_type
        self.position = position
        self.rotation = rotation


class AssemblyMeshBasedMembrane:
    """An AssemblyMeshBasedMembrane is a Mesh-based membrane that belongs to an assembly"""

    def __init__(self, assembly_name, name, mesh_source, protein_sources, recenter=True, density=1,
                 surface_fixed_offset=0, surface_variable_offset=0, assembly_params=list(),
                 atom_radius_multiplier=1.0,
                 representation=BioExplorer.REPRESENTATION_ATOMS, random_seed=0, position=Vector3(),
                 rotation=Quaternion(), scale=Vector3()):
        """
        An AssemblyMeshBasedMembrane is a mesh-based membrane that belongs to an assembly

        :assembly_name: Name of the assembly
        :name: Name of the mesh-based membrane
        :mesh_source: Full paths to the mesh file defining the shape of the membrane
        :protein_sources: Full paths to the PDB files defining the protein (Maximum 4)
        :recenter: Centers the protein if True
        :density: Density of proteins in surface of the mesh
        :surface_fixed_offset: Fixed offset for the position of the protein above the
                                      surface of the mesh
        :surface_variable_offset: Variable (randomized) offset for the position of the
                                         protein above the surface of the mesh
        :atom_radius_multiplier: Multiplier applied to atom radius
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :random_seed: Seed for randomization of the variable offset
        :position: Position of the mesh in the scene
        :rotation: rotation of the mesh in the scene
        :scale: Scale of the mesh in the scene
        """
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(scale, Vector3)

        self.assembly_name = assembly_name
        self.name = name
        self.mesh_source = mesh_source
        self.protein_sources = protein_sources
        self.recenter = recenter
        self.density = density
        self.surface_fixed_offset = surface_fixed_offset
        self.surface_variable_offset = surface_variable_offset
        self.atom_radius_multiplier = atom_radius_multiplier
        self.assembly_params = assembly_params
        self.representation = representation
        self.random_seed = random_seed
        self.position = position
        self.rotation = rotation
        self.scale = scale


# External classes


class ParametricMembrane:
    """A Parametric membrane is a shaped assembly of phospholipids"""

    def __init__(self, sources, atom_radius_multiplier=1.0, load_bonds=False,
                 representation=BioExplorer.REPRESENTATION_ATOMS, load_non_polymer_chemicals=False,
                 chain_ids=list(), recenter=True, occurences=1,
                 position=Vector3(), rotation=Quaternion()):
        """
        A parametric membrane is an assembly of proteins with a given size and shape

        :sources: Full paths of the PDB files containing the building blocks of the membrane
        :atom_radius_multiplier: Multiplies atom radius by the specified value
        :load_bonds: Defines if bonds should be loaded
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :load_non_polymer_chemicals: Defines if non-polymer chemical should be loaded
        :chain_ids: IDs of the protein chains to be loaded
        :recenter: Defines if proteins should be centered when loaded from PDB files
        :occurences: Number of occurences of proteins defining the parametric membrane
        :position: Position of the parametric membrane in the assembly
        :rotation: Rotation of the parametric membrane in the assembly
        """
        self.sources = sources
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.representation = representation
        self.chain_ids = chain_ids
        self.recenter = recenter
        self.occurences = occurences
        self.position = position
        self.rotation = rotation


class Sugars:
    """Sugars are glycan trees that can be added to the glycosylation sites of a given protein"""

    def __init__(self, assembly_name, name, source, protein_name, atom_radius_multiplier=1.0,
                 load_bonds=True, representation=BioExplorer.REPRESENTATION_ATOMS,
                 recenter=True, chain_ids=list(), site_indices=list(), rotation=Quaternion(),
                 assembly_params=list()):
        """
        Sugar descriptor

        :assembly_name: Name of the assembly in the scene
        :name: Name of sugar in the scene
        :source: Full path to the PDB file
        :protein_name: Name of the protein to which sugars are added
        :atom_radius_multiplier: Multiplier for the size of the atoms
        :load_bonds: Defines if bonds should be loaded
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :recenter: Centers the protein if True
        :chain_ids: Ids of chains to be loaded
        :site_indices: Indices on which sugars should be added on the protein
        :rotation: Rotation of the sugar on the protein
        :assembly_params: Extra optional parameters for positioning on the protein
        """
        assert isinstance(chain_ids, list)
        assert isinstance(site_indices, list)
        assert isinstance(rotation, Quaternion)
        assert isinstance(assembly_params, list)

        self.assembly_name = assembly_name
        self.name = name
        self.contents = "".join(open(source).readlines())
        self.protein_name = protein_name
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.representation = representation
        self.recenter = recenter
        self.chain_ids = chain_ids
        self.site_indices = site_indices
        self.rotation = rotation
        self.assembly_params = assembly_params


class RNASequence:
    """An RNASequence is an assembly of a given shape holding a given genetic code"""

    def __init__(self, source, shape, assembly_params, t_range=Vector2(), shape_params=Vector3(),
                 position=Vector3()):
        """
        RNA sequence descriptor

        :source: Full path of the file containing the RNA sequence
        :shape: Shape of the sequence (Trefoil knot, torus, star, spring, heart, Moebiusknot,
                      etc)
        :assembly_params: Assembly parameters (radius, etc.)
        :t_range: Range of values used to enroll the RNA thread
        :shape_params: Shape parameters
        :position: Position of the RNA sequence in the 3D scene
        """
        assert isinstance(t_range, Vector2)
        assert isinstance(shape_params, Vector3)
        assert isinstance(assembly_params, list)
        assert isinstance(position, Vector3)
        self.source = source
        self.shape = shape
        self.assembly_params = assembly_params
        self.t_range = t_range
        self.shape_params = shape_params
        self.position = position


class Surfactant:
    """A Surfactant is a lipoprotein complex composed of multiple branches + head structures"""

    def __init__(self, name, surfactant_protein, head_source, branch_source):
        """
        Surfactant descriptor

        :name: Name of the surfactant in the scene
        :surfactant_protein: Type of surfactant (A, D, etc.)
        :head_source: Full path to the PDB file for the head of the surfactant
        :branch_source: Full path to the PDB file for the branch of the surfactant
        """
        self.surfactant_protein = surfactant_protein
        self.name = name
        self.head_source = head_source
        self.branch_source = branch_source


class Cell:
    """A Cell is a membrane with receptors"""

    def __init__(self, name, size, shape, membrane, receptor, random_position_seed=0,
                 random_position_strength=0.0, random_rotation_seed=0,
                 random_rotation_strength=0.0, extra_parameters=list()):
        """
        Cell descriptor

        :name: Name of the cell in the scene
        :size: Size of the cell in the scene (in nanometers)
        :shape: Shape of the membrane (Spherical, planar, sinusoidal, cubic, etc)
        :membrane: Membrane descriptor
        :receptor: Receptor descriptor
        :random_position_seed: TODO
        :random_position_strength: TODO
        :random_rotation_seed: TODO
        :random_rotation_strength: TODO
        :extra_parameters: Extra parameters depending on the selected shape
        """
        assert isinstance(size, float)
        assert isinstance(membrane, ParametricMembrane)
        assert isinstance(receptor, Protein)
        assert isinstance(random_position_seed, int)
        assert isinstance(random_position_strength, float)
        assert isinstance(random_rotation_seed, int)
        assert isinstance(random_rotation_strength, float)
        assert isinstance(extra_parameters, list)
        self.name = name
        self.shape = shape
        self.params = [
            size, random_position_seed, random_position_strength, random_rotation_seed,
            random_rotation_strength] + extra_parameters
        self.membrane = membrane
        self.receptor = receptor


class Volume:
    """A volume define a 3D space in which proteins can be added"""

    def __init__(self, name, size, protein, random_position_seed=0, random_position_stength=0.0,
                 random_rotation_seed=0, random_rotation_stength=0.0):
        """
        Volume description

        :name: Name of the volume in the scene
        :size: Size of the volume in the scene (in nanometers)
        :random_position_seed: TODO
        :random_position_stength: TODO
        :random_rotation_seed: TODO
        :random_rotation_stength: TODO
        :protein: Protein descriptor
        """
        assert isinstance(random_position_seed, int)
        assert isinstance(random_position_stength, float)
        assert isinstance(random_rotation_seed, int)
        assert isinstance(random_rotation_stength, float)
        assert isinstance(protein, Protein)

        self.name = name
        self.shape = BioExplorer.ASSEMBLY_SHAPE_CUBIC
        self.params = [
            size, random_position_seed, random_position_stength, random_rotation_seed,
            random_rotation_stength, 0]
        self.protein = protein


class Protein:
    """A Protein holds the 3D structure of a protein as well as it Amino Acid sequences"""

    def __init__(self, sources, occurences=1, assembly_params=list(), load_bonds=False,
                 load_hydrogen=False, load_non_polymer_chemicals=False, position=Vector3(),
                 rotation=Quaternion(), instance_indices=list()):
        """
        Protein descriptor

        :sources: Full paths to the PDB files for the various conformations
        :occurences: Number of occurences to be added to the assembly
        :assembly_params: Assembly parameters (Virus radius and maximum range for random positions
                          of membrane components)
        :load_bonds: Loads bonds if True
        :load_hydrogen: Loads hydrogens if True
        :load_non_polymer_chemicals: Loads non-polymer chemicals if True
        :position: Position of the mesh in the scene
        :rotation: Rotation of the mesh in the scene
        :instance_indices: Specific indices for which an instance is added to the assembly
        """
        assert isinstance(sources, list)
        assert sources
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(instance_indices, list)
        assert isinstance(assembly_params, list)
        self.sources = sources
        self.occurences = occurences
        self.assembly_params = assembly_params
        self.load_bonds = load_bonds
        self.load_hydrogen = load_hydrogen
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.position = position
        self.rotation = rotation
        self.instance_indices = instance_indices


class MeshBasedMembrane:
    """A MeshBasedMembrane is a membrane shaped by a 3D mesh"""

    def __init__(self, mesh_source, protein_sources, density=1, surface_fixed_offset=0.0,
                 surface_variable_offset=0.0, assembly_params=list(), atom_radius_multiplier=1.0,
                 representation=BioExplorer.REPRESENTATION_ATOMS, random_seed=0, recenter=True,
                 position=Vector3(), rotation=Quaternion(), scale=Vector3()):
        """
        Mesh-based membrane descriptor

        :mesh_source: Full path to the OBJ file
        :protein_sources: Full path to the PDB files (Maximum 4)
        :density: Density of proteins on the surface of the mesh
        :surface_fixed_offset: Fixed offset of the protein position at the surface of the
                                      mesh
        :surface_variable_offset: Random ranged offset of the protein position at the
                                         surface of the mesh
        :assembly_params: Assembly parameters for animation purposes
        :atom_radius_multiplier: Multiplies atom radius by the specified value
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :random_seed: Rand seed for surface_variable_offset parameter
        :recenter: Centers the protein if True
        :position: Position of the mesh in the scene
        :rotation: rotation of the mesh in the scene
        :scale: Scale of the mesh in the scene
        """
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(scale, Vector3)
        assert len(protein_sources) <= 4
        assert isinstance(assembly_params, list)

        self.mesh_source = mesh_source
        self.protein_sources = protein_sources
        self.density = density
        self.surface_fixed_offset = surface_fixed_offset
        self.assembly_params = assembly_params
        self.surface_variable_offset = surface_variable_offset
        self.atom_radius_multiplier = atom_radius_multiplier
        self.representation = representation
        self.recenter = recenter
        self.random_seed = random_seed
        self.position = position
        self.rotation = rotation
        self.scale = scale


class Virus:
    """A Virus is an assembly of proteins (S, M and E), a membrane, and an RNA sequence"""

    def __init__(self, name, assembly_params, protein_s=None, protein_e=None, protein_m=None,
                 membrane=None, rna_sequence=None):
        """
        Virus descriptor

        :name: Name of the virus in the scene
        :assembly_params: Assembly parameters (Virus radius and maximum range for random
                                positions of membrane components)
        :protein_s: Protein S descriptor
        :protein_e: Protein E descriptor
        :protein_m: Protein M descriptor
        :membrane: Membrane descriptor
        :rna_sequence: RNA descriptor
        """
        if protein_s is not None:
            assert isinstance(protein_s, Protein)
        if protein_e is not None:
            assert isinstance(protein_e, Protein)
        if protein_m is not None:
            assert isinstance(protein_m, Protein)
        if membrane is not None:
            assert isinstance(membrane, ParametricMembrane)
        if rna_sequence is not None:
            assert isinstance(rna_sequence, RNASequence)
        assert isinstance(assembly_params, list)
        self.name = name
        self.protein_s = protein_s
        self.protein_e = protein_e
        self.protein_m = protein_m
        self.membrane = membrane
        self.rna_sequence = rna_sequence
        self.assembly_params = assembly_params
