# !/usr/bin/env python
"""BioExplorer class"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2022 Blue BrainProject / EPFL
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
import os

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
# pylint: disable=too-many-public-methods
# pylint: disable=inconsistent-return-statements
# pylint: disable=wrong-import-order
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=missing-return-type-doc
# pylint: disable=missing-return-doc
# pylint: disable=missing-raises-doc


class AnimationParams:
    """
    Parameters used to introduce some randomness in the position and orientation of the protein.

    This is mainly used to make assemblies more realistic, and for animation purpose too.
    """

    def __init__(self, seed=0, position_seed=0, position_strength=0.0, rotation_seed=0,
                 rotation_strength=0.0, morphing_step=0.0):
        """
        Animation parameters are used to define how molecules should be animated

        :seed: (int, optional): Randomization seed. Defaults to 0.
        :position_seed: (int, optional): Randomization seed for the position of the molecule.
        Defaults to 0.
        :position_strength: (float, optional): Strength of the position alteration. Defaults to 0.0.
        :rotation_seed: (int, optional): Randomization seed for the rotation of the molecule.
        Defaults to 0.
        :rotation_strength: (float, optional): Strength of the rotation alteration. Defaults to 0.0.
        :morphing_step: (float, optional): Morphing step between 0 and 1 for assemblies that
        transition from one shape to another. Defaults to 0.0.
        """
        self.seed = seed
        self.position_seed = position_seed
        self.position_strength = position_strength
        self.rotation_seed = rotation_seed
        self.rotation_strength = rotation_strength
        self.morphing_step = morphing_step

    def to_list(self):
        """
        A list containing the values of class members

        :return: A list containing the values of class members
        :rtype: list
        """
        return [self.seed, self.position_seed, self.position_strength, self.rotation_seed,
                self.rotation_strength, self.morphing_step]

    def copy(self):
        """
        Copy the current object

        :return: AnimationParams: A copy of the object
        """
        return AnimationParams(self.seed, self.position_seed, self.position_strength,
                               self.rotation_seed, self.rotation_strength, self.morphing_step)


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

    def copy(self):
        """
        Copy the current object

        :return: Vector3: A copy of the object
        """
        return Vector3(self.x, self.y, self.z)


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

    def copy(self):
        """
        Copy the current object

        :return: Vector2: A copy of the object
        """
        return Vector2(self.x, self.y)


class BioExplorer:
    """Blue Brain BioExplorer"""

    PLUGIN_API_PREFIX = 'be-'
    CONTENTS_DELIMITER = '||||'

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
    REPRESENTATION_MESH = 6

    ASSEMBLY_SHAPE_POINT = 0
    ASSEMBLY_SHAPE_EMPTY_SPHERE = 1
    ASSEMBLY_SHAPE_PLANE = 2
    ASSEMBLY_SHAPE_SINUSOID = 3
    ASSEMBLY_SHAPE_CUBE = 4
    ASSEMBLY_SHAPE_FAN = 5
    ASSEMBLY_SHAPE_BEZIER = 6
    ASSEMBLY_SHAPE_MESH = 7
    ASSEMBLY_SHAPE_HELIX = 8
    ASSEMBLY_SHAPE_FILLED_SPHERE = 9
    ASSEMBLY_SHAPE_CELL_DIFFUSION = 10

    NAME_PROTEIN_S_OPEN = "Protein S (open)"
    NAME_PROTEIN_S_CLOSED = "Protein S (closed)"
    NAME_PROTEIN_M = "Protein M"
    NAME_PROTEIN_E = "Protein E"
    NAME_RNA_SEQUENCE = "RNA sequence"
    NAME_MEMBRANE = "Membrane"
    NAME_TRANS_MEMBRANE = "Trans-membrane"
    NAME_RECEPTOR = "Receptor"
    NAME_PROTEIN = "Protein"
    NAME_ION_CHANNEL = "IonChannel"

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

    POSITION_CONSTRAINT_INSIDE = 0
    POSITION_CONSTRAINT_OUTSIDE = 1

    VASCULATURE_REPRESENTATION_GRAPH = 0
    VASCULATURE_REPRESENTATION_SECTION = 1
    VASCULATURE_REPRESENTATION_SEGMENT = 2

    VASCULATURE_COLOR_SCHEME_NONE = 0
    VASCULATURE_COLOR_SCHEME_NODE = 1
    VASCULATURE_COLOR_SCHEME_SECTION = 2
    VASCULATURE_COLOR_SCHEME_SUBGRAPH = 3
    VASCULATURE_COLOR_SCHEME_PAIR = 4
    VASCULATURE_COLOR_SCHEME_ENTRYNODE = 5
    VASCULATURE_COLOR_SCHEME_RADIUS = 6
    VASCULATURE_COLOR_SCHEME_SECTION_POINTS = 7
    VASCULATURE_COLOR_SCHEME_SECTION_ORIENTATION = 8

    MORPHOLOGY_COLOR_SCHEME_NONE = 0
    MORPHOLOGY_COLOR_SCHEME_SECTION_TYPE = 1
    MORPHOLOGY_COLOR_SCHEME_SECTION_ORIENTATION = 2

    POPULATION_COLOR_SCHEME_NONE = 0
    POPULATION_COLOR_SCHEME_ID = 1

    MORPHOLOGY_REPRESENTATION_GRAPH = 0
    MORPHOLOGY_REPRESENTATION_SECTION = 1
    MORPHOLOGY_REPRESENTATION_SEGMENT = 2
    MORPHOLOGY_REPRESENTATION_ORIENTATION = 3

    # Material offsets in neurons
    NB_MATERIALS_PER_MORPHOLOGY = 10
    NEURON_MATERIAL_VARICOSITY = 0
    NEURON_MATERIAL_SOMA = 1
    NEURON_MATERIAL_AXON = 2
    NEURON_MATERIAL_BASAL_DENDRITE = 3
    NEURON_MATERIAL_APICAL_DENDRITE = 4
    NEURON_MATERIAL_AFFERENT_SYNPASE = 5
    NEURON_MATERIAL_EFFERENT_SYNPASE = 6
    NEURON_MATERIAL_MITOCHONDRION = 7
    NEURON_MATERIAL_NUCLEUS = 8
    NEURON_MATERIAL_MYELIN_STEATH = 9

    def __init__(self, url='localhost:5000'):
        """Create a new BioExplorer instance"""
        self._url = url
        self._client = Client(url)
        self._v1_compatibility = False

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

    @staticmethod
    def _check(response):
        if not response["status"]:
            raise RuntimeError(response["contents"])
        return response

    def _invoke(self, method, params=None):
        prefixed_method = self.PLUGIN_API_PREFIX + method
        return self._client.rockets_client.request(
            method=prefixed_method, params=params)

    def _invoke_and_check(self, method, params=None):
        prefixed_method = self.PLUGIN_API_PREFIX + method
        response = self._client.rockets_client.request(
            method=prefixed_method, params=params)
        return self._check(response)

    def version(self):
        """
        Version of the BioExplorer application

        :rtype: string
        """
        if self._client is None:
            return __version__

        result = self._invoke_and_check("get-version")
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

        return self._invoke_and_check("get-scene-information")

    @ staticmethod
    def authors():
        """
        List of authors

        :rtype: string
        """
        return "Cyrille Favreau (cyrille.favreau@epfl.ch)"

    def reset_scene(self):
        """
        Remove all assemblies

        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        return self._invoke_and_check("reset-scene")

    def reset_camera(self):
        """
        Reset the camera for the current scene

        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        return self._invoke_and_check("reset-camera")

    def set_focus_on(self, model_id, instance_id=0, distance=0.0,
                     direction=Vector3(0.0, 0.0, 1.0), max_number_of_instances=1e6):
        """
        Set focus on a specific instance of a model

        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        assert isinstance(direction, Vector3)

        params = dict()
        params["modelId"] = model_id
        params["instanceId"] = instance_id
        params["distance"] = distance
        params["direction"] = direction.to_list()
        params["maxNbInstances"] = max_number_of_instances
        result = self._invoke_and_check("set-focus-on", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

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
        result = self._invoke_and_check("export-to-file", params)
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
        result = self._invoke_and_check("export-to-database", params)
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
        result = self._invoke_and_check("import-from-cache", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def export_to_xyz(self, filename, file_format, low_bounds=Vector3(-1e38, -1e38, -1e38),
                      high_bounds=Vector3(1e38, 1e38, 1e38)):
        """
        Exports current scene to file as a XYZ file

        :filename: Full path of the XYZ file
        :file_format: Defines the format of the XYZ file
        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        params = dict()
        params["filename"] = filename
        params["lowBounds"] = low_bounds.to_list()
        params["highBounds"] = high_bounds.to_list()
        params["fileFormat"] = file_format
        result = self._invoke_and_check("export-to-xyz", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def add_sars_cov_2(self, name, resource_folder,
                       shape_params=Vector3(45.0, 0.0, 0.0),
                       animation_params=AnimationParams(0, 1, 0.25, 1, 0.1),
                       nb_protein_s=62, nb_protein_m=50, nb_protein_e=42,
                       open_protein_s_indices=[0], atom_radius_multiplier=1.0,
                       add_glycans=False, add_rna_sequence=False,
                       representation=REPRESENTATION_ATOMS_AND_STICKS, clipping_planes=list(),
                       position=Vector3(), rotation=Quaternion(), apply_colors=False):
        """
        Add a virus with the default SARS-COV-2 coronavirus parameters

        :name: Name of the SARS-COV-2 coronavirus
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
        :rotation: rotation of the SARS-COV-2 coronavirus in the scene
        :apply_colors: Applies default colors to the virus
        """
        pdb_folder = os.path.join(resource_folder, "pdb")
        rna_folder = os.path.join(resource_folder, "rna")
        glycan_folder = os.path.join(pdb_folder, "glycans")
        membrane_folder = os.path.join(pdb_folder, "membrane")

        membrane_proteins = list()

        open_conformation_indices = open_protein_s_indices
        closed_conformation_indices = list()
        for i in range(nb_protein_s):
            if i not in open_conformation_indices:
                closed_conformation_indices.append(i)

        ap = animation_params.copy()

        # Protein S (open)
        ap.seed = 0
        membrane_proteins.append(Protein(
            name=name + '_' + self.NAME_PROTEIN_S_OPEN,
            source=os.path.join(pdb_folder, "6vyb.pdb"),
            occurrences=nb_protein_s,
            rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            allowed_occurrences=open_conformation_indices,
            transmembrane_params=Vector2(10.5, 10.5),
            animation_params=ap
        ))

        # Protein S (closed)
        membrane_proteins.append(Protein(
            name=name + '_' + self.NAME_PROTEIN_S_CLOSED,
            source=os.path.join(pdb_folder, "sars-cov-2-v1.pdb"),
            occurrences=nb_protein_s,
            rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            allowed_occurrences=closed_conformation_indices,
            transmembrane_params=Vector2(10.5, 10.5),
            animation_params=ap
        ))

        # Protein M (QHD43419)
        ap.seed = 1
        membrane_proteins.append(Protein(
            name=name + '_' + self.NAME_PROTEIN_M,
            source=os.path.join(pdb_folder, "QHD43419a.pdb"),
            occurrences=nb_protein_m,
            position=Vector3(2.5, 0.0, 0.0),
            rotation=Quaternion(0.135, 0.99, 0.0, 0.0),
            transmembrane_params=Vector2(0.5, 2.0),
            animation_params=ap
        ))

        # Protein E (QHD43418 P0DTC4)
        ap.seed = 3
        membrane_proteins.append(Protein(
            name=name + '_' + self.NAME_PROTEIN_E,
            source=os.path.join(pdb_folder, "QHD43418a.pdb"),
            occurrences=nb_protein_e,
            position=Vector3(2.5, 0.0, 0.0),
            rotation=Quaternion(0.0, 0.707, 0.707, 0.0),
            transmembrane_params=Vector2(0.5, 2.0),
            animation_params=ap
        ))

        # Virus membrane
        ap.seed = 4
        lipid_sources = [
            os.path.join(membrane_folder, "segA.pdb"),
            os.path.join(membrane_folder, "segB.pdb"),
            os.path.join(membrane_folder, "segC.pdb"),
            os.path.join(membrane_folder, "segD.pdb")
        ]

        virus_membrane = Membrane(
            lipid_sources=lipid_sources,
            lipid_rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            load_bonds=True, load_non_polymer_chemicals=True,
            animation_params=ap
        )

        # Cell
        virus_cell = Cell(
            name=name,
            shape=self.ASSEMBLY_SHAPE_EMPTY_SPHERE,
            shape_params=shape_params, membrane=virus_membrane,
            proteins=membrane_proteins)

        self.add_cell(
            cell=virus_cell,
            atom_radius_multiplier=atom_radius_multiplier, representation=representation,
            position=position, rotation=rotation,
            clipping_planes=clipping_planes)

        # RNA Sequence
        if add_rna_sequence:
            params = Vector2(shape_params.x * 0.55, 0.5)
            rna_sequence = RNASequence(
                source=os.path.join(rna_folder, "sars-cov-2.rna"),
                protein_source=os.path.join(pdb_folder, "7bv1.pdb"),
                shape=self.RNA_SHAPE_TREFOIL_KNOT,
                shape_params=params,
                values_range=Vector2(-8.0 * math.pi, 8.0 * math.pi),
                curve_params=Vector3(1.51, 1.12, 1.93),
                atom_radius_multiplier=atom_radius_multiplier, representation=representation,
                animation_params=ap
            )
            self.add_rna_sequence(
                assembly_name=name,
                name=name + '_' + BioExplorer.NAME_RNA_SEQUENCE,
                rna_sequence=rna_sequence)

        if add_glycans:
            glycan_representation = representation
            if representation == BioExplorer.REPRESENTATION_MESH:
                glycan_representation = BioExplorer.REPRESENTATION_ATOMS_AND_STICKS

            complex_folder = os.path.join(glycan_folder, "complex")
            complex_paths = [
                os.path.join(complex_folder, "5.pdb"),
                os.path.join(complex_folder, "15.pdb"),
                os.path.join(complex_folder, "25.pdb"),
                os.path.join(complex_folder, "35.pdb"),
            ]

            high_mannose_folder = os.path.join(glycan_folder, "high-mannose")
            high_mannose_paths = [
                os.path.join(high_mannose_folder, "1.pdb"),
                os.path.join(high_mannose_folder, "2.pdb"),
                os.path.join(high_mannose_folder, "3.pdb"),
                os.path.join(high_mannose_folder, "4.pdb"),
            ]

            o_glycan_paths = [os.path.join(glycan_folder, "o-glycan", "12.pdb")]

            # High-mannose
            indices_closed = [61, 122, 234, 603, 709, 717, 801, 1074]
            indices_open = [61, 122, 234, 709, 717, 801, 1074]
            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=self.NAME_PROTEIN_S_CLOSED,
                paths=high_mannose_paths,
                indices=indices_closed,
                representation=glycan_representation,
                atom_radius_multiplier=atom_radius_multiplier,
                animation_params=AnimationParams(0, 0, 0.0,
                                                 animation_params.rotation_seed + 7,
                                                 animation_params.rotation_strength)
            )
            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=high_mannose_paths,
                indices=indices_open,
                representation=glycan_representation,
                atom_radius_multiplier=atom_radius_multiplier,
                animation_params=AnimationParams(0, 0, 0.0,
                                                 animation_params.rotation_seed + 7,
                                                 animation_params.rotation_strength)
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
                representation=glycan_representation,
                atom_radius_multiplier=atom_radius_multiplier,
                animation_params=AnimationParams(0, 0, 0.0,
                                                 animation_params.rotation_seed + 8,
                                                 2.0 * animation_params.rotation_strength)
            )

            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_COMPLEX,
                protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=complex_paths,
                indices=indices_open,
                representation=glycan_representation,
                atom_radius_multiplier=atom_radius_multiplier,
                animation_params=AnimationParams(0, 0, 0.0,
                                                 animation_params.rotation_seed + 8,
                                                 2.0 * animation_params.rotation_strength)
            )

            # O-Glycans
            for index in [323, 325]:
                o_glycan_name = (name + "_" + self.NAME_GLYCAN_O_GLYCAN + "_" +
                                 str(index))
                o_glycan = Sugar(
                    assembly_name=name,
                    name=o_glycan_name,
                    source=o_glycan_paths[0],
                    protein_name=name + "_" + self.NAME_PROTEIN_S_CLOSED,
                    site_indices=[index],
                    representation=glycan_representation,
                    atom_radius_multiplier=atom_radius_multiplier,
                    animation_params=AnimationParams(0, 0, 0.0,
                                                     animation_params.rotation_seed + 9,
                                                     2.0 * animation_params.rotation_strength)
                )
                self.add_sugar(o_glycan)

            # High-mannose glycans on Protein M
            indices = [5]
            protein_name = name + "_" + self.NAME_PROTEIN_M
            high_mannose_glycans = Sugar(
                rotation=Quaternion(0.707, 0.0, 0.0, 0.707),
                assembly_name=name,
                name=protein_name + "_" + self.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=protein_name,
                source=high_mannose_paths[0],
                site_indices=indices,
                representation=glycan_representation,
                animation_params=AnimationParams(0, 0, 0.0,
                                                 animation_params.rotation_seed + 10,
                                                 2.0 * animation_params.rotation_strength)
            )
            self.add_glycan(high_mannose_glycans)

            # Complex glycans on Protein E
            indices = [48, 66]
            protein_name = name + "_" + self.NAME_PROTEIN_E
            complex_glycans = Sugar(
                rotation=Quaternion(0.707, 0.0, 0.0, 0.707),
                assembly_name=name,
                name=protein_name + "_" + self.NAME_GLYCAN_COMPLEX,
                protein_name=protein_name,
                source=complex_paths[0],
                site_indices=indices,
                representation=glycan_representation,
                animation_params=AnimationParams(0, 0, 0.0,
                                                 animation_params.rotation_seed + 11,
                                                 2.0 * animation_params.rotation_strength)
            )
            self.add_glycan(complex_glycans)

        if apply_colors:
            # Apply default materials
            self.apply_default_color_scheme(
                shading_mode=self.SHADING_MODE_BASIC)

    def add_cell(self, cell, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                 clipping_planes=list(), position=Vector3(), rotation=Quaternion()):
        """
        Add a cell assembly to the scene

        :cell: Description of the cell
        :atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        :representation: Multiplies atom radius by the specified value
        :clipping_planes: List of clipping planes to apply to the virus assembly
        :position: Position of the cell in the scene
        :rotation: Rotation of the cell in the scene
        :animation_params: Seed used to randomise position the elements in the membrane
        """
        assert isinstance(cell, Cell)
        assert isinstance(clipping_planes, list)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)

        self.remove_assembly(cell.name)
        self.add_assembly(
            name=cell.name,
            shape=cell.shape,
            shape_params=cell.shape_params,
            shape_mesh_source=cell.shape_mesh_source,
            position=position,
            rotation=rotation,
            clipping_planes=clipping_planes
        )

        for protein in cell.proteins:
            if protein.occurrences != 0:
                _protein = AssemblyProtein(
                    assembly_name=cell.name,
                    name=protein.name,
                    source=protein.source,
                    load_non_polymer_chemicals=protein.
                    load_non_polymer_chemicals,
                    occurrences=protein.occurrences,
                    allowed_occurrences=protein.allowed_occurrences,
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=True,
                    representation=representation,
                    transmembrane_params=protein.transmembrane_params,
                    animation_params=protein.animation_params,
                    position=protein.position,
                    rotation=protein.rotation,
                    chain_ids=protein.chain_ids,
                )
                self.add_assembly_protein(_protein)

        cell.membrane.representation = representation
        cell.membrane.atom_radius_multiplier = atom_radius_multiplier
        return self.add_membrane(
            assembly_name=cell.name,
            name=cell.name + "_" + self.NAME_MEMBRANE,
            animation_params=cell.membrane.animation_params,
            membrane=cell.membrane
        )

    def add_volume(self, volume, atom_radius_multiplier=1.0,
                   representation=REPRESENTATION_ATOMS_AND_STICKS, position=Vector3(),
                   rotation=Quaternion(), constraints=list()):
        """
        Add a volume assembly to the scene

        :volume: Description of the volume
        :atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        :representation: Multiplies atom radius by the specified value
        :position: Position of the volume in the scene
        :rotation: Rotation of the volume in the scene
        :animation_params: Random seed used to define the positions of the proteins in the volume
        :constraints: List of assemblies that constraint the placememnt of the proteins
        """
        assert isinstance(volume, Volume)
        assert isinstance(position, Vector3)
        assert isinstance(constraints, list)

        _protein = AssemblyProtein(
            assembly_name=volume.name,
            name=volume.protein.name,
            source=volume.protein.source,
            load_non_polymer_chemicals=volume.protein.
            load_non_polymer_chemicals,
            occurrences=volume.protein.occurrences,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=volume.protein.load_bonds,
            load_hydrogen=volume.protein.load_hydrogen,
            representation=representation,
            animation_params=volume.protein.animation_params,
            position=volume.protein.position,
            rotation=volume.protein.rotation,
            constraints=constraints
        )

        self.remove_assembly(volume.name)
        result = self.add_assembly(
            name=volume.name,
            shape=volume.shape, shape_params=volume.shape_params,
            position=position, rotation=rotation)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        result = self.add_assembly_protein(_protein)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def add_surfactant(self, surfactant, atom_radius_multiplier=1.0,
                       representation=REPRESENTATION_ATOMS, position=Vector3(),
                       rotation=Quaternion(), animation_params=AnimationParams()):
        """
        Add a surfactant assembly to the scene

        :surfactant: Description of the surfactant
        :atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        :representation: Multiplies atom radius by the specified value
        :position: Position of the volume in the scene
        :rotation: rotation of the cell in the scene
        :animation_params: Random seed used to define the shape of the branches
        """
        assert isinstance(surfactant, Surfactant)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(animation_params, AnimationParams)

        shape = self.ASSEMBLY_SHAPE_EMPTY_SPHERE
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

        d = collagen_size * (nb_collagens + 1) - 9.0
        p = Vector3(0.0, 0.0, d)
        r = Quaternion(-0.343, 0.883, 0.115, 0.303)
        if self._v1_compatibility:
            p = Vector3(0.0, 0.0, -d)
            r = Quaternion(0.115, -0.297, 0.344, 0.883)

        protein_sp_d = AssemblyProtein(
            assembly_name=surfactant.name,
            name=head_name,
            source=surfactant.head_source,
            occurrences=nb_branches,
            atom_radius_multiplier=atom_radius_multiplier,
            animation_params=animation_params,
            representation=representation,
            position=p, rotation=r
        )

        collagens = list()
        for i in range(nb_collagens):
            d = collagen_size * (i + 1) - 7.0
            p = Vector3(0.0, 0.0, d)
            if self._v1_compatibility:
                p = Vector3(0.0, 0.0, -d)

            collagens.append(
                AssemblyProtein(
                    assembly_name=surfactant.name,
                    name=branch_name + str(i),
                    position=p, atom_radius_multiplier=atom_radius_multiplier,
                    source=surfactant.branch_source,
                    occurrences=nb_branches,
                    animation_params=animation_params,
                    representation=representation
                ))

        self.remove_assembly(surfactant.name)
        result = self.add_assembly(name=surfactant.name,
                                   shape=shape,
                                   shape_params=Vector3(),
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

    def add_enzyme_reaction(self, enzyme_reaction):
        """
        Add an enzyme reaction to the scene

        :enzyme_reaction: Description of the enzyme reaction
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(enzyme_reaction, EnzymeReaction)

        self.remove_assembly(enzyme_reaction.assembly_name)
        result = self.add_assembly(name=enzyme_reaction.assembly_name,
                                   shape=BioExplorer.ASSEMBLY_SHAPE_POINT,
                                   shape_params=Vector3())
        if not result["status"]:
            raise RuntimeError(result["contents"])

        substrate_names = ''
        for substrate in enzyme_reaction.substrates:
            assert isinstance(substrate, Protein)
            if substrate_names != '':
                substrate_names += BioExplorer.CONTENTS_DELIMITER
            substrate_names += substrate.name

        product_names = ''
        for product in enzyme_reaction.products:
            assert isinstance(product, Protein)
            if product_names != '':
                product_names += BioExplorer.CONTENTS_DELIMITER
            product_names += product.name

        params = dict()
        params["assemblyName"] = enzyme_reaction.assembly_name
        params["name"] = enzyme_reaction.name
        params["enzymeName"] = enzyme_reaction.enzyme.name
        params["substrateNames"] = substrate_names
        params["productNames"] = product_names
        return self._invoke_and_check("add-enzyme-reaction", params)

    def set_enzyme_reaction_progress(self, enzyme_reaction, instance_id=0, progress=0.0):
        """
        Set the progress of a given enzyme reaction

        :enzyme_reaction: EnzymeReaction object
        :instance_id: Enzyme protein instance. Defaults to 0.
        :progress: Progress of the enzyme reaction (between 0 and 1). Defaults to 0.
        :return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params["assemblyName"] = enzyme_reaction.assembly_name
        params["name"] = enzyme_reaction.name
        params["instanceId"] = instance_id
        params["progress"] = progress
        return self._invoke_and_check("set-enzyme-reaction-progress", params)

    @ staticmethod
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

    def add_assembly(self, name, shape=ASSEMBLY_SHAPE_POINT, shape_params=Vector3(),
                     shape_mesh_source='', clipping_planes=list(), position=Vector3(),
                     rotation=Quaternion()):
        """
        Add an assembly to the scene

        :name: Name of the assembly
        :clipping_planes: List of clipping planes to apply to the virus assembly
        :position: Position of the scene in the scene
        :rotation: rotation of the assembly in the scene
        """
        assert isinstance(clipping_planes, list)
        assert isinstance(shape_params, Vector3)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)

        clipping_planes_values = list()
        for plane in clipping_planes:
            for i in range(4):
                clipping_planes_values.append(plane[i])

        mesh_contents = ''
        if shape_mesh_source:
            mesh_contents = ''.join(open(shape_mesh_source).readlines())

        params = dict()
        params["name"] = name
        params["shape"] = shape
        params["shapeParams"] = shape_params.to_list()
        params["shapeMeshContents"] = mesh_contents
        params["position"] = position.to_list()
        params["rotation"] = list(rotation)
        params["clippingPlanes"] = clipping_planes_values
        return self._invoke_and_check("add-assembly", params)

    def remove_assembly(self, name):
        """
        Removes the specified assembly

        :name: Name of the assembly
        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        params = dict()
        params['name'] = name
        params['shape'] = BioExplorer.ASSEMBLY_SHAPE_POINT
        params['shapeParams'] = list()
        params["shapeMeshContents"] = ''
        params['position'] = Vector3().to_list()
        params["rotation"] = list(Quaternion())
        params["clippingPlanes"] = list()
        return self._invoke_and_check("remove-assembly", params)

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
        result = self._invoke_and_check("set-protein-color-scheme", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
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
        result = self._invoke_and_check("set-protein-amino-acid-sequence-as-string", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
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
        result = self._invoke_and_check("set-protein-amino-acid-sequence-as-ranges", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
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
        result = self._invoke_and_check("get-protein-amino-acid-information", params)
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
        result = self._invoke_and_check("set-protein-amino-acid", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
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
        result = self._invoke_and_check("set-protein-instance-transformation", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
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
        result = self._invoke_and_check("get-protein-instance-transformation", params)
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
        values_range = Vector2(0.0, 2.0 * math.pi)
        if rna_sequence.values_range is None:
            # Defaults
            if rna_sequence.shape == self.RNA_SHAPE_TORUS:
                values_range = Vector2(0.0, 2.0 * math.pi)
            elif rna_sequence.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                values_range = Vector2(0.0, 4.0 * math.pi)
        else:
            values_range = rna_sequence.values_range

        curve_params = [1.0, 1.0, 1.0]
        if rna_sequence.shape_params is None:
            # Defaults
            if rna_sequence.shape == self.RNA_SHAPE_TORUS:
                curve_params = Vector3(0.5, 10.0, 0.0)
            elif rna_sequence.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                curve_params = Vector3(2.5, 2.0, 2.2)

        else:
            curve_params = rna_sequence.curve_params

        protein_contents = ''
        if rna_sequence.protein_source:
            protein_contents = "".join(open(rna_sequence.protein_source).readlines())

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["pdbId"] = os.path.splitext(os.path.basename(rna_sequence.protein_source))[0].lower()
        params["contents"] = "".join(open(rna_sequence.source).readlines())
        params["proteinContents"] = protein_contents
        params["shape"] = rna_sequence.shape
        params["shapeParams"] = rna_sequence.shape_params.to_list()
        params["valuesRange"] = values_range.to_list()
        params["curveParams"] = curve_params.to_list()
        params["atomRadiusMultiplier"] = rna_sequence.atom_radius_multiplier
        params["representation"] = rna_sequence.representation
        params["animationParams"] = rna_sequence.animation_params.to_list()
        params["position"] = rna_sequence.position.to_list()
        params["rotation"] = list(rna_sequence.rotation)
        return self._invoke_and_check("add-rna-sequence", params)

    def add_membrane(self, assembly_name, name, membrane, animation_params=AnimationParams()):
        """
        Add a membrane to the scene

        :assembly_name: Name of the assembly
        :name: Name of the cell
        :membrane: Description of the membrane
        :shape: Shape of the membrane
        :shape_params: Size of the membrane
        :animation_params: Seed used to randomise position the elements in the membrane
        :rotation: rotation of the proteins in the membrane
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(membrane, Membrane)
        assert isinstance(animation_params, AnimationParams)

        lipid_pdb_ids = ''
        lipid_contents = ''
        for lipid_source in membrane.lipid_sources:
            if lipid_contents != '':
                lipid_contents += BioExplorer.CONTENTS_DELIMITER
            if lipid_pdb_ids != '':
                lipid_pdb_ids += BioExplorer.CONTENTS_DELIMITER
            lipid_contents += ''.join(open(lipid_source).readlines())
            lipid_pdb_ids += os.path.splitext(os.path.basename(lipid_source))[0].lower()

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["lipidPDBIds"] = lipid_pdb_ids
        params["lipidContents"] = lipid_contents
        params["lipidRotation"] = list(membrane.lipid_rotation)
        params["lipidDensity"] = membrane.lipid_density
        params["atomRadiusMultiplier"] = membrane.atom_radius_multiplier
        params["loadBonds"] = membrane.load_bonds
        params["loadNonPolymerChemicals"] = membrane.load_non_polymer_chemicals
        params["representation"] = membrane.representation
        params["chainIds"] = membrane.chain_ids
        params["recenter"] = membrane.recenter
        params["animationParams"] = animation_params.to_list()
        return self._invoke_and_check("add-membrane", params)

    def add_protein(self, protein, recenter=True, representation=REPRESENTATION_ATOMS_AND_STICKS,
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
        assert isinstance(protein, Protein)

        assembly_name = protein.name
        self.remove_assembly(assembly_name)
        self.add_assembly(
            name=assembly_name,
            position=position, rotation=rotation)

        _protein = AssemblyProtein(
            assembly_name=assembly_name,
            name=protein.name, recenter=recenter,
            source=protein.source,
            load_hydrogen=protein.load_hydrogen,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=protein.load_bonds,
            load_non_polymer_chemicals=protein.load_non_polymer_chemicals,
            representation=representation,
            position=protein.position,
            rotation=protein.rotation, chain_ids=protein.chain_ids
        )
        return self.add_assembly_protein(_protein)

    def add_assembly_protein(self, protein):
        """
        Add a protein to an assembly

        :protein: Description of the protein
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(protein, AssemblyProtein)

        constraints = ''
        for constraint in protein.constraints:
            if constraints != '':
                constraints += self.CONTENTS_DELIMITER
            if constraint[0] == BioExplorer.POSITION_CONSTRAINT_INSIDE:
                constraints += '+' + constraint[1]
            elif constraint[0] == BioExplorer.POSITION_CONSTRAINT_OUTSIDE:
                constraints += '-' + constraint[1]
            else:
                raise RuntimeError("Unknown position constraint")

        params = dict()
        params["assemblyName"] = protein.assembly_name
        params["name"] = protein.name
        params["pdbId"] = protein.pdb_id
        params["contents"] = "".join(open(protein.source).readlines())
        params["atomRadiusMultiplier"] = protein.atom_radius_multiplier
        params["loadBonds"] = protein.load_bonds
        params["loadNonPolymerChemicals"] = protein.load_non_polymer_chemicals
        params["loadHydrogen"] = protein.load_hydrogen
        params["representation"] = protein.representation
        params["chainIds"] = protein.chain_ids
        params["recenter"] = protein.recenter
        params["transmembraneParams"] = protein.transmembrane_params.to_list()
        params["occurrences"] = protein.occurrences
        params["allowedOccurrences"] = protein.allowed_occurrences
        params["animationParams"] = protein.animation_params.to_list()
        params["position"] = protein.position.to_list()
        params["rotation"] = list(protein.rotation)
        params["constraints"] = constraints
        return self._invoke_and_check("add-protein", params)

    def add_glycan(self, glycan):
        """
        Add glycan to an protein in an assembly

        :glycan: Description of the glycan
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(glycan, Sugar)

        params = dict()
        params["assemblyName"] = glycan.assembly_name
        params["name"] = glycan.name
        params["pdbId"] = glycan.pdb_id
        params["contents"] = glycan.contents
        params["proteinName"] = glycan.protein_name
        params["atomRadiusMultiplier"] = glycan.atom_radius_multiplier
        params["loadBonds"] = glycan.load_bonds
        params["representation"] = glycan.representation
        params["recenter"] = glycan.recenter
        params["chainIds"] = glycan.chain_ids
        params["siteIndices"] = glycan.site_indices
        params["rotation"] = list(glycan.rotation)
        params["animationParams"] = glycan.animation_params.to_list()
        return self._invoke_and_check("add-glycan", params)

    def add_multiple_glycans(
            self, assembly_name, glycan_type, protein_name, paths, representation, chain_ids=list(),
            indices=list(), load_bonds=True, atom_radius_multiplier=1.0, rotation=Quaternion(),
            animation_params=AnimationParams()):
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
        :shape_params: Extra optional parameters for positioning on the protein
        """
        assert isinstance(chain_ids, list)
        assert isinstance(indices, list)
        assert isinstance(rotation, Quaternion)
        assert isinstance(animation_params, AnimationParams)

        path_index = 0
        for source in paths:
            site_indices = list()
            if indices is not None:
                for index in indices:
                    if index % len(paths) == path_index:
                        site_indices.append(index)

            if site_indices:
                _glycan = Sugar(
                    assembly_name=assembly_name,
                    name=assembly_name + "_" + protein_name + "_" +
                    glycan_type + "_" + str(path_index),
                    source=source,
                    protein_name=assembly_name + "_" + protein_name,
                    chain_ids=chain_ids,
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=load_bonds,
                    representation=representation,
                    recenter=True,
                    site_indices=site_indices,
                    rotation=rotation,
                    animation_params=animation_params
                )
                self.add_glycan(_glycan)
            path_index += 1

    def add_sugar(self, sugar):
        """
        Add sugar to a protein in an assembly

        :sugar: Description of the sugars
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(sugar, Sugar)

        params = dict()
        params["assemblyName"] = sugar.assembly_name
        params["name"] = sugar.name
        params["pdbId"] = sugar.pdb_id
        params["contents"] = sugar.contents
        params["proteinName"] = sugar.protein_name
        params["atomRadiusMultiplier"] = sugar.atom_radius_multiplier
        params["loadBonds"] = sugar.load_bonds
        params["representation"] = sugar.representation
        params["recenter"] = sugar.recenter
        params["chainIds"] = sugar.chain_ids
        params["siteIndices"] = sugar.site_indices
        params["rotation"] = list(sugar.rotation)
        params["animationParams"] = sugar.animation_params.to_list()
        return self._invoke_and_check("add-sugar", params)

    def set_rendering_quality(self, image_quality):
        """
        Set rendering quality using presets

        :image_quality: Quality of the image (RENDERING_QUALITY_LOW or RENDERING_QUALITY_HIGH)
        :return: Result of the call to the BioExplorer backend
        """
        if image_quality == self.RENDERING_QUALITY_HIGH:
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
        assert isinstance(model_id, int)

        params = dict()
        params["modelId"] = model_id
        params["maxNbInstances"] = 0
        return self._invoke("get-model-name", params)

    def get_model_ids(self):
        """
        Return the list of model ids in the current scene

        :return: List of model Ids
        """
        return self._invoke("get-model-ids")

    def get_model_instances(self, model_id, max_number_of_instances=1e6):
        """
        Return the list of instances in the specified model

        :model_id: Id of the model
        :max_number_of_instances: Maximum number of instances (this can be huge and difficult to
        handle by the python code)
        :return: List of instance Ids
        """
        assert isinstance(model_id, int)
        assert isinstance(max_number_of_instances, int)

        params = dict()
        params["modelId"] = model_id
        params["maxNbInstances"] = max_number_of_instances
        return self._invoke("get-model-instances", params)

    def get_material_ids(self, model_id):
        """
        Return the list of material Ids for a given model

        :model_id: Id of the model
        :return: List of material Ids
        """
        assert isinstance(model_id, int)

        params = dict()
        params["modelId"] = model_id
        params["maxNbInstances"] = 0
        return self._invoke("get-material-ids", params)

    def set_materials(self, model_ids, material_ids, diffuse_colors, specular_colors,
                      specular_exponents=list(), opacities=list(), reflection_indices=list(),
                      refraction_indices=list(), glossinesses=list(), shading_modes=list(),
                      emissions=list(), cast_user_datas=list(), user_parameters=list(),
                      chameleon_modes=list()):
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
        :cast_user_datas: List of user data casts
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

        d_colors = list()
        for diffuse in diffuse_colors:
            for k in range(3):
                d_colors.append(diffuse[k])

        s_colors = list()
        for specular in specular_colors:
            for k in range(3):
                s_colors.append(specular[k])

        params = dict()
        params["modelIds"] = model_ids
        params["materialIds"] = material_ids
        params["diffuseColors"] = d_colors
        params["specularColors"] = s_colors
        params["specularExponents"] = specular_exponents
        params["reflectionIndices"] = reflection_indices
        params["opacities"] = opacities
        params["refractionIndices"] = refraction_indices
        params["emissions"] = emissions
        params["glossinesses"] = glossinesses
        params["castUserData"] = cast_user_datas
        params["shadingModes"] = shading_modes
        params["userParameters"] = user_parameters
        params["chameleonModes"] = chameleon_modes
        return self._invoke_and_check("set-materials", params)

    def set_material_extra_attributes(self, model_id):
        """
        Set extra BioExplorer specific attributes to materials

        :model_id: ID of the model
        :return: Result of the request submission
        """
        if self._client is None:
            return

        params = dict()
        params["modelId"] = model_id
        params["maxNbInstances"] = 1
        return self._invoke_and_check("set-material-extra-attributes", params)

    def set_materials_from_palette(self, model_ids, material_ids, palette, shading_mode,
                                   specular_exponent, user_parameter=1.0, glossiness=1.0,
                                   emission=0.0, opacity=1.0, reflection_index=0.0,
                                   refraction_index=1.0, cast_user_data=False,
                                   chameleon_mode=SHADING_CHAMELEON_MODE_NONE):
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
        assert isinstance(specular_exponent, float)
        assert isinstance(user_parameter, float)
        assert isinstance(glossiness, float)
        assert isinstance(emission, float)
        assert isinstance(opacity, float)
        assert isinstance(reflection_index, float)
        assert isinstance(refraction_index, float)
        assert isinstance(opacity, float)
        assert isinstance(cast_user_data, bool)

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
        cast_user_datas = list()
        for color in palette:
            colors.append(color)
            shading_modes.append(shading_mode)
            user_parameters.append(user_parameter)
            specular_exponents.append(specular_exponent)
            glossinesses.append(glossiness)
            emissions.append(emission)
            opacities.append(opacity)
            reflection_indices.append(reflection_index)
            refraction_indices.append(refraction_index)
            chameleon_modes.append(chameleon_mode)
            cast_user_datas.append(cast_user_data)
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
            chameleon_modes=chameleon_modes,
            cast_user_datas=cast_user_datas
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
                    specular_exponent=specular_exponent
                )
            elif glycans and self.NAME_GLYCAN_COMPLEX in model_name:
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
                    specular_exponent=specular_exponent
                )
            elif glycans and self.NAME_GLYCAN_HYBRID in model_name:
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
                    specular_exponent=specular_exponent
                )
            elif glycans and self.NAME_GLYCAN_O_GLYCAN in model_name:
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
                    specular_exponent=specular_exponent
                )
            elif membranes and self.NAME_MEMBRANE in model_name:
                palette = sns.color_palette("gist_heat", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent
                )
            elif proteins and (self.NAME_RECEPTOR in model_name or
                               self.NAME_TRANS_MEMBRANE in model_name or
                               self.NAME_ION_CHANNEL in model_name or
                               self.NAME_RNA_SEQUENCE in model_name):
                palette = sns.color_palette("OrRd_r", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent
                )
            elif proteins and (self.NAME_PROTEIN_S_CLOSED in model_name
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
                    specular_exponent=specular_exponent
                )
            elif proteins and self.NAME_GLUCOSE in model_name:
                palette = sns.color_palette("Blues", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent
                )
            elif proteins and self.NAME_LACTOFERRIN in model_name:
                palette = sns.color_palette("afmhot", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent
                )
            elif proteins and self.NAME_DEFENSIN in model_name:
                palette = sns.color_palette("plasma_r", nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent
                )
            elif proteins and (self.NAME_SURFACTANT_HEAD in model_name):
                palette = sns.color_palette("OrRd_r", nb_materials)
                emission = 0.0
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
                    specular_exponent=specular_exponent
                )
            elif collagen and self.NAME_COLLAGEN in model_name:
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

    def set_material_colors_from_id(self, model_id):
        """
        Set material color according to material id.
        
        The id of the model represents a 16bit RGB value for which the corresponding color is set

        :model_id: Id of the model
        """
        if self._client is None:
            return

        material_ids = self.get_material_ids(model_id)['ids']
        palette = list()
        for material_id in material_ids:
            r = (material_id >> 16) & 255
            g = (material_id >> 8) & 255
            b = material_id & 255
            palette.append([r / 255.0, g / 255.0, b / 255.0])
        return self.set_materials(
            [model_id], material_ids,
            diffuse_colors=palette,
            specular_colors=palette)

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
        if self._client is None:
            return

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

        # Build fields acceleration structures
        result = self.build_fields(voxel_size=voxel_size, density=density)
        fields_model_id = int(result['contents'])

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
        return self._invoke_and_check("build-fields", params)

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
        return self._invoke_and_check("import-fields-from-file", params)

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
        return self._invoke_and_check("export-fields-to-file", params)

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
        return self._invoke_and_check("add-grid", params)

    def add_bounding_box(self, name, bottom_left_corner, top_right_corner, radius=1.0,
                         color=Vector3(1.0, 1.0, 1.0)):
        """
        Add a bounding box to the scene

        :bottom_left_corner: Bottom left corner
        :top_right_corner: Top right corner
        :radius: Radius of box lines
        :color: Color of the bounding box
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(bottom_left_corner, Vector3)
        assert isinstance(top_right_corner, Vector3)
        assert isinstance(radius, float)
        assert isinstance(color, Vector3)
        params = dict()
        params["name"] = name
        params["bottomLeft"] = bottom_left_corner.to_list()
        params["topRight"] = top_right_corner.to_list()
        params["radius"] = radius
        params["color"] = color.to_list()
        return self._invoke_and_check("add-bounding-box", params)

    def add_sphere(self, name, position, radius, color=Vector3(1.0, 1.0, 1.0), opacity=1.0):
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
        params["opacity"] = opacity
        return self._invoke_and_check("add-sphere", params)

    def add_cone(
            self, name, origin, target, origin_radius, target_radius,
            color=Vector3(1.0, 1.0, 1.0), opacity=1.0):
        """
        Add a cone to the scene

        :name: Name of the cone
        :origin: Origin of the cone
        :target: Target of the cone
        :origin_radius: Origin radius of the cone
        :target_radius: target radius of the cone
        :color: RGB Color of the cone (0..1)
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(origin, Vector3)
        assert isinstance(target, Vector3)
        assert isinstance(color, Vector3)

        params = dict()
        params["name"] = name
        params["origin"] = origin.to_list()
        params["target"] = target.to_list()
        params["originRadius"] = origin_radius
        params["targetRadius"] = target_radius
        params["color"] = color.to_list()
        params["opacity"] = opacity
        return self._invoke_and_check("add-cone", params)

    def add_sdf_demo(self):
        """
        Add an SDF demo model

        :return: Result of the request submission
        """
        if self._client is None:
            return

        return self._invoke_and_check("add-sdf-demo")

    def add_streamlines(self, name, indices, vertices, colors):
        """
        Add streamlines to the scene

        :name: Name of the streamlines
        :indices: Start indices of each streamline in the array of vertices
        :vertices: Vertices
        :colors: RGBA Color of each vertex
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(indices, list)
        assert isinstance(vertices, list)
        assert isinstance(colors, list)

        params = dict()
        params["name"] = name
        params["indices"] = indices
        params["vertices"] = vertices
        params["colors"] = colors
        return self._invoke_and_check("add-streamlines", params)

    def set_general_settings(self, model_visibility_on_creation=True, mesh_folder='/tmp',
                             logging_level=0, v1_compatibility=False):
        """
        Set general settings for the plugin

        :model_visibility_on_creation: Visibility of the model on creation
        :off_folder: Folder where off files are stored (to avoid recomputation of molecular surface)
        :logging_level: Back-end logging level (0=no information logs, 3=full logging)
        :return: Result of the request submission
        """
        self._v1_compatibility = v1_compatibility
        params = dict()
        params["modelVisibilityOnCreation"] = model_visibility_on_creation
        params["meshFolder"] = mesh_folder
        params["loggingLevel"] = logging_level
        params["v1Compatibility"] = v1_compatibility
        response = self._invoke_and_check("set-general-settings", params)
        return response

    def set_models_visibility(self, visible):
        """
        Set the visibility of all models in the scene

        :visible: Visibility of the models
        :return: Result of the request submission
        """
        params = dict()
        params["visible"] = visible
        return self._invoke("set-models-visibility", params)

    def get_out_of_core_configuration(self):
        """
        Returns the out-of-core configuration

        :rtype: string
        """
        result = self._invoke_and_check("get-out-of-core-configuration")
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
        return self._invoke_and_check("get-out-of-core-progress")

    def get_out_of_core_average_loading_time(self):
        """
        Returns the out-of-core average loading time (per brick, in milliseconds)

        :rtype: float
        """
        return self._invoke_and_check("get-out-of-core-average-loading-time")

    def add_atlas(
            self, assembly_name,
            load_cells=True, cell_radius=1.0, load_meshes=False,
            region_sql_filter='', cell_sql_filter='', scale=Vector3(1.0, 1.0, 1.0),
            mesh_position=Vector3(), mesh_rotation=Quaternion(), mesh_scale=Vector3()):
        """
        Add a brain atlas the 3D scene

        :assembly_name: Name of the assembly to which the astrocytes should be added
        :load_cells: Load cells if set to true
        :cell_radius: Cell radius
        :load_meshes: Load meshes if set to true
        :region_sql_filter: Condition added to the SQL statement loading the regions
        :cell_sql_filter: Condition added to the SQL statement loading the cells
        :scale: Scale in the 3D scene

        :return: Result of the request submission
        """
        assert isinstance(scale, Vector3)
        assert isinstance(mesh_position, Vector3)
        assert isinstance(mesh_rotation, Quaternion)
        assert isinstance(mesh_scale, Vector3)

        params = dict()
        params["assemblyName"] = assembly_name
        params["loadCells"] = load_cells
        params["cellRadius"] = cell_radius
        params["loadMeshes"] = load_meshes
        params["cellSqlFilter"] = cell_sql_filter
        params["regionSqlFilter"] = region_sql_filter
        params["scale"] = scale.to_list()
        params["meshPosition"] = mesh_position.to_list()
        params["meshRotation"] = list(mesh_rotation)
        params["meshScale"] = mesh_scale.to_list()
        return self._invoke_and_check('add-atlas', params)

    def add_vasculature(
            self, assembly_name, population_name, color_scheme=VASCULATURE_COLOR_SCHEME_NONE,
            use_sdf=False, section_gids=list(),
            load_capilarities=False, representation=VASCULATURE_REPRESENTATION_SEGMENT,
            radius_multiplier=1.0, sql_filter='', scale=Vector3(1.0, 1.0, 1.0)):
        """
        Add a vasculature to the 3D scene

        :assembly_name: Name of the assembly to which the vasculature should be added
        :population_name Name of the node population
        :color_scheme: Color scheme applied to the vasculature (node, graph, section, etc)
        :use_sdf: Use sign distance fields geometry to create the vasculature. Defaults to False
        :node_gids: List of segment GIDs to load. Defaults to list()
        :representation: Representation of the vasculature geometry (Graph, sections or segments)
        :radius_muliplier: Applies the multiplier to all radii of the vasculature sections
        :sql_filter: Condition added to the SQL statement loading the vasculature
        :scale: Scale in the 3D scene

        :return: Result of the request submission
        """
        assert isinstance(section_gids, list)
        assert isinstance(scale, Vector3)

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["colorScheme"] = color_scheme
        params["useSdf"] = use_sdf
        params["gids"] = section_gids
        params["loadCapilarities"] = load_capilarities
        params["representation"] = representation
        params["radiusMultiplier"] = radius_multiplier
        params["sqlFilter"] = sql_filter
        params["scale"] = scale.to_list()
        return self._invoke_and_check('add-vasculature', params)

    def get_vasculature_info(self, assembly_name):
        """
        Return information about the vasculature

        :return: A dictionary of attributes (Model Id, Number of edges, pairs, sections and
                 sub-graphs)
        """
        params = dict()
        params["name"] = assembly_name
        response = self._invoke_and_check('get-vasculature-info', params)
        vasculature_info = dict()
        for param in response["contents"].split(self.CONTENTS_DELIMITER):
            s = param.split('=')
            vasculature_info[s[0]] = int(s[1])
        return vasculature_info

    def set_vasculature_report(self, assembly_name, population_name, report_simulation_id):
        """
        Attach a simulation report to the vasculature

        :assembly_name: Name of the assembly containing the vasculature
        :population_name: Name of the node population in the report. Defaults to 'vasculature'
        :report_simulation_id: Report simulation identifier

        :return: Result of the request submission
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["simulationReportId"] = report_simulation_id
        return self._invoke_and_check('set-vasculature-report', params)

    def set_vasculature_radius_report(
            self, assembly_name, population_name, report_simulation_id,
            frame, amplitude=1.0):
        """
        Attach a radius report to the vasculature

        :assembly_name: Name of the assembly containing the vasculature
        :population_name: Name of the node population in the report. Defaults to 'vasculature'
        :report_simulation_id: Report simulation identifier
        :frame: Index of the frame in the simulation report
        :amplitude: Amplitude of the radius modification. Defaults to 1.0

        :return: Result of the request submission
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["simulationReportId"] = report_simulation_id
        params['frame'] = frame
        params['amplitude'] = amplitude
        return self._invoke_and_check('set-vasculature-radius-report', params)

    def add_astrocytes(
            self, assembly_name, population_name, vasculature_population_name='',
            use_sdf=False, load_somas=True,
            load_dendrites=True, load_end_feet=True, generate_internals=False,
            morphology_representation=MORPHOLOGY_REPRESENTATION_SEGMENT,
            morphology_color_scheme=MORPHOLOGY_COLOR_SCHEME_NONE,
            population_color_scheme=POPULATION_COLOR_SCHEME_NONE,
            radius_multiplier=1.0, sql_filter='', scale=Vector3(1.0, 1.0, 1.0),
            animation_params=AnimationParams()):
        """
        Add a population of astrocytes to the 3D scene

        :assembly_name: Name of the assembly to which the astrocytes should be added
        :population_name: Name of the population of astrocytes
        :vasculature_population_name: Name of the vasculature population. Automatically load
                                      end-feet if not empty
        :load_somas: Load somas if set to true
        :load_dendrites: Load dendrites if set to true
        :generate_internals: Generate internals (Nucleus and mitochondria)
        :use_sdf: Use sign distance fields geometry to create the astrocytes. Defaults to False
        :morphology_representation: Geometry representation (graph, section or segment)
        :morphology_color_scheme: Color scheme of the sections of the astrocytes
        :populationColorScheme: Color scheme of the population of astrocytes
        :radius_muliplier: Applies the multiplier to all radii of the astrocyte sections
        :sql_filter: Condition added to the SQL statement loading the astrocytes
        :scale: Scale in the 3D scene
        :animation_params: Extra optional parameters for animation purposes

        :return: Result of the request submission
        """
        assert isinstance(scale, Vector3)
        assert isinstance(animation_params, AnimationParams)

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["vasculaturePopulationName"] = vasculature_population_name
        params["loadSomas"] = load_somas
        params["loadDendrites"] = load_dendrites
        params["generateInternals"] = generate_internals
        params["useSdf"] = use_sdf
        params["morphologyRepresentation"] = morphology_representation
        params["morphologyColorScheme"] = morphology_color_scheme
        params["populationColorScheme"] = population_color_scheme
        params["radiusMultiplier"] = radius_multiplier
        params["sqlFilter"] = sql_filter
        params["scale"] = scale.to_list()
        params["animationParams"] = animation_params.to_list()
        return self._invoke_and_check('add-astrocytes', params)

    def add_neurons(
            self, assembly_name, population_name,
            use_sdf=False,
            load_somas=True, load_axon=True,
            load_basal_dendrites=True, load_apical_dendrites=True,
            load_synapses=False,
            generate_internals=False, generate_externals=False,
            generate_varicosities=False, show_membrane=True,
            morphology_representation=MORPHOLOGY_REPRESENTATION_SEGMENT,
            morphology_color_scheme=MORPHOLOGY_COLOR_SCHEME_NONE,
            population_color_scheme=POPULATION_COLOR_SCHEME_NONE,
            radius_multiplier=1.0, simulation_report_id=-1,
            sql_node_filter='', sql_section_filter='',
            scale=Vector3(1.0, 1.0, 1.0), animation_params=AnimationParams()):
        """
        Add a population of astrocytes to the 3D scene

        :assembly_name: Name of the assembly to which the astrocytes should be added
        :population_name: Name of the population of astrocytes
        :use_sdf: Use sign distance fields geometry to create the astrocytes. Defaults to False
        :load_somas: Load somas if set to true
        :load_axon: Load axon if set to true
        :load_basal_dendrites: Load basal dendrites if set to true
        :load_apical_dendrites: Load apical dendrites if set to true
        :load_synapses: Load synapses if set to true
        :generate_internals: Generate internals (Nucleus and mitochondria)
        :generate_externals: Generate externals (Myelin steath)
        :show_membrane: Show membrane (Typically used to isolate internal and external components
        :generate_varicosities: Generate random varicosities along the axon
        :morphology_representation: Geometry representation (graph, section or segment)
        :morphology_color_scheme: Color scheme of the sections of the astrocytes
        :populationColorScheme: Color scheme of the population of astrocytes
        :radius_muliplier: Applies the multiplier to all radii of the astrocyte sections
        :simulation_report_id: Identifier of the simulation report (Optional)
        :sql_node_filter: Condition added to the SQL statement loading the nodes
        :sql_section_filter: Condition added to the SQL statement loading the sections
        :scale: Scale in the 3D scene
        :animation_params: Extra optional parameters for animation purposes

        :return: Result of the request submission
        """
        assert isinstance(scale, Vector3)
        assert isinstance(animation_params, AnimationParams)

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["loadSomas"] = load_somas
        params["loadAxon"] = load_axon
        params["loadBasalDendrites"] = load_basal_dendrites
        params["loadApicalDendrites"] = load_apical_dendrites
        params["loadSynapses"] = load_synapses
        params["generateInternals"] = generate_internals
        params["generateExternals"] = generate_externals
        params["showMembrane"] = show_membrane
        params["generateVaricosities"] = generate_varicosities
        params["useSdf"] = use_sdf
        params["morphologyRepresentation"] = morphology_representation
        params["morphologyColorScheme"] = morphology_color_scheme
        params["populationColorScheme"] = population_color_scheme
        params["radiusMultiplier"] = radius_multiplier
        params["simulationReportId"] = simulation_report_id
        params["sqlNodeFilter"] = sql_node_filter
        params["sqlSectionFilter"] = sql_section_filter
        params["scale"] = scale.to_list()
        params["animationParams"] = animation_params.to_list()
        return self._invoke_and_check('add-neurons', params)

    def get_neuron_section_points(self, assembly_name, neuron_guid, section_guid):
        """
        Return the list of 3D points for a given section of a given neuron

        :assembly_name: Name of the assembly
        :neuron_guid: Neuron identifier
        :section_guid: Section identifier

        :return: A list of 3D points
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["neuronId"] = neuron_guid
        params["sectionId"] = section_guid
        response = self._invoke('get-neuron-section-points', params)
        if not response['status']:
            raise "Failed to get neuron section points"
        points = response['points']
        ps = list()
        for i in range(0, len(points), 4):
            p = list()
            for j in range(4):
                p.append(points[i+j])
            ps.append(p)
        return ps

    def get_neuron_varicosities(self, assembly_name, neuron_guid):
        """
        Return the list of 3D locations for the varicosities of a given neuron

        :assembly_name: Name of the assembly
        :neuron_guid: Neuron identifier

        :return: A list of 3D points
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["neuronId"] = neuron_guid
        response = self._invoke('get-neuron-varicosities', params)
        if not response['status']:
            raise "Failed to get neuron varicosities"
        points = response['points']
        ps = list()
        for i in range(0, len(points), 3):
            p = list()
            for j in range(3):
                p.append(points[i+j])
            ps.append(p)
        return ps

    def add_white_matter(
            self, assembly_name, population_name, radius=1.0,
            sql_filter='', scale=Vector3(1.0, 1.0, 1.0)):
        """
        Add white matter to the 3D scene

        :assembly_name: Name of the assembly to which the vasculature should be added
        :population_name Name of the node population
        :radius: Applies the radius to the white matter streamlines
        :sql_filter: Condition added to the SQL statement loading the white matter streamlines
        :scale: Scale in the 3D scene

        :return: Result of the request submission
        """
        assert isinstance(scale, Vector3)

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["radius"] = radius
        params["sqlFilter"] = sql_filter
        params["scale"] = scale.to_list()
        return self._invoke_and_check('add-white-matter', params)

    def look_at(self, source, target):
        """
        Computes quaternion from given direction
        
        Generate a quaternion to make a object rotate in the direction of a vector
        defined by two 3D points, a source and a target

        :source: Source 3d point
        :target: Target 3d point

        :return: The resulting rotation
        """
        assert isinstance(source, Vector3)
        assert isinstance(target, Vector3)

        params = dict()
        params["source"] = source.to_list()
        params["target"] = target.to_list()
        response = self._invoke('look-at', params)
        q = response['rotation']
        return Quaternion(q[3], q[0], q[1], q[2])

# Private classes


class AssemblyProtein:
    """An AssemblyProtein is a Protein that belongs to an assembly"""

    def __init__(self, assembly_name, name, source,
                 atom_radius_multiplier=1.0,
                 load_bonds=True, representation=BioExplorer.REPRESENTATION_ATOMS,
                 load_non_polymer_chemicals=False, load_hydrogen=True, chain_ids=list(),
                 recenter=True, occurrences=1, transmembrane_params=Vector2(),
                 position=Vector3(), rotation=Quaternion(), allowed_occurrences=list(),
                 animation_params=AnimationParams(), constraints=list()):
        """
        An AssemblyProtein is a protein that belongs to an assembly

        :assembly_name: Name of the assembly
        :name: Name of the protein
        :contents: PDB representation of the protein
        :atom_radius_multiplier: Multiplier applied to atom radius
        :load_bonds: Loads bonds if True
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :load_non_polymer_chemicals: Loads non-polymer chemicals if True
        :load_hydrogen: Loads hydrogens if True
        :chain_ids: IDs of the protein chains to be loaded
        :recenter: Centers the protein if True
        :occurrences: Number of occurrences to be added to the assembly
        :animation_params: Seed for position randomization
        :position: Relative position of the protein in the assembly
        :rotation: Relative rotation of the protein in the assembly
        :allowed_occurrences: Indices of protein occurrences in the assembly for
                                    which proteins are added
        :constraints: List of assemblies that constraint the placememnt of the proteins
        """
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(animation_params, AnimationParams)

        self.assembly_name = assembly_name
        self.name = name
        self.pdb_id = os.path.splitext(os.path.basename(source))[0].lower()
        self.source = source
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.load_hydrogen = load_hydrogen
        self.representation = representation
        self.transmembrane_params = transmembrane_params.copy()
        self.chain_ids = chain_ids.copy()
        self.recenter = recenter
        self.occurrences = occurrences
        self.allowed_occurrences = allowed_occurrences.copy()
        self.animation_params = animation_params.copy()
        self.position = position.copy()
        self.rotation = rotation
        self.constraints = constraints


# Public classes


class Membrane:
    """A membrane is a shaped assembly of phospholipids"""

    def __init__(self, lipid_sources, lipid_rotation=Quaternion(), lipid_density=1.0,
                 atom_radius_multiplier=1.0, load_bonds=False,
                 representation=BioExplorer.REPRESENTATION_ATOMS_AND_STICKS,
                 load_non_polymer_chemicals=False, chain_ids=list(), recenter=True,
                 animation_params=AnimationParams()):
        """
        A membrane is an assembly of proteins with a given size and shape

        :size: Size of the cell in the scene (in nanometers)
        :shape: Shape of the membrane (Spherical, planar, sinusoidal, cubic, etc)
        :lipid_sources: Full paths of the PDB files containing the building blocks of the membrane
        :atom_radius_multiplier: Multiplies atom radius by the specified value
        :load_bonds: Defines if bonds should be loaded
        :representation: Representation of the protein (Atoms, atoms and sticks, etc)
        :load_non_polymer_chemicals: Defines if non-polymer chemical should be loaded
        :chain_ids: IDs of the protein chains to be loaded
        :recenter: Defines if proteins should be centered when loaded from PDB files
        :position: Relative position of the membrane in the assembly
        :rotation: Relative rotation of the membrane in the assembly
        """
        assert isinstance(lipid_rotation, Quaternion)
        assert isinstance(animation_params, AnimationParams)
        assert lipid_density > 0.0
        assert lipid_density <= 10.0

        self.lipid_sources = lipid_sources
        self.lipid_rotation = lipid_rotation
        self.lipid_density = lipid_density
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.representation = representation
        self.chain_ids = chain_ids.copy()
        self.recenter = recenter
        self.animation_params = animation_params.copy()


class Sugar:
    """Sugars are glycan trees that can be added to the glycosylation sites of a given protein"""

    def __init__(self, assembly_name, name, source, protein_name, atom_radius_multiplier=1.0,
                 load_bonds=True, representation=BioExplorer.REPRESENTATION_ATOMS,
                 recenter=True, chain_ids=list(), site_indices=list(), rotation=Quaternion(),
                 animation_params=AnimationParams()):
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
        :shape_params: Extra optional parameters for positioning on the protein
        """
        assert isinstance(chain_ids, list)
        assert isinstance(site_indices, list)
        assert isinstance(rotation, Quaternion)
        assert isinstance(animation_params, AnimationParams)

        self.assembly_name = assembly_name
        self.name = name
        self.pdb_id = os.path.splitext(os.path.basename(source))[0].lower()
        self.contents = "".join(open(source).readlines())
        self.protein_name = protein_name
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.representation = representation
        self.recenter = recenter
        self.chain_ids = chain_ids.copy()
        self.site_indices = site_indices.copy()
        self.rotation = rotation
        self.animation_params = animation_params.copy()


class RNASequence:
    """An RNASequence is an assembly of a given shape holding a given genetic code"""

    def __init__(self, source, shape, shape_params, protein_source='', values_range=Vector2(),
                 curve_params=Vector3(), position=Vector3(), rotation=Quaternion(),
                 atom_radius_multiplier=1.0, representation=BioExplorer.REPRESENTATION_ATOMS,
                 animation_params=AnimationParams()):
        """
        RNA sequence descriptor

        :source: Full path of the file containing the RNA sequence
        :lipid_sources: Full path of the file containing the PDB representation of the N protein
        :shape: Shape of the sequence (Trefoil knot, torus, star, spring, heart, Moebiusknot, etc)
        :shape_params: Assembly parameters (radius, etc.)
        :t_range: Range of values used to enroll the RNA thread
        :shape_params: Shape parameters
        :position: Relative position of the RNA sequence in the assembly
        :rotation: Relative position of the RNA sequence in the assembly
        """
        assert isinstance(values_range, Vector2)
        assert isinstance(shape_params, Vector2)
        assert isinstance(curve_params, Vector3)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(animation_params, AnimationParams)

        self.source = source
        self.protein_source = protein_source
        self.shape = shape
        self.shape_params = shape_params.copy()
        self.values_range = values_range.copy()
        self.curve_params = curve_params.copy()
        self.atom_radius_multiplier = atom_radius_multiplier
        self.representation = representation
        self.animation_params = animation_params.copy()
        self.position = position.copy()
        self.rotation = rotation


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
    """A Cell is a membrane with transmembrane proteins"""

    def __init__(self, name, shape, shape_params, membrane, proteins=list(), shape_mesh_source=''):
        """
        Cell descriptor

        :name: Name of the cell in the scene
        :membrane: Membrane descriptor
        :proteins: List of trans-membrane proteins
        """
        assert isinstance(shape_params, Vector3)
        assert isinstance(membrane, Membrane)
        assert isinstance(proteins, list)
        self.name = name
        self.shape = shape
        self.shape_params = shape_params.copy()
        self.shape_mesh_source = shape_mesh_source
        self.membrane = membrane
        self.proteins = proteins.copy()


class Volume:
    """A volume define a 3D space in which proteins can be added"""

    def __init__(self, name, shape, shape_params, protein):
        """
        Volume description

        :name: Name of the volume in the scene
        :shape_params: Size of the volume in the scene (in nanometers)
        :protein: Protein descriptor
        """
        assert isinstance(shape_params, Vector3)
        assert isinstance(protein, Protein)

        self.name = name
        self.shape = shape
        self.shape_params = shape_params.copy()
        self.protein = protein


class Protein:
    """A Protein holds the 3D structure of a protein as well as it Amino Acid sequences"""

    def __init__(self, name, source, occurrences=1, load_bonds=False,
                 load_hydrogen=False, load_non_polymer_chemicals=False, position=Vector3(),
                 rotation=Quaternion(), allowed_occurrences=list(), chain_ids=list(),
                 transmembrane_params=Vector2(), animation_params=AnimationParams()):
        """
        Protein descriptor

        :source: Full path to the protein PDB file
        :occurrences: Number of occurrences to be added to the assembly
        :load_bonds: Loads bonds if True
        :load_hydrogen: Loads hydrogens if True
        :load_non_polymer_chemicals: Loads non-polymer chemicals if True
        :position: Position of the mesh in the scene
        :rotation: Rotation of the mesh in the scene
        :allowed_occurrences: Specific occurrences for which an instance is added to the assembly
        """
        assert isinstance(occurrences, int)
        assert isinstance(position, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(transmembrane_params, Vector2)
        assert isinstance(allowed_occurrences, list)
        assert isinstance(animation_params, AnimationParams)

        self.name = name
        self.pdb_id = os.path.splitext(os.path.basename(source))[0].lower()
        self.source = source
        self.occurrences = occurrences
        self.load_bonds = load_bonds
        self.load_hydrogen = load_hydrogen
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.position = position.copy()
        self.rotation = rotation
        self.allowed_occurrences = allowed_occurrences.copy()
        self.chain_ids = chain_ids.copy()
        self.transmembrane_params = transmembrane_params.copy()
        self.animation_params = animation_params.copy()


class Virus:
    """A Virus is an assembly of proteins (S, M and E), a membrane, and an RNA sequence"""

    def __init__(self, name, shape_params, protein_s=None, protein_e=None, protein_m=None,
                 membrane=None, rna_sequence=None):
        """
        Virus descriptor

        :name: Name of the virus in the scene
        :shape_params: Assembly parameters (Virus radius and maximum range for random
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
            assert isinstance(membrane, Membrane)
        if rna_sequence is not None:
            assert isinstance(rna_sequence, RNASequence)
        assert isinstance(shape_params, list)
        self.name = name
        self.protein_s = protein_s
        self.protein_e = protein_e
        self.protein_m = protein_m
        self.membrane = membrane
        self.rna_sequence = rna_sequence
        self.shape_params = shape_params


class EnzymeReaction:
    """
    Enzyme reaction descriptor

    Enzymes are catalysts and increase the speed of a chemical reaction without themselves
    undergoing any permanent chemical change. They are neither used up in the reaction nor do
    they appear as reaction products.

    The basic enzymatic reaction can be represented as follows:
    - E represents the enzyme catalyzing the reaction,
    - S the substrate, the substance being changed
    - P the product of the reaction

    S + E -> P + E
    """

    def __init__(self, assembly_name, name, enzyme, substrates, products):
        """
        Enzyme reaction descriptor

        :name: Name of the reaction in the scene
        :enzyme: The enzyme catalyzing the reaction
        :substrates: List of substrates by name
        :products: List of products by name
        """
        assert isinstance(enzyme, Protein)
        assert isinstance(substrates, list)
        assert isinstance(products, list)
        self.assembly_name = assembly_name
        self.name = name
        self.enzyme = enzyme
        self.substrates = substrates
        self.products = products
