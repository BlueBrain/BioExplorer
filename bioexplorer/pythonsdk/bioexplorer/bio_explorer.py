#!/usr/bin/env python3

# Copyright 2020 - 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module bio_explorer

This module provides SDK for the BioExplorer plug-in.
"""

import math
import os
from pyquaternion import Quaternion
from typing import Dict, Any, List, Optional
import seaborn as sns

from .version import VERSION as __version__
from .core.client import Client
from .transfer_function import *
from .math_utils import *
from .report_parameters import *
from .displacement_parameters import *
from .animation_parameters import *
from .molecular_systems import *
from .enums import *


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

class BioExplorer:
    """Blue Brain BioExplorer"""

    PLUGIN_API_PREFIX = "be-"
    CONTENTS_DELIMITER = "||||"

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

    NB_MATERIALS_PER_MORPHOLOGY = 10

    model_loading_transaction = TransactionType

    shading_mode = ShadingMode
    shading_chameleon_mode = ShadingChameleonMode
    shading_clipping_mode = ShadingClippingMode

    camera_projection = CameraProjection

    rendering_quality = RenderingQuality

    assembly_shape = AssemblyShape

    material_offset = MaterialOffset

    position_constraint = PositionConstraint

    protein_color_scheme = ProteinColorScheme
    protein_representation = ProteinRepresentation
    surfactant_type = SurfactantType
    rna_shape = RNAShape

    vascular_representation = VascularRepresentation
    vascular_color_scheme = VascularColorScheme
    vascular_realism_level = VascularRealismLevel

    morphology_representation = MorphologyRepresentation
    morphology_color_scheme = MorphologyColorScheme
    morphology_realism_level = MorphologyRealismLevel

    population_color_scheme = PopulationColorScheme

    synapse_representation = SynapseRepresentation

    neuron_synapse_type = NeuronSynapseType
    neuron_material = NeuronMaterial

    astrocyte_material = AstrocyteMaterial

    field_data_type = FieldDataType
    file_format = FileFormat

    def __init__(self, url="localhost:5000"):
        """
        Create a new BioExplorer instance.

        :param url: The URL of the BioExplorer backend. Defaults to "localhost:5000".
        :type url: str
        """
        self._url = url
        self._client = Client(url)
        self._v1_compatibility = False

        backend_version = self.version()
        if __version__ != backend_version:
            raise RuntimeError(
                "Wrong version of the back-end ("
                + backend_version
                + "). Use version "
                + __version__
                + " for this version of the BioExplorer python library"
            )

    def __str__(self):
        """
        A pretty-print of the class.

        :returns: The name of the class.
        :rtype: str
        """
        return "Blue Brain BioExplorer"

    def core_api(self):
        """
        Access to the underlying core API (Brayns core API).

        :returns: The client class of the core API.
        :rtype: Client
        """
        return self._client

    def _check(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the response from the BioExplorer backend.

        :param response: The response to check.
        :type response: Dict[str, Any]
        :returns: The validated response.
        :rtype: Dict[str, Any]
        :raises RuntimeError: If the response indicates an error.
        """
        if not response.get("status"):
            raise RuntimeError(f"Error from BioExplorer backend: {response.get('contents')}")
        return response

    def _invoke_and_check(
            self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke a method on the BioExplorer client and check the response.

        :param method: The method to invoke.
        :type method: str
        :param params: Parameters to pass to the method. Defaults to None.
        :type params: Optional[Dict[str, Any]]
        :returns: The response from the BioExplorer backend after validation.
        :rtype: Dict[str, Any]
        :raises RuntimeError: If the response from the backend indicates an error.
        """
        if params is None:
            params = {}

        prefixed_method = self.PLUGIN_API_PREFIX + method
        response = self._client.rockets_client.request(
            method=prefixed_method, params=params
        )

        return self._check(response)

    def _invoke(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke a method on the BioExplorer client.

        :param method: The method to invoke.
        :type method: str
        :param params: Parameters to pass to the method. Defaults to None.
        :type params: Optional[Dict[str, Any]]
        :returns: The response from the BioExplorer backend after validation.
        :rtype: Dict[str, Any]
        """
        prefixed_method = self.PLUGIN_API_PREFIX + method
        return self._client.rockets_client.request(
            method=prefixed_method, params=params
        )

    @staticmethod
    def _enums_to_list(the_enums: List) -> List:
        """
        Convert enum values to a list.

        :param the_enums: A list of enums.
        :type the_enums: list
        :returns: A list of enum values.
        :rtype: list
        """
        the_list = list()
        for item in the_enums:
            the_list.append(item.value)
        return the_list

    def version(self) -> str:
        """
        Get the version of the BioExplorer application.

        :returns: The version of the BioExplorer application.
        :rtype: str
        """
        if self._client is None:
            return __version__

        result = self._invoke_and_check("get-version")
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result["contents"]

    def scene_information(self) -> 'Metrics':
        """
        Get metrics about the scene handled by the BioExplorer backend.

        :returns: Metrics about the current scene.
        :rtype: Metrics
        """
        if self._client is None:
            return __version__

        return self._invoke_and_check("get-scene-information")

    @staticmethod
    def authors() -> List[str]:
        """
        Get the list of authors.

        :returns: A list of author names.
        :rtype: List[str]
        """
        return [
            "Cyrille Favreau",
            "Daniel Nachbaur",
            "Jonas Karlsson",
            "Andrei-Roland Groza",
            "Grigori Chevtchenko",
            "Raphael Dumusc",
            "Ahmet Bilgili",
            "Juan Bautista Hernando Vieites",
            "Pawel Jozef Podhajski",
            "Jafet Villafranca Diaz",
        ]

    def reset_scene(self):
        """
        Remove all assemblies

        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        return self._invoke_and_check("reset-scene")

    def reset_camera(self) -> Dict[str, Any]:
        """
        Reset the camera for the current scene

        :return: Result of the call to the BioExplorer backend
        :rtype: Response
        """
        return self._invoke_and_check("reset-camera")

    def set_focus_on(
        self,
        model_id: int,
        instance_id: int = 0,
        distance: float = 0.0,
        direction: Vector3 = None,
        max_number_of_instances: float = 1e6,
    ) -> Dict[str, Any]:
        """
        Set focus on a specific instance of a model.

        :param model_id: The ID of the model to focus on.
        :type model_id: int
        :param instance_id: The instance ID of the model. Defaults to 0.
        :type instance_id: int, optional
        :param distance: The distance for the focus setting. Defaults to 0.0.
        :type distance: float, optional
        :param direction: The direction vector for focus. Defaults to Vector3(0.0, 0.0, 1.0).
        :type direction: Vector3, optional
        :param max_number_of_instances: Maximum number of instances to consider. Defaults to 1e6.
        :type max_number_of_instances: float, optional

        :returns: Result of the backend API call.
        :rtype: Dict[str, Any]

        :raises ValueError: If the direction is not provided or not an instance of Vector3.
        :raises RuntimeError: If the backend call fails.
        """
        if direction is None:
            direction = Vector3(0.0, 0.0, 1.0)
        if not isinstance(direction, Vector3):
            raise ValueError("Direction must be an instance of Vector3")

        params = {
            "modelId": model_id,
            "instanceId": instance_id,
            "distance": distance,
            "direction": direction.to_list(),
            "maxNbInstances": max_number_of_instances
        }

        result = self._invoke_and_check("set-focus-on", params)
        if not result["status"]:
            raise RuntimeError(f"Failed to set focus: {result['contents']}")
        return result

    def export_to_file(
        self,
        filename: str,
        low_bounds: Vector3 = Vector3(-1e38, -1e38, -1e38),
        high_bounds: Vector3 = Vector3(1e38, 1e38, 1e38)
    ) -> Dict[str, Any]:
        """
        Export the current scene to a file.

        :param filename: The name of the file to export to.
        :type filename: str
        :param low_bounds: The lower bounds of the export region. Defaults to
        Vector3(-1e38, -1e38, -1e38).
        :type low_bounds: Vector3, optional
        :param high_bounds: The upper bounds of the export region. Defaults to
        Vector3(1e38, 1e38, 1e38).
        :type high_bounds: Vector3, optional

        :returns: Result of the backend API call.
        :rtype: Dict[str, Any]

        :raises RuntimeError: If the backend call fails.
        """
        params = {
            "filename": filename,
            "lowBounds": low_bounds.to_list(),
            "highBounds": high_bounds.to_list(),
            "fileFormat": FileFormat.UNSPECIFIED
        }
        result = self._invoke_and_check("export-to-file", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def export_to_database(
        self,
        connection_string: str,
        schema: str,
        brick_id: int,
        low_bounds: Vector3 = None,
        high_bounds: Vector3 = None
    ) -> Dict[str, Any]:
        """
        Export the current scene to a database.

        :param connection_string: The connection string to the database.
        :type connection_string: str
        :param schema: The schema to use in the database.
        :type schema: str
        :param brick_id: The ID of the brick to export.
        :type brick_id: int
        :param low_bounds: The lower bounds of the export region. Defaults to
        Vector3(-1e38, -1e38, -1e38).
        :type low_bounds: Vector3, optional
        :param high_bounds: The upper bounds of the export region. Defaults to
        Vector3(1e38, 1e38, 1e38).
        :type high_bounds: Vector3, optional

        :returns: Result of the backend API call.
        :rtype: Dict[str, Any]

        :raises RuntimeError: If the backend call fails.
        """
        assert isinstance(low_bounds or Vector3(-1e38, -1e38, -1e38), Vector3)
        assert isinstance(high_bounds or Vector3(1e38, 1e38, 1e38), Vector3)
        params = {
            "connectionString": connection_string,
            "schema": schema,
            "brickId": brick_id,
            "lowBounds": (low_bounds or Vector3(-1e38, -1e38, -1e38)).to_list(),
            "highBounds": (high_bounds or Vector3(1e38, 1e38, 1e38)).to_list()
        }
        result = self._invoke_and_check("export-to-database", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def import_from_cache(self, filename: str) -> Dict[str, Any]:
        """
        Import a scene from a cache file.

        :param filename: The name of the file to import from.
        :type filename: str

        :returns: Result of the backend API call.
        :rtype: Dict[str, Any]

        :raises RuntimeError: If the backend call fails.
        """
        params = {
            "filename": filename,
            "fileFormat": FileFormat.UNSPECIFIED
        }
        result = self._invoke_and_check("import-from-cache", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def export_to_xyz(
        self,
        filename: str,
        file_format: FileFormat,
        low_bounds: Vector3 = Vector3(-1e38, -1e38, -1e38),
        high_bounds: Vector3 = Vector3(1e38, 1e38, 1e38),
    ) -> Dict[str, Any]:
        """
        Export the current scene to an XYZ file.

        :param filename: The name of the file to export to.
        :type filename: str
        :param file_format: The format of the file to export.
        :type file_format: FileFormat
        :param low_bounds: The lower bounds of the export region. Defaults to
        Vector3(-1e38, -1e38, -1e38).
        :type low_bounds: Vector3, optional
        :param high_bounds: The upper bounds of the export region. Defaults to
        Vector3(1e38, 1e38, 1e38).
        :type high_bounds: Vector3, optional

        :returns: Result of the backend API call.
        :rtype: Dict[str, Any]

        :raises RuntimeError: If the backend call fails.
        """
        params = {
            "filename": filename,
            "lowBounds": low_bounds.to_list(),
            "highBounds": high_bounds.to_list(),
            "fileFormat": file_format.value
        }
        result = self._invoke_and_check("export-to-xyz", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def export_to_las(
        self,
        filename: str,
        model_ids: list = list(),
        material_ids: list = list(),
        export_colors: bool = False
    ) -> Dict[str, Any]:
        """
        Export the current scene to a LAS file.

        :param filename: The name of the file to export to.
        :type filename: str
        :param file_format: The format of the file to export.
        :type export_colors: Export sphere RGB information

        :returns: Result of the backend API call.
        :rtype: Dict[str, Any]

        :raises RuntimeError: If the backend call fails.
        """
        params = {
            "filename": filename,
            "modelIds": model_ids,
            "materialIds": material_ids,
            "exportColors": export_colors
        }
        result = self._invoke_and_check("export-to-las", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def add_sars_cov_2(
        self,
        name,
        resource_folder,
        shape_params=Vector3(45.0, 0.0, 0.0),
        animation_params=MolecularSystemAnimationParams(0, 1, 0.25, 1, 0.1),
        nb_protein_s=62,
        nb_protein_m=50,
        nb_protein_e=42,
        open_protein_s_indices=[0],
        atom_radius_multiplier=1.0,
        add_glycans=False,
        add_rna_sequence=False,
        representation=ProteinRepresentation.ATOMS_AND_STICKS,
        clipping_planes=list(),
        position=Vector3(),
        rotation=Quaternion(),
        apply_colors=False,
    ):
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
        membrane_proteins.append(
            Protein(
                name=name + "_" + self.NAME_PROTEIN_S_OPEN,
                source=os.path.join(pdb_folder, "6vyb.pdb"),
                occurrences=nb_protein_s,
                rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
                allowed_occurrences=open_conformation_indices,
                transmembrane_params=Vector2(10.5, 10.5),
                animation_params=ap,
            )
        )

        # Protein S (closed)
        membrane_proteins.append(
            Protein(
                name=name + "_" + self.NAME_PROTEIN_S_CLOSED,
                source=os.path.join(pdb_folder, "sars-cov-2-v1.pdb"),
                occurrences=nb_protein_s,
                rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
                allowed_occurrences=closed_conformation_indices,
                transmembrane_params=Vector2(10.5, 10.5),
                animation_params=ap,
            )
        )

        # Protein M (QHD43419)
        ap.seed = 1
        membrane_proteins.append(
            Protein(
                name=name + "_" + self.NAME_PROTEIN_M,
                source=os.path.join(pdb_folder, "QHD43419a.pdb"),
                occurrences=nb_protein_m,
                position=Vector3(2.5, 0.0, 0.0),
                rotation=Quaternion(0.135, 0.99, 0.0, 0.0),
                transmembrane_params=Vector2(0.5, 2.0),
                animation_params=ap,
            )
        )

        # Protein E (QHD43418 P0DTC4)
        ap.seed = 3
        membrane_proteins.append(
            Protein(
                name=name + "_" + self.NAME_PROTEIN_E,
                source=os.path.join(pdb_folder, "QHD43418a.pdb"),
                occurrences=nb_protein_e,
                position=Vector3(2.5, 0.0, 0.0),
                rotation=Quaternion(0.0, 0.707, 0.707, 0.0),
                transmembrane_params=Vector2(0.5, 2.0),
                animation_params=ap,
            )
        )

        # Virus membrane
        ap.seed = 4
        lipid_sources = [
            os.path.join(membrane_folder, "segA.pdb"),
            os.path.join(membrane_folder, "segB.pdb"),
            os.path.join(membrane_folder, "segC.pdb"),
            os.path.join(membrane_folder, "segD.pdb"),
        ]

        virus_membrane = Membrane(
            lipid_sources=lipid_sources,
            lipid_rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            load_bonds=True,
            load_non_polymer_chemicals=True,
            animation_params=ap,
        )

        # Cell
        virus_cell = Cell(
            name=name,
            shape=AssemblyShape.EMPTY_SPHERE,
            shape_params=shape_params,
            membrane=virus_membrane,
            proteins=membrane_proteins,
        )

        self.add_cell(
            cell=virus_cell,
            atom_radius_multiplier=atom_radius_multiplier,
            representation=representation,
            position=position,
            rotation=rotation,
            clipping_planes=clipping_planes,
        )

        # RNA Sequence
        if add_rna_sequence:
            params = Vector2(shape_params.x * 0.55, 0.5)
            rna_sequence = RNASequence(
                source=os.path.join(rna_folder, "sars-cov-2.rna"),
                protein_source=os.path.join(pdb_folder, "7bv1.pdb"),
                shape=self.rna_shape.TREFOIL_KNOT,
                shape_params=params,
                values_range=Vector2(-8.0 * math.pi, 8.0 * math.pi),
                curve_params=Vector3(1.51, 1.12, 1.93),
                atom_radius_multiplier=atom_radius_multiplier,
                representation=representation,
                animation_params=ap,
            )
            self.add_rna_sequence(
                assembly_name=name,
                name=name + "_" + BioExplorer.NAME_RNA_SEQUENCE,
                rna_sequence=rna_sequence,
            )

        if add_glycans:
            glycan_representation = representation
            if representation == ProteinRepresentation.MESH:
                glycan_representation = ProteinRepresentation.ATOMS_AND_STICKS

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
                animation_params=MolecularSystemAnimationParams(
                    0,
                    0,
                    0.0,
                    animation_params.get_params('rotation_seed') + 7,
                    animation_params.get_params('rotation_strength'),
                ),
            )
            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=high_mannose_paths,
                indices=indices_open,
                representation=glycan_representation,
                atom_radius_multiplier=atom_radius_multiplier,
                animation_params=MolecularSystemAnimationParams(
                    0,
                    0,
                    0.0,
                    animation_params.get_params('rotation_seed') + 7,
                    animation_params.get_params('rotation_strength'),
                ),
            )

            # Complex
            indices_closed = [
                17, 74, 149, 165, 282, 331, 343, 616, 657, 1098, 1134, 1158, 1173, 1194]
            indices_open = [
                17, 74, 149, 165, 282, 331, 343, 657, 1098, 1134, 1158, 1173, 1194]
            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_COMPLEX,
                protein_name=self.NAME_PROTEIN_S_CLOSED,
                paths=complex_paths,
                indices=indices_closed,
                representation=glycan_representation,
                atom_radius_multiplier=atom_radius_multiplier,
                animation_params=MolecularSystemAnimationParams(
                    0,
                    0,
                    0.0,
                    animation_params.get_params('rotation_seed') + 8,
                    2.0 * animation_params.get_params('rotation_strength'),
                ),
            )

            self.add_multiple_glycans(
                assembly_name=name,
                glycan_type=self.NAME_GLYCAN_COMPLEX,
                protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=complex_paths,
                indices=indices_open,
                representation=glycan_representation,
                atom_radius_multiplier=atom_radius_multiplier,
                animation_params=MolecularSystemAnimationParams(
                    0,
                    0,
                    0.0,
                    animation_params.get_params('rotation_seed') + 8,
                    2.0 * animation_params.get_params('rotation_strength'),
                ),
            )

            # O-Glycans
            for index in [323, 325]:
                o_glycan_name = (
                    name + "_" + self.NAME_GLYCAN_O_GLYCAN + "_" + str(index)
                )
                o_glycan = Sugar(
                    assembly_name=name,
                    name=o_glycan_name,
                    source=o_glycan_paths[0],
                    protein_name=name + "_" + self.NAME_PROTEIN_S_CLOSED,
                    site_indices=[index],
                    representation=glycan_representation,
                    atom_radius_multiplier=atom_radius_multiplier,
                    animation_params=MolecularSystemAnimationParams(
                        0,
                        0,
                        0.0,
                        animation_params.get_params('rotation_seed') + 9,
                        2.0 * animation_params.get_params('rotation_strength'),
                    ),
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
                animation_params=MolecularSystemAnimationParams(
                    0,
                    0,
                    0.0,
                    animation_params.get_params('rotation_seed') + 10,
                    2.0 * animation_params.get_params('rotation_strength'),
                ),
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
                animation_params=MolecularSystemAnimationParams(
                    0,
                    0,
                    0.0,
                    animation_params.get_params('rotation_seed') + 11,
                    2.0 * animation_params.get_params('rotation_strength'),
                ),
            )
            self.add_glycan(complex_glycans)

        if apply_colors:
            # Apply default materials
            self.apply_default_color_scheme(shading_mode=self.BASIC)

    def add_cell(
        self,
        cell,
        atom_radius_multiplier=1.0,
        representation=ProteinRepresentation.ATOMS,
        clipping_planes=list(),
        position=Vector3(),
        rotation=Quaternion(),
    ):
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
            clipping_planes=clipping_planes,
        )

        for protein in cell.proteins:
            if protein.occurrences != 0:
                _protein = AssemblyProtein(
                    assembly_name=cell.name,
                    name=protein.name,
                    source=protein.source,
                    load_non_polymer_chemicals=protein.load_non_polymer_chemicals,
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
            membrane=cell.membrane,
        )

    def add_volume(
        self,
        volume,
        atom_radius_multiplier=1.0,
        representation=ProteinRepresentation.ATOMS_AND_STICKS,
        position=Vector3(),
        rotation=Quaternion(),
        constraints=list(),
    ) -> Dict[str, Any]:
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
        if constraints is None:
            constraints = []

        if not isinstance(volume, Volume):
            raise TypeError("volume must be an instance of Volume")
        if not isinstance(position, Vector3):
            raise TypeError("position must be an instance of Vector3")
        if not isinstance(constraints, list):
            raise TypeError("constraints must be a list")

        _protein = AssemblyProtein(
            assembly_name=volume.name,
            name=volume.protein.name,
            source=volume.protein.source,
            load_non_polymer_chemicals=volume.protein.load_non_polymer_chemicals,
            occurrences=volume.protein.occurrences,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=volume.protein.load_bonds,
            load_hydrogen=volume.protein.load_hydrogen,
            representation=representation,
            animation_params=volume.protein.animation_params,
            position=volume.protein.position,
            rotation=volume.protein.rotation,
            constraints=constraints,
        )

        self.remove_assembly(volume.name)

        result = self.add_assembly(
            name=volume.name,
            shape=volume.shape,
            shape_params=volume.shape_params,
            position=position,
            rotation=rotation,
        )
        if not result["status"]:
            raise RuntimeError(result["contents"])

        result = self.add_assembly_protein(_protein)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def add_surfactant(
        self,
        surfactant,
        atom_radius_multiplier=1.0,
        representation=ProteinRepresentation.ATOMS,
        position=Vector3(),
        rotation=Quaternion(),
        animation_params=MolecularSystemAnimationParams(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)

        shape = AssemblyShape.EMPTY_SPHERE
        nb_branches = 1
        if surfactant.surfactant_protein == SurfactantType.PROTEIN_A:
            shape = AssemblyShape.FAN
            nb_branches = 6
        elif surfactant.surfactant_protein == SurfactantType.PROTEIN_D:
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
            position=p,
            rotation=r,
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
                    position=p,
                    atom_radius_multiplier=atom_radius_multiplier,
                    source=surfactant.branch_source,
                    occurrences=nb_branches,
                    animation_params=animation_params,
                    representation=representation,
                )
            )

        self.remove_assembly(surfactant.name)
        result = self.add_assembly(
            name=surfactant.name,
            shape=shape,
            shape_params=Vector3(),
            position=position,
            rotation=rotation,
        )
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
        result = self.add_assembly(
            name=enzyme_reaction.assembly_name,
            shape=AssemblyShape.POINT,
            shape_params=Vector3(),
        )
        if not result["status"]:
            raise RuntimeError(result["contents"])

        substrate_names = ""
        for substrate in enzyme_reaction.substrates:
            assert isinstance(substrate, Protein)
            if substrate_names != "":
                substrate_names += BioExplorer.CONTENTS_DELIMITER
            substrate_names += substrate.name

        product_names = ""
        for product in enzyme_reaction.products:
            assert isinstance(product, Protein)
            if product_names != "":
                product_names += BioExplorer.CONTENTS_DELIMITER
            product_names += product.name

        params = dict()
        params["assemblyName"] = enzyme_reaction.assembly_name
        params["name"] = enzyme_reaction.name
        params["enzymeName"] = enzyme_reaction.enzyme.name
        params["substrateNames"] = substrate_names
        params["productNames"] = product_names
        return self._invoke_and_check("add-enzyme-reaction", params)

    def set_enzyme_reaction_progress(
        self, enzyme_reaction, instance_id=0, progress=0.0
    ):
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
                [32341, 39916800],
                [89063, 1814400],
                [937889, 725760],
                [234527, 12096],
                [221003071, 1209600],
                [97061099, 86400],
                [3289126079, 725760],
                [106494629, 9072],
                [16747876289, 907200],
                [8214593, 525],
                [73270357, 13860],
                [97, 1],
            ],
            [
                [2039, 13305600],
                [481, 43200],
                [251567, 725760],
                [6809, 1120],
                [80200787, 1209600],
                [2243617, 4800],
                [172704437, 80640],
                [13473253, 2160],
                [9901558223, 907200],
                [16059286, 1575],
                [10483043, 2772],
                [108, 1],
            ],
            [
                [13, 86400],
                [661, 72576],
                [269, 1120],
                [437929, 120960],
                [6933401, 201600],
                [3696647, 17280],
                [7570727, 8640],
                [839289373, 362880],
                [20844207, 5600],
                [32765923, 10080],
                [64229, 56],
                [100, 0],
            ],
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

        result = ""
        for x in range(33):
            result += chr(mol(x % 12, int(x / 12)))
        return "You asked for the meaning of life and the answer is: " + result

    def add_assembly(
        self,
        name,
        shape=AssemblyShape.POINT,
        shape_params=Vector3(),
        shape_mesh_source="",
        clipping_planes=list(),
        position=Vector3(),
        rotation=Quaternion(),
    ):
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

        mesh_contents = ""
        if shape_mesh_source:
            mesh_contents = "".join(open(shape_mesh_source).readlines())

        params = dict()
        params["name"] = name
        params["shape"] = shape.value
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
        params["name"] = name
        params["shape"] = AssemblyShape.POINT.value
        params["shapeParams"] = list()
        params["shapeMeshContents"] = ""
        params["position"] = Vector3().to_list()
        params["rotation"] = list(Quaternion())
        params["clippingPlanes"] = list()
        return self._invoke_and_check("remove-assembly", params)

    def set_protein_color_scheme(
        self,
        assembly_name,
        name,
        color_scheme,
        palette_name="",
        palette_size=256,
        palette=list(),
        chain_ids=list(),
    ):
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
        params["colorScheme"] = color_scheme.value
        params["palette"] = local_palette
        params["chainIds"] = chain_ids
        result = self._invoke_and_check("set-protein-color-scheme", params)
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def set_protein_amino_acid_sequence_as_string(
        self, assembly_name, name, amino_acid_sequence
    ):
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
        result = self._invoke_and_check(
            "set-protein-amino-acid-sequence-as-string", params
        )
        if not result["status"]:
            raise RuntimeError(result["contents"])
        return result

    def set_protein_amino_acid_sequence_as_ranges(
        self, assembly_name, name, amino_acid_ranges
    ):
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
        result = self._invoke_and_check(
            "set-protein-amino-acid-sequence-as-ranges", params
        )
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

    def set_protein_amino_acid(
        self, assembly_name, name, index, amino_acid_short_name, chain_ids=list()
    ):
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

    def set_protein_instance_transformation(
        self,
        assembly_name,
        name,
        instance_index,
        position=Vector3(),
        rotation=Quaternion(),
    ):
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
            s = param.split("=")
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
            if rna_sequence.shape == self.rna_shape.TORUS:
                values_range = Vector2(0.0, 2.0 * math.pi)
            elif rna_sequence.shape == self.rna_shape.TREFOIL_KNOT:
                values_range = Vector2(0.0, 4.0 * math.pi)
        else:
            values_range = rna_sequence.values_range

        curve_params = [1.0, 1.0, 1.0]
        if rna_sequence.shape_params is None:
            # Defaults
            if rna_sequence.shape == self.rna_shape.TORUS:
                curve_params = Vector3(0.5, 10.0, 0.0)
            elif rna_sequence.shape == self.rna_shape.TREFOIL_KNOT:
                curve_params = Vector3(2.5, 2.0, 2.2)

        else:
            curve_params = rna_sequence.curve_params

        protein_contents = ""
        if rna_sequence.protein_source:
            protein_contents = "".join(open(rna_sequence.protein_source).readlines())

        params = dict()
        params["assemblyName"] = assembly_name
        params["name"] = name
        params["pdbId"] = os.path.splitext(
            os.path.basename(rna_sequence.protein_source)
        )[0].lower()
        params["contents"] = "".join(open(rna_sequence.source).readlines())
        params["proteinContents"] = protein_contents
        params["shape"] = rna_sequence.shape
        params["shapeParams"] = rna_sequence.shape_params.to_list()
        params["valuesRange"] = values_range.to_list()
        params["curveParams"] = curve_params.to_list()
        params["atomRadiusMultiplier"] = rna_sequence.atom_radius_multiplier
        params["representation"] = rna_sequence.representation.value
        params["animationParams"] = rna_sequence.animation_params.to_list()
        params["position"] = rna_sequence.position.to_list()
        params["rotation"] = list(rna_sequence.rotation)
        return self._invoke_and_check("add-rna-sequence", params)

    def add_membrane(
        self,
        assembly_name,
        name,
        membrane,
        animation_params=MolecularSystemAnimationParams(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)

        lipid_pdb_ids = ""
        lipid_contents = ""
        for lipid_source in membrane.lipid_sources:
            if lipid_contents != "":
                lipid_contents += BioExplorer.CONTENTS_DELIMITER
            if lipid_pdb_ids != "":
                lipid_pdb_ids += BioExplorer.CONTENTS_DELIMITER
            lipid_contents += "".join(open(lipid_source).readlines())
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
        params["representation"] = membrane.representation.value
        params["chainIds"] = membrane.chain_ids
        params["recenter"] = membrane.recenter
        params["animationParams"] = animation_params.to_list()
        return self._invoke_and_check("add-membrane", params)

    def add_protein(
        self,
        protein,
        recenter=True,
        representation=ProteinRepresentation.ATOMS_AND_STICKS,
        atom_radius_multiplier=1.0,
        position=Vector3(),
        rotation=Quaternion(),
    ):
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
        self.add_assembly(name=assembly_name, position=position, rotation=rotation)

        _protein = AssemblyProtein(
            assembly_name=assembly_name,
            name=protein.name,
            recenter=recenter,
            source=protein.source,
            load_hydrogen=protein.load_hydrogen,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=protein.load_bonds,
            load_non_polymer_chemicals=protein.load_non_polymer_chemicals,
            representation=representation,
            position=protein.position,
            rotation=protein.rotation,
            chain_ids=protein.chain_ids,
        )
        return self.add_assembly_protein(_protein)

    def add_assembly_protein(self, protein):
        """
        Add a protein to an assembly

        :protein: Description of the protein
        :return: Result of the call to the BioExplorer backend
        """
        assert isinstance(protein, AssemblyProtein)

        constraints = ""
        for constraint in protein.constraints:
            if constraints != "":
                constraints += self.CONTENTS_DELIMITER
            if constraint[0] == PositionConstraint.INSIDE:
                constraints += "+" + constraint[1]
            elif constraint[0] == PositionConstraint.OUTSIDE:
                constraints += "-" + constraint[1]
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
        params["representation"] = protein.representation.value
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
        params["representation"] = glycan.representation.value
        params["recenter"] = glycan.recenter
        params["chainIds"] = glycan.chain_ids
        params["siteIndices"] = glycan.site_indices
        params["rotation"] = list(glycan.rotation)
        params["animationParams"] = glycan.animation_params.to_list()
        return self._invoke_and_check("add-glycan", params)

    def add_multiple_glycans(
        self,
        assembly_name,
        glycan_type,
        protein_name,
        paths,
        representation,
        chain_ids=list(),
        indices=list(),
        load_bonds=True,
        atom_radius_multiplier=1.0,
        rotation=Quaternion(),
        animation_params=MolecularSystemAnimationParams(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)

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
                    name=assembly_name
                    + "_"
                    + protein_name
                    + "_"
                    + glycan_type
                    + "_"
                    + str(path_index),
                    source=source,
                    protein_name=assembly_name + "_" + protein_name,
                    chain_ids=chain_ids,
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=load_bonds,
                    representation=representation,
                    recenter=True,
                    site_indices=site_indices,
                    rotation=rotation,
                    animation_params=animation_params,
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
        params["representation"] = sugar.representation.value
        params["recenter"] = sugar.recenter
        params["chainIds"] = sugar.chain_ids
        params["siteIndices"] = sugar.site_indices
        params["rotation"] = list(sugar.rotation)
        params["animationParams"] = sugar.animation_params.to_list()
        return self._invoke_and_check("add-sugar", params)

    def set_rendering_quality(self, image_quality):
        """
        Set rendering quality using presets

        :image_quality: Quality of the image (LOW or HIGH)
        :return: Result of the call to the BioExplorer backend
        """
        if image_quality == RenderingQuality.HIGH:
            self._client.set_renderer(
                head_light=False,
                background_color=[96 / 255, 125 / 255, 139 / 255],
                current="advanced",
                samples_per_pixel=1,
                subsampling=4,
                max_accum_frames=128,
            )
            params = self._client.AdvancedRendererParams()
            params.main_exposure = 1.0
            params.gi_samples = 1
            params.gi_strength = 0.3
            params.gi_ray_length = 5000
            params.shadow_intensity = 1.0
            params.soft_shadow_strength = 0.1
            params.fog_start = 1200.0
            params.fog_thickness = 300.0
            params.max_ray_depth = 1
            params.use_hardware_randomizer = False
            return self._client.set_renderer_params(params)
        return self._client.set_renderer(
            background_color=Vector3(),
            current="basic",
            samples_per_pixel=1,
            subsampling=4,
            max_accum_frames=16,
        )

    def get_model_name(self, model_id):
        """
        Return the name of a model in the current scene

        :return: Name of a model in the current scene
        """
        assert isinstance(model_id, int)

        params = dict()
        params["modelId"] = model_id
        params["maxNbInstances"] = 0
        return self._invoke("get-model-name", params)

    def get_model_transformation(self, model_id):
        """
        Return the transformation of a model in the current scene

        :return: Transformation of a model in the current scene
        """
        assert isinstance(model_id, int)

        params = dict()
        params["modelId"] = model_id
        params["maxNbInstances"] = 0
        t = self._invoke("get-model-transformation", params)
        tr = t["translation"]
        ro = t["rotation"]
        rc = t["rotationCenter"]
        s = t["scale"]
        return Transformation(
            translation=Vector3(tr[0], tr[1], tr[2]),
            rotation=Quaternion(ro[0], ro[1], ro[2], ro[3]),
            rotation_center=Vector3(rc[0], rc[1], rc[2]),
            scale=Vector3(s[0], s[1], s[2]),
        )

    def get_model_bounds(self, model_id):
        """
        Return the bounds of a model

        :return: Bounds of a model
        """
        assert isinstance(model_id, int)

        params = dict()
        params["modelId"] = model_id
        params["maxNbInstances"] = 0
        model_bounds = self._invoke("get-model-bounds", params)
        value = model_bounds["minAABB"]
        min_aabb = Vector3(value[0], value[1], value[2])
        value = model_bounds["maxAABB"]
        max_aabb = Vector3(value[0], value[1], value[2])
        return Bounds(min_aabb, max_aabb)

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

    def add_model_instance(
        self,
        model_id,
        translation=Vector3(),
        rotation=Quaternion(),
        rotation_center=Vector3(),
        scale=Vector3(1.0, 1.0, 1.0),
    ):
        """
        Return the list of instances in the specified model

        :model_id: Id of the model
        :translation: Instance translation
        :rotation: Instance rotation
        :rotation_center: Instance rotation center
        :scale: Instance scale
        """
        assert isinstance(model_id, int)
        assert isinstance(translation, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(rotation_center, Vector3)
        assert isinstance(scale, Vector3)

        params = dict()
        params["modelId"] = model_id
        params["translation"] = translation.to_list()
        params["rotation"] = list(rotation)
        params["rotationCenter"] = rotation_center.to_list()
        params["scale"] = scale.to_list()
        return self._invoke("add-model-instance", params)

    def set_model_instances(
        self,
        model_id,
        translations=list(),
        rotations=list(),
        rotation_centers=list(),
        scales=list(),
    ):
        """
        Return the list of instances in the specified model

        :model_id: Id of the model
        :translations: List of translations (array of floats x,y,z,x,y,z,...)
        :rotations: List of rotation quaternions (array of floats x,y,z,w,x,y,z,w,...)
        :rotation_centers: List of rotation centers (array of floats x,y,z,x,y,z,...)
        :scales: List of scales (array of floats x,y,z,x,y,z,...)
        """
        assert isinstance(model_id, int)
        assert isinstance(translations, list)
        assert isinstance(rotations, list)
        assert isinstance(rotation_centers, list)
        assert isinstance(scales, list)

        params = dict()
        params["modelId"] = model_id
        params["translations"] = translations
        params["rotations"] = rotations
        params["rotationCenters"] = rotation_centers
        params["scales"] = scales
        return self._invoke("set-model-instances", params)

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

    def set_materials(
        self,
        model_ids,
        material_ids,
        diffuse_colors,
        specular_colors,
        specular_exponents=list(),
        opacities=list(),
        reflection_indices=list(),
        refraction_indices=list(),
        glossinesses=list(),
        shading_modes=list(),
        emissions=list(),
        cast_user_datas=list(),
        user_parameters=list(),
        chameleon_modes=list(),
        clipping_modes=list(),
    ):
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
        :shading_modes: List of shading modes (NONE, BASIC, DIFFUSE, ELECTRON, CARTOON,
        ELECTRON_TRANSPARENCY, PERLIN or DIFFUSE_TRANSPARENCY)
        :emissions: List of light emission intensities
        :user_parameters: List of convenience parameter used by some of the shaders
        :chameleon_modes: List of chameleon mode attributes. If receiver, material take the color of
        surrounding emitter geometry
        :clipping_modes: List of clipping modes applied to the geometry attached to the material
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

        shading_mode_values = self._enums_to_list(shading_modes)
        chameleon_mode_values = self._enums_to_list(chameleon_modes)
        clipping_modes_values = self._enums_to_list(clipping_modes)
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
        params["shadingModes"] = shading_mode_values
        params["userParameters"] = user_parameters
        params["chameleonModes"] = chameleon_mode_values
        params["clippingModes"] = clipping_modes_values
        return self._invoke_and_check("set-materials", params)

    def set_materials_from_palette(
        self,
        model_ids,
        material_ids,
        palette,
        shading_mode,
        specular_exponent,
        user_parameter=1.0,
        glossiness=1.0,
        emission=0.0,
        opacity=1.0,
        reflection_index=0.0,
        refraction_index=1.0,
        cast_user_data=False,
        chameleon_mode=ShadingChameleonMode.NONE,
        clipping_mode=ShadingClippingMode.NONE
    ):
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
        :clipping_mode: Clipping mode applied to the geometry attached to the material
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
        clipping_modes = list()
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
            clipping_modes.append(clipping_mode)
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
            cast_user_datas=cast_user_datas,
            clipping_modes=clipping_modes,
        )

    def apply_default_color_scheme(
        self,
        shading_mode,
        user_parameter=3.0,
        specular_exponent=5.0,
        glossiness=1.0,
        glycans=True,
        proteins=True,
        membranes=True,
        collagen=True,
    ):
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
            model_name = self.get_model_name(model_id)["name"]
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
                    specular_exponent=specular_exponent,
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
                    specular_exponent=specular_exponent,
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
                    specular_exponent=specular_exponent,
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
                    specular_exponent=specular_exponent,
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
                    specular_exponent=specular_exponent,
                )
            elif proteins and (
                self.NAME_RECEPTOR in model_name
                or self.NAME_TRANS_MEMBRANE in model_name
                or self.NAME_ION_CHANNEL in model_name
                or self.NAME_RNA_SEQUENCE in model_name
            ):
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
            elif proteins and (
                self.NAME_PROTEIN_S_CLOSED in model_name
                or self.NAME_PROTEIN_S_OPEN in model_name
                or self.NAME_PROTEIN_E in model_name
                or self.NAME_PROTEIN_M in model_name
            ):
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
            elif proteins and self.NAME_GLUCOSE in model_name:
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
            elif proteins and self.NAME_LACTOFERRIN in model_name:
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
            elif proteins and self.NAME_DEFENSIN in model_name:
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
                    specular_exponent=specular_exponent,
                )
            elif collagen and self.NAME_COLLAGEN in model_name:
                palette = list()
                emissions = list()
                for _ in range(nb_materials):
                    palette.append([1, 1, 1])
                    emissions.append(0.2)
                self.set_materials(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    diffuse_colors=palette,
                    specular_colors=palette,
                    emissions=emissions,
                )

    def set_material_colors_from_id(self, model_id):
        """
        Set material color according to material id.

        The id of the model represents a 16bit RGB value for which the corresponding color is set

        :model_id: Id of the model
        """
        if self._client is None:
            return

        material_ids = self.get_material_ids(model_id)["ids"]
        palette = list()
        for material_id in material_ids:
            r = (material_id >> 16) & 255
            g = (material_id >> 8) & 255
            b = material_id & 255
            palette.append([r / 255.0, g / 255.0, b / 255.0])
        return self.set_materials(
            [model_id], material_ids, diffuse_colors=palette, specular_colors=palette
        )

    def go_magnetic(
        self,
        colormap_filename=None,
        colormap_range=[0, 0.008],
        voxel_size=0.01,
        density=1.0,
        samples_per_pixel=64,
    ):
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
            current="advanced",
            samples_per_pixel=1,
            subsampling=8,
            max_accum_frames=samples_per_pixel,
        )

        # Build fields acceleration structures
        result = self.build_fields(voxel_size=voxel_size, density=density)
        fields_model_id = int(result["contents"])

        if colormap_filename:
            tf = TransferFunction(
                bioexplorer=self, model_id=fields_model_id, filename=colormap_filename
            )
            tf.set_range(colormap_range)
            return tf
        return None

    def build_fields(
        self, voxel_size, density=1.0, data_type=FieldDataType.POINT, model_ids=list()
    ):
        """
        Build fields acceleration structures and creates according data handler

        :voxel_size: Voxel size
        :voxel_size: Density of atoms to consider (between 0 and 1)
        :data_type: Type of field (FieldDataType.POINT or FIELD_DATA_TYPE_VECTOR)
        :model_ids: List of model ids used to build the field
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(voxel_size, float)
        assert isinstance(density, float)
        assert isinstance(model_ids, list)

        params = dict()
        params["voxelSize"] = voxel_size
        params["density"] = density
        params["dataType"] = data_type.value
        params["modelIds"] = model_ids
        return self._invoke_and_check("build-fields", params)

    def add_grid(
        self,
        min_value,
        max_value,
        interval,
        radius=1.0,
        opacity=0.5,
        show_axis=True,
        show_planes=True,
        show_full_grid=False,
        colored=True,
        position=Vector3(),
    ):
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

    def add_bounding_box(
        self,
        name,
        bottom_left_corner,
        top_right_corner,
        radius=1.0,
        color=Vector3(1.0, 1.0, 1.0),
    ):
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

    def add_box(
        self, name, bottom_left_corner, top_right_corner, color=Vector3(1.0, 1.0, 1.0)
    ):
        """
        Add a box to the scene

        :bottom_left_corner: Bottom left corner
        :top_right_corner: Top right corner
        :color: Color of the bounding box
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(bottom_left_corner, Vector3)
        assert isinstance(top_right_corner, Vector3)
        assert isinstance(color, Vector3)
        params = dict()
        params["name"] = name
        params["bottomLeft"] = bottom_left_corner.to_list()
        params["topRight"] = top_right_corner.to_list()
        params["color"] = color.to_list()
        return self._invoke_and_check("add-box", params)

    def add_spheres(
        self, name, positions, radii, color=Vector3(1.0, 1.0, 1.0), opacity=1.0
    ):
        """
        Add a list of spheres to the scene

        :name: Name of the list of spheres
        :positions: List of spheres positions
        :radii: Spheres Radii
        :color: RGB Color of the sphere (0..1)
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(positions, list)
        assert isinstance(radii, list)
        assert isinstance(color, Vector3)

        pos = list()
        for p in positions:
            isinstance(p, Vector3)
            pos.append(p.x)
            pos.append(p.y)
            pos.append(p.z)

        params = dict()
        params["name"] = name
        params["positions"] = pos
        params["radii"] = radii
        params["color"] = color.to_list()
        params["opacity"] = opacity
        return self._invoke_and_check("add-spheres", params)

    def add_cones(
        self,
        name,
        origins,
        targets,
        origins_radii,
        targets_radii,
        color=Vector3(1.0, 1.0, 1.0),
        opacity=1.0,
    ):
        """
        Add a cone to the scene

        :name: Name of the cone
        :origins: Origins of the cone
        :targets: Targets of the cone
        :origins_radii: Origin radii of the cones
        :targets_radii: target radii of the cones
        :color: RGB Color of the cone (0..1)
        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(origins, list)
        assert isinstance(targets, list)
        assert isinstance(origins_radii, list)
        assert isinstance(targets_radii, list)
        assert isinstance(color, Vector3)
        assert isinstance(opacity, float)

        origins_as_floats = list()
        for origin in origins:
            origins_as_floats.append(origin.x)
            origins_as_floats.append(origin.y)
            origins_as_floats.append(origin.z)

        targets_as_floats = list()
        for target in targets:
            targets_as_floats.append(target.x)
            targets_as_floats.append(target.y)
            targets_as_floats.append(target.z)

        params = dict()
        params["name"] = name
        params["origins"] = origins_as_floats
        params["targets"] = targets_as_floats
        params["originsRadii"] = origins_radii
        params["targetsRadii"] = targets_radii
        params["color"] = color.to_list()
        params["opacity"] = opacity
        return self._invoke_and_check("add-cones", params)

    def add_sdf_demo(self):
        """
        Add an SDF demo model

        :return: Result of the request submission
        """
        if self._client is None:
            return

        return self._invoke_and_check("add-sdf-demo")

    def add_torus(
        self,
        name,
        position,
        outer_radius,
        inner_radius,
        displacement_parameters=Vector3(),
    ):
        """
        Add a torus to the scene

        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(name, str)
        assert isinstance(position, Vector3)
        assert isinstance(outer_radius, float)
        assert isinstance(inner_radius, float)
        assert isinstance(displacement_parameters, Vector3)

        params = dict()
        params["name"] = name
        params["position"] = position.to_list()
        params["outerRadius"] = outer_radius
        params["innerRadius"] = inner_radius
        params["displacement"] = displacement_parameters.to_list()
        return self._invoke_and_check("add-torus", params)

    def add_vesica(
        self,
        name,
        source_position,
        target_position,
        radius,
        displacement_parameters=Vector3(),
    ):
        """
        Add a vesica to the scene

        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(name, str)
        assert isinstance(source_position, Vector3)
        assert isinstance(target_position, Vector3)
        assert isinstance(radius, float)
        assert isinstance(displacement_parameters, Vector3)

        params = dict()
        params["name"] = name
        params["srcPosition"] = source_position.to_list()
        params["dstPosition"] = target_position.to_list()
        params["radius"] = radius
        params["displacement"] = displacement_parameters.to_list()
        return self._invoke_and_check("add-vesica", params)

    def add_ellipsoid(
        self,
        name,
        position,
        radii,
        displacement_parameters=Vector3(),
    ):
        """
        Add a ellipsoid to the scene

        :return: Result of the request submission
        """
        if self._client is None:
            return

        assert isinstance(name, str)
        assert isinstance(position, Vector3)
        assert isinstance(radii, Vector3)
        assert isinstance(displacement_parameters, Vector3)

        params = dict()
        params["name"] = name
        params["position"] = position.to_list()
        params["radii"] = radii.to_list()
        params["displacement"] = displacement_parameters.to_list()
        return self._invoke_and_check("add-ellipsoid", params)

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

    def set_general_settings(
        self,
        mesh_folder="/tmp",
        logging_level=1,
        database_logging_level=1,
        v1_compatibility=False,
        cache_enabled=False
    ):
        """
        Set general settings for the plugin

        :off_folder: Folder where off files are stored (to avoid recomputation of molecular surface)
        :logging_level: Back-end logging level (0=no information logs, 3=full logging)
        :database_logging_level: Back-end logging level for database (0=no information logs, 3=full
        logging)
        :cache_enabled: Enabled memory cache
        :return: Result of the request submission
        """
        self._v1_compatibility = v1_compatibility
        params = dict()
        params["meshFolder"] = mesh_folder
        params["loggingLevel"] = logging_level
        params["databaseLoggingLevel"] = database_logging_level
        params["v1Compatibility"] = v1_compatibility
        params["cacheEnabled"] = cache_enabled
        response = self._invoke_and_check("set-general-settings", params)
        return response

    def start_model_loading_transaction(self):
        """
        Starts the loading of models

        All models loaded from this point will only appear in the scene once the
        commit_model_loading_transaction function is invoked

        :return: Result of the request submission
        """
        params = dict()
        params["action"] = TransactionType.START.value
        return self._invoke("model-loading-transaction", params)

    def commit_model_loading_transaction(self):
        """
        Commits the loading of models

        :return: Result of the request submission
        """
        params = dict()
        params["action"] = TransactionType.COMMIT.value
        return self._invoke("model-loading-transaction", params)

    def get_out_of_core_configuration(self):
        """
        Returns the out-of-core configuration

        :rtype: string
        """
        result = self._invoke_and_check("get-out-of-core-configuration")
        d = dict()
        for param in result["contents"].split("|"):
            s = param.split("=")
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
        self,
        assembly_name,
        population_name,
        load_cells=True,
        cell_radius=1.0,
        load_meshes=False,
        region_sql_filter="",
        cell_sql_filter="",
        scale=Vector3(1.0, 1.0, 1.0),
        mesh_position=Vector3(),
        mesh_rotation=Quaternion(),
        mesh_scale=Vector3(),
    ):
        """
        Add a brain atlas the 3D scene

        :assembly_name: Name of the assembly to which the astrocytes should be added
        :population_name: Name of the atlas population
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
        params["populationName"] = population_name
        params["loadCells"] = load_cells
        params["cellRadius"] = cell_radius
        params["loadMeshes"] = load_meshes
        params["cellSqlFilter"] = cell_sql_filter
        params["regionSqlFilter"] = region_sql_filter
        params["scale"] = scale.to_list()
        params["meshPosition"] = mesh_position.to_list()
        params["meshRotation"] = list(mesh_rotation)
        params["meshScale"] = mesh_scale.to_list()
        return self._invoke_and_check("add-atlas", params)

    def add_vasculature(
        self,
        assembly_name,
        population_name,
        color_scheme=VascularColorScheme.NONE,
        realism_level=VascularRealismLevel.NONE,
        section_gids=list(),
        load_capilarities=False,
        representation=VascularRepresentation.SEGMENT,
        radius_multiplier=1.0,
        sql_filter="",
        scale=Vector3(1.0, 1.0, 1.0),
        animation_params=CellAnimationParams(),
        displacement_params=VasculatureDisplacementParams(),
        align_to_grid=0.0,
    ):
        """
        Add a vasculature to the 3D scene

        :assembly_name: Name of the assembly to which the vasculature should be added
        :population_name Name of the node population
        :color_scheme: Color scheme applied to the vasculature (node, graph, section, etc)
        :realism_level: Use sign distance fields geometry to create the vasculature. Defaults to
        none
        :node_gids: List of segment GIDs to load. Defaults to list()
        :representation: Representation of the vasculature geometry (Graph, sections or segments)
        :radius_multiplier: Applies the multiplier to all radii of the vasculature sections
        :sql_filter: Condition added to the SQL statement loading the vasculature
        :scale: Scale in the 3D scene
        :animation_params: Extra optional parameters for animation purposes
        :displacement_params: Extra optional parameters for geometry displacement
        :align_to_grid: If different from zero, 3D position are aligned to the specified grid size

        :return: Result of the request submission
        """
        assert isinstance(section_gids, list)
        assert isinstance(scale, Vector3)
        assert isinstance(animation_params, CellAnimationParams)
        assert isinstance(displacement_params, VasculatureDisplacementParams)

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["colorScheme"] = color_scheme.value
        params["realismLevel"] = realism_level.value
        params["gids"] = section_gids
        params["loadCapilarities"] = load_capilarities
        params["representation"] = representation.value
        params["radiusMultiplier"] = radius_multiplier
        params["sqlFilter"] = sql_filter
        params["scale"] = scale.to_list()
        params["animationParams"] = animation_params.to_list()
        params["displacementParams"] = displacement_params.to_list()
        params["alignToGrid"] = align_to_grid
        return self._invoke_and_check("add-vasculature", params)

    def get_vasculature_info(self, assembly_name):
        """
        Return information about the vasculature

        :return: A dictionary of attributes (Model Id, Number of edges, pairs, sections and
                 sub-graphs)
        """
        params = dict()
        params["name"] = assembly_name
        response = self._invoke_and_check("get-vasculature-info", params)
        vasculature_info = dict()
        for param in response["contents"].split(self.CONTENTS_DELIMITER):
            s = param.split("=")
            vasculature_info[s[0]] = int(s[1])
        return vasculature_info

    def set_vasculature_report(
        self, assembly_name, population_name, report_simulation_id, show_evolution=False
    ):
        """
        Attach a simulation report to the vasculature

        :assembly_name: Name of the assembly containing the vasculature
        :population_name: Name of the node population in the report. Defaults to 'vasculature'
        :report_simulation_id: Report simulation identifier
        :show_evolution: Show value evolution (percentage) between 2 time steps. Raw value otherwise

        :return: Result of the request submission
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["simulationReportId"] = report_simulation_id
        params["showEvolution"] = show_evolution
        return self._invoke_and_check("set-vasculature-report", params)

    def set_vasculature_radius_report(
        self, assembly_name, population_name, report_simulation_id, frame, amplitude=1.0
    ):
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
        params["frame"] = frame
        params["amplitude"] = amplitude
        return self._invoke_and_check("set-vasculature-radius-report", params)

    def add_astrocytes(
        self,
        assembly_name,
        population_name,
        vasculature_population_name="",
        connectome_population_name="",
        realism_level=MorphologyRealismLevel.NONE,
        load_somas=True,
        load_dendrites=True,
        generate_internals=False,
        load_micro_domains=False,
        morphology_representation=MorphologyRepresentation.SEGMENT,
        micro_domain_representation=MicroDomainRepresentation.MESH,
        morphology_color_scheme=MorphologyColorScheme.NONE,
        population_color_scheme=PopulationColorScheme.NONE,
        radius_multiplier=1.0,
        sql_filter="",
        scale=Vector3(1.0, 1.0, 1.0),
        animation_params=CellAnimationParams(),
        displacement_params=AstrocyteDisplacementParams(),
        max_distance_to_soma=0.0,
        align_to_grid=0.0,
    ):
        """
        Add a population of astrocytes to the 3D scene

        :assembly_name: Name of the assembly to which the astrocytes should be added
        :population_name: Name of the population of astrocytes
        :vasculature_population_name: Name of the vasculature population. Automatically load
                                      end-feet if not empty
        :connectome_population_name: Name of the connectome population. Automatically load
                                      end-feet if not empty
        :load_somas: Load somas if set to true
        :load_dendrites: Load dendrites if set to true
        :generate_internals: Generate internals (Nucleus and mitochondria)
        :load_micro_domains: Load micro-domains if set to true
        :realism_level: Use sign distance fields geometry to create the astrocytes. Defaults to none
        :morphology_representation: Geometry representation (graph, section or segment)
        :morphology_color_scheme: Color scheme of the sections of the astrocytes
        :populationColorScheme: Color scheme of the population of astrocytes
        :radius_multiplier: Applies the multiplier to all radii of the astrocyte sections
        :sql_filter: Condition added to the SQL statement loading the astrocytes
        :scale: Scale in the 3D scene
        :animation_params: Extra optional parameters for animation purposes
        :displacement_params: Extra optional parameters for geometry displacement
        :max_distance_to_soma: Maximum distance to soma for segment loading. Ignored if 0.
        :align_to_grid: If different from zero, 3D position are aligned to the specified grid size

        :return: Result of the request submission
        """
        assert isinstance(scale, Vector3)
        assert isinstance(animation_params, CellAnimationParams)
        assert isinstance(displacement_params, AstrocyteDisplacementParams)
        if vasculature_population_name != '':
            if connectome_population_name == '':
                raise RuntimeError("A connectome population must be specified together with the \
                                   vasculature population")

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["vasculaturePopulationName"] = vasculature_population_name
        params["connectomePopulationName"] = connectome_population_name
        params["loadSomas"] = load_somas
        params["loadDendrites"] = load_dendrites
        params["generateInternals"] = generate_internals
        params["loadMicroDomains"] = load_micro_domains
        params["realismLevel"] = realism_level.value
        params["morphologyRepresentation"] = morphology_representation.value
        params["microDomainRepresentation"] = micro_domain_representation.value
        params["morphologyColorScheme"] = morphology_color_scheme.value
        params["populationColorScheme"] = population_color_scheme.value
        params["radiusMultiplier"] = radius_multiplier
        params["sqlFilter"] = sql_filter
        params["scale"] = scale.to_list()
        params["animationParams"] = animation_params.to_list()
        params["displacementParams"] = displacement_params.to_list()
        params["maxDistanceToSoma"] = max_distance_to_soma
        params["alignToGrid"] = align_to_grid
        return self._invoke_and_check("add-astrocytes", params)

    def add_neurons(
        self,
        assembly_name,
        population_name,
        realism_level=MorphologyRealismLevel.NONE,
        load_somas=True,
        load_axon=True,
        load_basal_dendrites=True,
        load_apical_dendrites=True,
        synapses_type=NeuronSynapseType.NONE,
        generate_internals=False,
        generate_externals=False,
        generate_varicosities=False,
        show_membrane=True,
        morphology_representation=MorphologyRepresentation.SEGMENT,
        morphology_color_scheme=MorphologyColorScheme.NONE,
        population_color_scheme=PopulationColorScheme.NONE,
        radius_multiplier=1.0,
        report_params=NeuronReportParams(),
        sql_node_filter="",
        sql_section_filter="",
        scale=Vector3(1.0, 1.0, 1.0),
        animation_params=CellAnimationParams(),
        displacement_params=NeuronDisplacementParams(),
        max_distance_to_soma=0.0,
        align_to_grid=0.0,
    ):
        """
        Add a population of astrocytes to the 3D scene

        :assembly_name: Name of the assembly to which the astrocytes should be added
        :population_name: Name of the population of astrocytes
        :realism_level: Use sign distance fields geometry to create the astrocytes. Defaults to none
        :load_somas: Load somas if set to true
        :load_axon: Load axon if set to true
        :load_basal_dendrites: Load basal dendrites if set to true
        :load_apical_dendrites: Load apical dendrites if set to true
        :load_synapses: Load synapses if set to true
        :generate_internals: Generate internals (Nucleus and mitochondria)
        :generate_externals: Generate externals (Myelin sheath)
        :show_membrane: Show membrane (Typically used to isolate internal and external components
        :generate_varicosities: Generate random varicosities along the axon
        :morphology_representation: Geometry representation (graph, section or segment)
        :morphology_color_scheme: Color scheme of the sections of the astrocytes
        :populationColorScheme: Color scheme of the population of astrocytes
        :radius_multiplier: Applies the multiplier to all radii of the astrocyte sections
        :report_params: Report parameters (Optional)
        :sql_node_filter: Condition added to the SQL statement loading the nodes
        :sql_section_filter: Condition added to the SQL statement loading the sections
        :scale: Scale in the 3D scene
        :animation_params: Extra optional parameters for animation purposes
        :displacement_params: Extra optional parameters for geometry displacement purposes
        :max_distance_to_soma: Maximum distance to soma for segment loading. Ignored if 0.
        :align_to_grid: If different from zero, 3D position are aligned to the specified grid size

        :return: Result of the request submission
        """
        assert isinstance(scale, Vector3)
        assert isinstance(animation_params, CellAnimationParams)
        assert isinstance(displacement_params, NeuronDisplacementParams)
        assert isinstance(report_params, NeuronReportParams)

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["loadSomas"] = load_somas
        params["loadAxon"] = load_axon
        params["loadBasalDendrites"] = load_basal_dendrites
        params["loadApicalDendrites"] = load_apical_dendrites
        params["synapsesType"] = synapses_type.value
        params["generateInternals"] = generate_internals
        params["generateExternals"] = generate_externals
        params["showMembrane"] = show_membrane
        params["generateVaricosities"] = generate_varicosities
        params["realismLevel"] = realism_level.value
        params["morphologyRepresentation"] = morphology_representation.value
        params["morphologyColorScheme"] = morphology_color_scheme.value
        params["populationColorScheme"] = population_color_scheme.value
        params["radiusMultiplier"] = radius_multiplier
        params["reportParams"] = report_params.to_list()
        params["sqlNodeFilter"] = sql_node_filter
        params["sqlSectionFilter"] = sql_section_filter
        params["scale"] = scale.to_list()
        params["animationParams"] = animation_params.to_list()
        params["displacementParams"] = displacement_params.to_list()
        params["maxDistanceToSoma"] = max_distance_to_soma
        params["alignToGrid"] = align_to_grid
        return self._invoke_and_check("add-neurons", params)

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
        response = self._invoke("get-neuron-section-points", params)
        if not response["status"]:
            raise RuntimeError("Failed to get neuron section points")
        points = response["points"]
        ps = list()
        for i in range(0, len(points), 4):
            p = list()
            for j in range(4):
                p.append(points[i + j])
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
        response = self._invoke("get-neuron-varicosities", params)
        if not response["status"]:
            raise RuntimeError("Failed to get neuron varicosities")
        points = response["points"]
        ps = list()
        for i in range(0, len(points), 3):
            p = list()
            for j in range(3):
                p.append(points[i + j])
            ps.append(p)
        return ps

    def set_spike_report_visualization_settings(
        self, model_id, rest_voltage=-65, spiking_voltage=-10, decay_speed=5.0
    ):
        """
        Set visualization settings for spike report

        :model_id: Id of the model holding the spike report
        :rest_voltage: Rest voltage. Defaults to -65.
        :spiking_voltage: Spiking voltage. Defaults to -10.
        :decay_speed: Decay speed. Defaults to 5.0.

        :return: Result of the request submission
        """
        params = dict()
        params["modelId"] = model_id
        params["restVoltage"] = rest_voltage
        params["spikingVoltage"] = spiking_voltage
        params["timeInterval"] = 0.0
        params["decaySpeed"] = decay_speed
        return self._invoke_and_check("set-spike-report-visualization-settings", params)

    def add_white_matter(
        self,
        assembly_name,
        population_name,
        radius=1.0,
        sql_filter="",
        scale=Vector3(1.0, 1.0, 1.0),
    ):
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
        return self._invoke_and_check("add-white-matter", params)

    def add_synapses(
        self,
        assembly_name,
        population_name,
        radius_multiplier=1.0,
        representation=SynapseRepresentation.SPHERE,
        realism_level=VascularRealismLevel.NONE,
        sql_filter="",
        displacement_params=list(),
    ):
        """
        Add synapse efficacy report to the 3D scene

        :assembly_name: Name of the assembly to which the vasculature should be added
        :population_name Name of the node population
        :radius_multiplier: Multiplies the spine radius by the specified value
        :representation: Representation of the synapse (sphere or spine)
        :realism_level: Use sign distance fields geometry to create the spines. Defaults to none
        :sql_filter: Condition added to the SQL statement loading the synapses

        :return: Result of the request submission
        """
        assert isinstance(displacement_params, list)

        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["radiusMultiplier"] = radius_multiplier
        params["representation"] = representation.value
        params["realismLevel"] = realism_level
        params["sqlFilter"] = sql_filter
        params["displacementParams"] = displacement_params
        return self._invoke_and_check("add-synapses", params)

    def add_synapse_efficacy_report(
        self,
        assembly_name,
        population_name,
        simulation_report_id,
        radius=1.0,
        sql_filter="",
        align_to_grid=0.0,
    ):
        """
        Add synapse efficacy report to the 3D scene

        :assembly_name: Name of the assembly to which the vasculature should be added
        :population_name Name of the node population
        :simulation_report_id: Identifier of the simulation report
        :radius: Applies the radius to the white matter streamlines
        :sql_filter: Condition added to the SQL statement loading the synapses
        :align_to_grid: If different from zero, 3D position are aligned to the specified grid size

        :return: Result of the request submission
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["radius"] = radius
        params["sqlFilter"] = sql_filter
        params["simulationReportId"] = simulation_report_id
        params["alignToGrid"] = align_to_grid
        return self._invoke_and_check("add-synapse-efficacy", params)

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
        response = self._invoke("look-at", params)
        q = response["rotation"]
        return Quaternion(q[3], q[0], q[1], q[2])

    def add_synaptome(
        self,
        assembly_name,
        population_name,
        radius=1.0,
        force=1.0,
        sql_node_filter="",
        sql_edge_filter="",
    ):
        """
        Add synaptome to the 3D scene

        :assembly_name: Name of the assembly to which the synaptome should be added
        :population_name: Name of the node population
        :radius: Radius of the sphere representing the node
        :sql_node_filter: Condition added to the SQL statement loading the neurons
        :sql_edge_filter: Condition added to the SQL statement loading the synapses

        :return: Result of the request submission
        """
        params = dict()
        params["assemblyName"] = assembly_name
        params["populationName"] = population_name
        params["radius"] = radius
        params["force"] = force
        params["sqlNodeFilter"] = sql_node_filter
        params["sqlEdgeFilter"] = sql_edge_filter
        return self._invoke_and_check("add-synaptome", params)


# Private classes


class AssemblyProtein:
    """An AssemblyProtein is a Protein that belongs to an assembly"""

    def __init__(
        self,
        assembly_name,
        name,
        source,
        atom_radius_multiplier=1.0,
        load_bonds=True,
        representation=ProteinRepresentation.ATOMS,
        load_non_polymer_chemicals=False,
        load_hydrogen=True,
        chain_ids=list(),
        recenter=True,
        occurrences=1,
        transmembrane_params=Vector2(),
        position=Vector3(),
        rotation=Quaternion(),
        allowed_occurrences=list(),
        animation_params=MolecularSystemAnimationParams(),
        constraints=list(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)

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
