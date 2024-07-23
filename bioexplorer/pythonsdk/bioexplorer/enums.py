#!/usr/bin/env

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
Module enums

This module provides enums for various types of parameters and settings used in the application.
"""

from enum import Enum


class TransactionType(Enum):
    """Enum representing types of transactions."""

    START = 0
    COMMIT = 1


class ProteinColorScheme(Enum):
    """Enum representing different protein color schemes."""

    NONE = 0
    ATOMS = 1
    CHAINS = 2
    RESIDUES = 3
    AMINO_ACID_SEQUENCE = 4
    GLYCOSYLATION_SITE = 5
    REGION = 6


class ShadingMode(Enum):
    """Enum representing different shading modes."""

    NONE = 0
    BASIC = 1
    DIFFUSE = 2
    ELECTRON = 3
    CARTOON = 4
    ELECTRON_TRANSPARENCY = 5
    PERLIN = 6
    DIFFUSE_TRANSPARENCY = 7
    CHECKER = 8
    GOODSELL = 9
    NORMAL = 10


class ShadingChameleonMode(Enum):
    """Enum representing different shading chameleon modes."""

    NONE = 0
    EMITTER = 1
    RECEIVER = 2


class ShadingClippingMode(Enum):
    """Enum representing different shading clipping modes."""

    NONE = 0
    PLANE = 1
    SPHERE = 2


class CameraProjection(Enum):
    """Enum representing different camera projection types."""

    PERSPECTIVE = 0
    FISHEYE = 1
    PANORAMIC = 2
    CYLINDRIC = 3


class RNAShape(Enum):
    """Enum representing different RNA shapes."""

    TREFOIL_KNOT = 0
    TORUS = 1
    STAR = 2
    SPRING = 3
    HEART = 4
    THING = 5
    MOEBIUS = 6


class RenderingQuality(Enum):
    """Enum representing different rendering quality levels."""

    LOW = 0
    HIGH = 1


class ProteinRepresentation(Enum):
    """Enum representing different protein representations."""

    ATOMS = 0
    ATOMS_AND_STICKS = 1
    CONTOURS = 2
    SURFACE = 3
    UNION_OF_BALLS = 4
    DEBUG = 5
    MESH = 6


class SurfactantType(Enum):
    """Enum representing different surfactant types."""

    NONE = 0
    PROTEIN_A = 0
    PROTEIN_D = 0


class AssemblyShape(Enum):
    """Enum representing different assembly shapes."""

    POINT = 0
    EMPTY_SPHERE = 1
    PLANE = 2
    SINUSOID = 3
    CUBE = 4
    FAN = 5
    BEZIER = 6
    MESH = 7
    HELIX = 8
    FILLED_SPHERE = 9
    CELL_DIFFUSION = 10


class MaterialOffset(Enum):
    """Enum representing different material offsets."""

    VARICOSITY = 0
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    AFFERENT_SYNAPSE = 5
    EFFERENT_SYNAPSE = 6
    MITOCHONDRION = 7
    NUCLEUS = 8
    MYELIN_SHEATH = 9


class FileFormat(Enum):
    """Enum representing different file formats."""

    UNSPECIFIED = 0
    XYZ_BINARY = 1
    XYZR_BINARY = 2
    XYZRV_BINARY = 3
    XYZ_ASCII = 4
    XYZR_ASCII = 5
    XYZRV_ASCII = 6
    XYZR_RGB_ASCII = 7


class PositionConstraint(Enum):
    """Enum representing different position constraints."""

    INSIDE = 0
    OUTSIDE = 1


class VascularRealismLevel(Enum):
    """Enum representing different levels of vascular realism."""

    NONE = 0
    SECTION = 1
    BIFURCATION = 2
    ALL = 255


class VascularRepresentation(Enum):
    """Enum representing different vascular representations."""

    GRAPH = 0
    SECTION = 1
    SEGMENT = 2
    OPTIMIZED_SEGMENT = 3
    BEZIER = 4


class VascularColorScheme(Enum):
    """Enum representing different vascular color schemes."""

    NONE = 0
    NODE = 1
    SECTION = 2
    SUBGRAPH = 3
    PAIR = 4
    ENTRYNODE = 5
    RADIUS = 6
    SECTION_POINTS = 7
    SECTION_ORIENTATION = 8
    REGION = 9


class MorphologyRepresentation(Enum):
    """Enum representing different morphology representations."""

    GRAPH = 0
    SECTION = 1
    SEGMENT = 2
    ORIENTATION = 3
    BEZIER = 4
    CONTOUR = 5
    SURFACE = 6


class MorphologyRealismLevel(Enum):
    """Enum representing different levels of morphology realism."""

    NONE = 0
    SOMA = 1
    AXON = 2
    DENDRITE = 4
    INTERNALS = 8
    EXTERNALS = 16
    SPINE = 32
    END_FOOT = 64
    ALL = 255


class MorphologyColorScheme(Enum):
    """Enum representing different morphology color schemes."""

    NONE = 0
    SECTION_TYPE = 1
    SECTION_ORIENTATION = 2
    DISTANCE_TO_SOMA = 3


class SynapseRepresentation(Enum):
    """Enum representing different synapse representations."""

    SPHERE = 0
    SPINE = 1


class MicroDomainRepresentation(Enum):
    """Enum representing different microdomain representations."""

    MESH = 0
    CONVEX_HULL = 1
    SURFACE = 2


class PopulationColorScheme(Enum):
    """Enum representing different population color schemes."""

    NONE = 0
    ID = 1


class NeuronSynapseType(Enum):
    """Enum representing different types of neuron synapses."""

    NONE = 0
    AFFERENT = 1
    EFFERENT = 2
    DEBUG = 4
    ALL = 8


class NeuronMaterial(Enum):
    """Enum representing different neuron materials."""

    VARICOSITY = 0
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    AFFERENT_SYNAPSE = 5
    EFFERENT_SYNAPSE = 6
    MITOCHONDRION = 7
    NUCLEUS = 8
    MYELIN_SHEATH = 9


class AstrocyteMaterial(Enum):
    """Enum representing different astrocyte materials."""

    SOMA = 1
    END_FOOT = 4
    MICRO_DOMAIN = 5
    MITOCHONDRION = 7
    NUCLEUS = 8


class FieldDataType(Enum):
    """Enum representing different field data types."""

    POINT = 0
    VECTOR = 1
