# !/usr/bin/env python
"""BioExplorer class"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2024 Blue BrainProject / EPFL
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

from enum import Enum

class TransactionType(Enum):
    START = 0
    COMMIT = 1

class ProteinColorScheme(Enum):
    NONE = 0
    ATOMS = 1
    CHAINS = 2
    RESIDUES = 3
    AMINO_ACID_SEQUENCE = 4
    GLYCOSYLATION_SITE = 5
    REGION = 6

class ShadingMode(Enum):
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
    NONE = 0
    EMITTER = 1
    RECEIVER = 2

class ShadingClippingMode(Enum):
    NONE = 0
    PLANE = 1
    SPHERE = 2

class CameraProjection(Enum):
    PERSPECTIVE = 0
    FISHEYE = 1
    PANORAMIC = 2
    CYLINDRIC = 3

class RNAShape(Enum):
    TREFOIL_KNOT = 0
    TORUS = 1
    STAR = 2
    SPRING = 3
    HEART = 4
    THING = 5
    MOEBIUS = 6

class RenderingQuality(Enum):
    LOW = 0
    HIGH = 1

class ProteinRepresentation(Enum):
    ATOMS = 0
    ATOMS_AND_STICKS = 1
    CONTOURS = 2
    SURFACE = 3
    UNION_OF_BALLS = 4
    DEBUG = 5
    MESH = 6

class SurfactantType(Enum):
    NONE = 0
    PROTEIN_A = 0
    PROTEIN_D = 0

class AssemblyShape(Enum):
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
    UNSPECIFIED = 0
    XYZ_BINARY = 1
    XYZR_BINARY = 2
    XYZRV_BINARY = 3
    XYZ_ASCII = 4
    XYZR_ASCII = 5
    XYZRV_ASCII = 6

class PositionConstraint(Enum):
    INSIDE = 0
    OUTSIDE = 1

class VascularRealismLevel(Enum):
    NONE = 0
    SECTION = 1
    BIFURCATION = 2
    ALL = 255

class VascularRepresentation(Enum):
    GRAPH = 0
    SECTION = 1
    SEGMENT = 2
    OPTIMIZED_SEGMENT = 3
    BEZIER = 4

class VascularColorScheme(Enum):
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
    GRAPH = 0
    SECTION = 1
    SEGMENT = 2
    ORIENTATION = 3
    BEZIER = 4
    CONTOUR = 5
    SURFACE = 6

class MorphologyRealismLevel(Enum):
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
    NONE = 0
    SECTION_TYPE = 1
    SECTION_ORIENTATION = 2
    DISTANCE_TO_SOMA = 3    

class SynapseRepresentation(Enum):
    SPHERE = 0
    SPINE = 1

class MicroDomainRepresentation(Enum):
    MESH = 0
    CONVEX_HULL = 1
    SURFACE = 2

class PopulationColorScheme(Enum):
    NONE = 0
    ID = 1

class NeuronSynapseType(Enum):
    NONE = 0
    AFFERENT = 1
    EFFERENT = 2
    DEBUG = 4
    ALL = 8

class NeuronMaterial(Enum):
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
    SOMA = 1
    END_FOOT = 4
    MICRO_DOMAIN = 5
    MITOCHONDRION = 7
    NUCLEUS = 8

class FieldDataType(Enum):
    POINT = 0
    VECTOR = 1