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
Module molecular_systems

This module provides classes to define molecular systems and cells.
"""

import os
from .math_utils import Vector2, Vector3, Quaternion
from .animation_parameters import CellAnimationParams, MolecularSystemAnimationParams
from .enums import *


class Membrane:
    """A membrane is a shaped assembly of phospholipids"""

    def __init__(
        self,
        lipid_sources,
        lipid_rotation=Quaternion(),
        lipid_density=1.0,
        atom_radius_multiplier=1.0,
        load_bonds=False,
        representation=ProteinRepresentation.ATOMS_AND_STICKS,
        load_non_polymer_chemicals=False,
        chain_ids=list(),
        recenter=True,
        animation_params=MolecularSystemAnimationParams(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)
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

    def __init__(
        self,
        assembly_name,
        name,
        source,
        protein_name,
        atom_radius_multiplier=1.0,
        load_bonds=True,
        representation=ProteinRepresentation.ATOMS,
        recenter=True,
        chain_ids=list(),
        site_indices=list(),
        rotation=Quaternion(),
        animation_params=MolecularSystemAnimationParams(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)

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

    def __init__(
        self,
        source,
        shape,
        shape_params,
        protein_source="",
        values_range=Vector2(),
        curve_params=Vector3(),
        position=Vector3(),
        rotation=Quaternion(),
        atom_radius_multiplier=1.0,
        representation=ProteinRepresentation.ATOMS,
        animation_params=MolecularSystemAnimationParams(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)

        self.source = source
        self.protein_source = protein_source
        self.shape = shape.value
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

    def __init__(
        self, name, shape, shape_params, membrane, proteins=list(), shape_mesh_source=""
    ):
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

    def __init__(
        self,
        name,
        source,
        occurrences=1,
        load_bonds=False,
        load_hydrogen=False,
        load_non_polymer_chemicals=False,
        position=Vector3(),
        rotation=Quaternion(),
        allowed_occurrences=list(),
        chain_ids=list(),
        transmembrane_params=Vector2(),
        animation_params=MolecularSystemAnimationParams(),
    ):
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
        assert isinstance(animation_params, MolecularSystemAnimationParams)

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

    def __init__(
        self,
        name,
        shape_params,
        protein_s=None,
        protein_e=None,
        protein_m=None,
        membrane=None,
        rna_sequence=None,
    ):
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
