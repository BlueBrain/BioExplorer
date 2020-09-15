#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of BioExplorer
# <https://github.com/BlueBrain/BioExplorer>
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# All rights reserved. Do not distribute without further notice.

import seaborn as sns
import math
from brayns import Client
from .version import __version__


class Vector3:

    def __init__(self, *args, **kwargs):
        if len(args) not in [0, 3]:
            raise RuntimeError('Invalid number of floats (0 or 3 expected)')

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        if len(args) == 3:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]

    def to_list(self):
        return [self.x, self.y, self.z]


class Vector2:

    def __init__(self, *args, **kwargs):
        if len(args) not in [0, 2]:
            raise RuntimeError('Invalid number of floats (0 or 2 expected)')

        self.x = 0.0
        self.y = 0.0
        if len(args) == 2:
            self.x = args[0]
            self.y = args[1]

    def to_list(self):
        return [self.x, self.y]


class Quaternion:

    def __init__(self, *args, **kwargs):
        if len(args) not in [0, 4]:
            raise RuntimeError('Invalid number of floats (0 or 4 expected)')

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0
        if len(args) == 4:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
            self.w = args[3]

    def to_list(self):
        return [self.x, self.y, self.z, self.w]


class BioExplorer(object):
    """ VirusExplorer """

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

    IMAGE_QUALITY_LOW = 0
    IMAGE_QUALITY_HIGH = 1

    REPRESENTATION_ATOMS = 0
    REPRESENTATION_ATOMS_AND_STICKS = 1
    REPRESENTATION_CONTOURS = 2
    REPRESENTATION_SURFACE = 3

    ASSEMBLY_SHAPE_SPHERICAL = 0
    ASSEMBLY_SHAPE_PLANAR = 1
    ASSEMBLY_SHAPE_SINUSOIDAL = 2
    ASSEMBLY_SHAPE_CUBIC = 3
    ASSEMBLY_SHAPE_FAN = 4
    ASSEMBLY_SHAPE_BEZIER = 5

    NAME_PROTEIN_S_OPEN = 'Protein S (open)'
    NAME_PROTEIN_S_CLOSED = 'Protein S (closed)'
    NAME_PROTEIN_M = 'Protein M'
    NAME_PROTEIN_E = 'Protein E'
    NAME_RNA_SEQUENCE = 'RNA sequence'
    NAME_MEMBRANE = 'Membrane'
    NAME_TRANS_MEMBRANE = 'Trans-membrane'
    NAME_RECEPTOR = 'Receptor'
    NAME_PROTEIN = 'Protein'

    NAME_SURFACTANT_HEAD = 'Head'
    NAME_COLLAGEN = 'Collagen'
    NAME_GLUCOSE = 'Glucose'

    NAME_LACTOFERRIN = 'Lactoferrin'
    NAME_DEFENSIN = 'Defensin'

    NAME_GLYCAN_HIGH_MANNOSE = 'High-mannose'
    NAME_GLYCAN_O_GLYCAN = 'O-glycan'
    NAME_GLYCAN_HYBRID = 'Hybrid'
    NAME_GLYCAN_COMPLEX = 'Complex'

    SURFACTANT_BRANCH = 0
    SURFACTANT_PROTEIN_A = 1
    SURFACTANT_PROTEIN_D = 2

    def __init__(self, url=None):
        """
        Create a new Steps instance
        """
        self._url = url
        self._client = None
        if url is not None:
            self._client = Client(url)

        if __version__ != self.version():
            raise RuntimeError(
                'Wrong version of the back-end. Use version ' + __version__ + \
                ' for this version of the BioExplorer python library')

    def __str__(self):
        """
        @return: A pretty-print of the class
        """
        return "Blue Brain BioExplorer"

    def core_api(self):
        """
        @return: Access to underlying core API (Brayns' core API)
        """
        return self._client

    def version(self):
        """
        @return: The version of the BioExplorer library
        """
        if self._client is None:
            return __version__

        result = self._client.rockets_client.request(method='version')
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result['contents']

    def reset(self):
        """
        Removes all assemblies

        @return: Result of the call to the BioExplorer backend
        """
        if self._client is None:
            return

        ids = list()
        for model in self._client.scene.models:
            ids.append(model['id'])
        return self._client.remove_model(array=ids)

    def export_to_cache(self, filename):
        """
        Exports current scene to file as an optimized binary cache file

        @param filename: Full path of the binary cache file
        @return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params['filename'] = filename
        result = self._client.rockets_client.request(method='export-to-cache', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result

    def export_to_xyzr(self, filename):
        """
        Exports current scene to file as a binary XYZR file

        @param filename: Full path of the binary XYZR file
        @return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params['filename'] = filename
        result = self._client.rockets_client.request(method='export-to-xyzr', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result

    def remove_assembly(self, name):
        """
        Removes the specified assembly

        @param name: Name of the assembly
        @return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params['name'] = name
        params['position'] = Vector3().to_list()
        params['clippingPlanes'] = list()
        result = self._client.rockets_client.request(method='remove-assembly', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result

    def add_coronavirus(self, name, resource_folder, radius=45.0, nb_protein_s=62, nb_protein_m=50, nb_protein_e=42,
                        open_protein_s_indices=[1], atom_radius_multiplier=1.0, add_glycans=False,
                        representation=REPRESENTATION_ATOMS, clipping_planes=list(), position=Vector3(),
                        orientation=Quaternion()):

        """
        Add a virus with the default coronavirus parameters

        @param name: Name of the coronavirus
        @param resource_folder: Folder containing the resources of the virus components (PDB and RNA files)
        @param radius: Radius of the virus in nanometers
        @param nb_protein_s: Number of S proteins
        @param nb_protein_m: Number of M proteins
        @param nb_protein_e: Number of E proteins
        @param open_protein_s_indices: Indices for the open S proteins
        @param atom_radius_multiplier:
        @param add_glycans: Defines if glycans should be added
        @param atom_radius_multiplier: Multiplies atom radius by the specified value
        @param representation: Representation of the protein (Atoms, atoms and sticks, etc)
        @param clipping_planes: List of clipping planes to apply to the virus assembly
        @param position: Position of the virus in the scene
        @param orientation: Orientation of the protein in the scene
        """
        pdb_folder = resource_folder + 'pdb/'
        rna_folder = resource_folder + 'rna/'
        glycan_folder = pdb_folder + 'glycans/'

        open_conformation_indices = open_protein_s_indices
        closed_conformation_indices = list()
        for i in range(nb_protein_s):
            if i not in open_conformation_indices:
                closed_conformation_indices.append(i)

        virus_protein_s = Protein(sources=[pdb_folder + '6vyb.pdb', pdb_folder + 'sars-cov-2-v1.pdb'],
                                  number_of_instances=nb_protein_s,
                                  assembly_params=Vector2(11.5, 0.0), cutoff_angle=0.999,
                                  orientation=Quaternion(0.087, 0.0, 0.996, 0.0),
                                  instance_indices=[open_conformation_indices, closed_conformation_indices])

        # Protein M (QHD43419)
        virus_protein_m = Protein(sources=[pdb_folder + 'QHD43419a.pdb'],
                                  number_of_instances=nb_protein_m, assembly_params=Vector2(2.0, 0.0),
                                  cutoff_angle=0.999,
                                  orientation=Quaternion(0.99, 0.0, 0.0, 0.135))
        # Protein E (QHD43418 P0DTC4)
        virus_protein_e = Protein(sources=[pdb_folder + 'QHD43418a.pdb'],
                                  number_of_instances=nb_protein_e, assembly_params=Vector2(3.0, 0.0),
                                  cutoff_angle=0.9999,
                                  orientation=Quaternion(0.705, 0.705, -0.04, -0.04))

        # Virus membrane
        virus_membrane = Membrane(sources=[pdb_folder + 'membrane/popc.pdb'], number_of_instances=15000)

        import math
        rna_sequence = RNASequence(source=rna_folder + 'sars-cov-2.rna', assembly_params=Vector2(radius / 4.0, 0.5),
                                   t_range=Vector2(0, 30.5 * math.pi), shape=self.RNA_SHAPE_TREFOIL_KNOT,
                                   shape_params=Vector3(1.51, 1.12, 1.93))

        coronavirus = Virus(name=name, protein_s=virus_protein_s, protein_e=virus_protein_e, protein_m=virus_protein_m,
                            membrane=virus_membrane, rna_sequence=rna_sequence, assembly_params=Vector2(radius, 1.5))

        self.add_virus(
            virus=coronavirus, representation=representation, atom_radius_multiplier=atom_radius_multiplier,
            clipping_planes=clipping_planes, position=position, orientation=orientation)

        if add_glycans:
            complex_paths = [glycan_folder + 'complex/5.pdb', glycan_folder + 'complex/15.pdb',
                             glycan_folder + 'complex/25.pdb', glycan_folder + 'complex/35.pdb']
            high_mannose_paths = [glycan_folder + 'high-mannose/1.pdb', glycan_folder + 'high-mannose/2.pdb',
                                  glycan_folder + 'high-mannose/3.pdb', glycan_folder + 'high-mannose/4.pdb']
            hybrid_paths = [glycan_folder + 'hybrid/20.pdb']
            o_glycan_paths = [glycan_folder + 'o-glycan/12.pdb']

            # High-mannose
            indices = [61, 122, 234, 603, 709, 717, 801, 1074]
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE, protein_name=self.NAME_PROTEIN_S_CLOSED,
                paths=high_mannose_paths, indices=indices, allowed_occurrences=closed_conformation_indices)
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE, protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=high_mannose_paths, indices=indices, index_offset=19,
                allowed_occurrences=open_conformation_indices)

            # Complex
            indices1 = [17, 74, 149, 165, 282, 331, 343, 616, 1098, 1134, 1158, 1173, 1194]
            indices2 = [17, 74, 149, 165, 282, 331, 343, 1098, 1134, 1158, 1173, 1194]
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_COMPLEX, protein_name=self.NAME_PROTEIN_S_CLOSED,
                paths=complex_paths, indices=indices1, allowed_occurrences=closed_conformation_indices)
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_COMPLEX, protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=complex_paths, indices=indices2, index_offset=19, allowed_occurrences=open_conformation_indices)

            # Hybrid
            indices = [657]
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_HYBRID, protein_name=self.NAME_PROTEIN_S_CLOSED,
                paths=hybrid_paths, indices=indices, allowed_occurrences=closed_conformation_indices)
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_HYBRID, protein_name=self.NAME_PROTEIN_S_OPEN,
                paths=hybrid_paths, indices=indices, index_offset=19, allowed_occurrences=open_conformation_indices)

            # O-Glycans
            for index in [323, 325]:
                o_glycan_name = name + '_' + self.NAME_GLYCAN_O_GLYCAN + '_' + str(index)
                o_glycan = Sugars(
                    assembly_name=name, name=o_glycan_name, source=o_glycan_paths[0],
                    protein_name=name + '_' + self.NAME_PROTEIN_S_CLOSED, site_indices=[index])
                self.add_sugars(o_glycan)

            # High-mannose glycans on Protein M
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_HIGH_MANNOSE, protein_name=self.NAME_PROTEIN_M,
                paths=high_mannose_paths)

            # Complex glycans on Protein E
            self.add_multiple_glycans(
                assembly_name=name, glycan_type=self.NAME_GLYCAN_COMPLEX, protein_name=self.NAME_PROTEIN_E,
                paths=complex_paths)

        # Apply default materials
        self.apply_default_color_scheme(shading_mode=self.SHADING_MODE_BASIC)

    def add_virus(self, virus, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS, clipping_planes=list(),
                  position=Vector3(), orientation=Quaternion()):

        """
        Adds a virus assembly to the scene

        @param virus: Description of the virus
        @param atom_radius_multiplier: Multiplies atom radius by the specified value
        @param representation: Representation of the protein (Atoms, atoms and sticks, etc)
        @param clipping_planes: List of clipping planes to apply to the virus assembly
        @param position: Position of the virus in the scene
        @param orientation: Orientation of the protein in the scene
        """
        assert isinstance(virus, Virus)
        assert isinstance(clipping_planes, list)
        assert isinstance(position, Vector3)
        assert isinstance(orientation, Quaternion)

        shape = self.ASSEMBLY_SHAPE_SPHERICAL
        _protein_s = virus.protein_s

        self.remove_assembly(virus.name)
        self.add_assembly(
            name=virus.name,
            clipping_planes=clipping_planes,
            position=position, orientation=orientation)

        if virus.protein_s is not None:
            radius = virus.protein_s.assembly_params.x + virus.assembly_params.x
            if len(_protein_s.instance_indices[0]) > 0:
                _protein_s_open = AssemblyProtein(
                    assembly_name=virus.name, name=virus.name + '_' + self.NAME_PROTEIN_S_OPEN,
                    source=_protein_s.sources[0],
                    shape=shape, load_hydrogen=_protein_s.load_hydrogen,
                    occurrences=_protein_s.number_of_instances,
                    assembly_params=Vector2(radius, _protein_s.assembly_params.y),
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=_protein_s.load_bonds,
                    load_non_polymer_chemicals=_protein_s.load_non_polymer_chemicals,
                    representation=representation, random_seed=1,
                    location_cutoff_angle=_protein_s.cutoff_angle,
                    orientation=_protein_s.orientation,
                    allowed_occurrences=_protein_s.instance_indices[0])
                self.add_assembly_protein(_protein_s_open)

            if len(_protein_s.instance_indices[1]) > 0:
                _protein_s_closed = AssemblyProtein(
                    assembly_name=virus.name, name=virus.name + '_' + self.NAME_PROTEIN_S_CLOSED,
                    source=_protein_s.sources[1],
                    shape=shape, load_hydrogen=_protein_s.load_hydrogen,
                    occurrences=_protein_s.number_of_instances,
                    assembly_params=Vector2(radius, _protein_s.assembly_params.y),
                    atom_radius_multiplier=atom_radius_multiplier,
                    load_bonds=_protein_s.load_bonds,
                    load_non_polymer_chemicals=_protein_s.load_non_polymer_chemicals,
                    representation=representation, random_seed=1,
                    location_cutoff_angle=_protein_s.cutoff_angle,
                    orientation=_protein_s.orientation,
                    allowed_occurrences=_protein_s.instance_indices[1])
                self.add_assembly_protein(_protein_s_closed)

        if virus.protein_m is not None:
            radius = virus.protein_m.assembly_params.x + virus.assembly_params.x
            _protein_m = AssemblyProtein(
                assembly_name=virus.name, name=virus.name + '_' + self.NAME_PROTEIN_M,
                source=virus.protein_m.sources[0], shape=shape,
                load_hydrogen=virus.protein_m.load_hydrogen,
                occurrences=virus.protein_m.number_of_instances,
                assembly_params=Vector2(radius, virus.protein_m.assembly_params.y),
                atom_radius_multiplier=atom_radius_multiplier,
                load_bonds=virus.protein_m.load_bonds,
                load_non_polymer_chemicals=virus.protein_m.load_non_polymer_chemicals,
                representation=representation, random_seed=2,
                location_cutoff_angle=virus.protein_m.cutoff_angle,
                orientation=virus.protein_m.orientation)
            self.add_assembly_protein(_protein_m)

        if virus.protein_e is not None:
            radius = virus.protein_e.assembly_params.x + virus.assembly_params.x
            _protein_e = AssemblyProtein(
                assembly_name=virus.name, name=virus.name + '_' + self.NAME_PROTEIN_E,
                source=virus.protein_e.sources[0], shape=shape,
                load_hydrogen=virus.protein_e.load_hydrogen,
                occurrences=virus.protein_e.number_of_instances,
                assembly_params=Vector2(radius, virus.protein_e.assembly_params.y),
                atom_radius_multiplier=atom_radius_multiplier,
                load_bonds=virus.protein_e.load_bonds,
                load_non_polymer_chemicals=virus.protein_e.load_non_polymer_chemicals,
                representation=representation, random_seed=3,
                location_cutoff_angle=virus.protein_e.cutoff_angle,
                orientation=virus.protein_e.orientation)
            self.add_assembly_protein(_protein_e)

        if virus.membrane is not None:
            virus.membrane.representation = representation
            virus.membrane.atom_radius_multiplier = atom_radius_multiplier
            self.add_membrane(
                assembly_name=virus.name,
                name=virus.name + '_' + self.NAME_MEMBRANE,
                membrane=virus.membrane, shape=BioExplorer.ASSEMBLY_SHAPE_SPHERICAL,
                position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_RADIAL,
                assembly_params=virus.assembly_params, random_seed=4)

        if virus.rna_sequence is not None:
            self.add_rna_sequence(
                assembly_name=virus.name,
                name=virus.name + '_' + self.NAME_RNA_SEQUENCE,
                rna_sequence=virus.rna_sequence)

    def add_cell(self, cell, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS, clipping_planes=list(),
                 position=Vector3()):
        """
        Add a cell assembly to the scene

        @param cell: Description of the cell
        @param atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        @param representation: Multiplies atom radius by the specified value
        @param clipping_planes: List of clipping planes to apply to the virus assembly
        @param position: Position of the cell in the scene
        """
        assert isinstance(cell, Cell)
        assert isinstance(clipping_planes, list)
        assert isinstance(position, Vector3)
        assert len(cell.receptor.sources) == 1

        _receptor = AssemblyProtein(
            assembly_name=cell.name,
            name=cell.name + '_' + self.NAME_RECEPTOR, shape=cell.shape,
            source=cell.receptor.sources[0], load_non_polymer_chemicals=cell.receptor.load_non_polymer_chemicals,
            occurrences=cell.receptor.number_of_instances,
            assembly_params=cell.size, atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=True, representation=representation, random_seed=1,
            position=cell.receptor.position, orientation=cell.receptor.orientation)

        self.remove_assembly(cell.name)
        self.add_assembly(name=cell.name, position=position, clipping_planes=clipping_planes)
        self.add_assembly_protein(_receptor)
        cell.membrane.representation = representation
        cell.membrane.atom_radius_multiplier = atom_radius_multiplier
        self.add_membrane(
            assembly_name=cell.name, name=cell.name + '_' + self.NAME_MEMBRANE, shape=cell.shape,
            assembly_params=cell.size, random_seed=2,
            position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL, membrane=cell.membrane)

    def add_volume(self, volume, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS, position=Vector3()):
        """
        Add a volume assembly to the scene

        @param volume: Description of the volume
        @param atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        @param representation: Multiplies atom radius by the specified value
        @param position: Position of the volume in the scene
        """
        assert isinstance(volume, Volume)
        assert isinstance(position, Vector3)
        assert len(volume.protein.sources) == 1

        _protein = AssemblyProtein(
            assembly_name=volume.name,
            name=volume.name + '_' + self.NAME_PROTEIN, shape=volume.shape,
            source=volume.protein.sources[0], load_non_polymer_chemicals=volume.protein.load_non_polymer_chemicals,
            occurrences=volume.protein.number_of_instances,
            assembly_params=volume.size, atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=True, representation=representation, random_seed=1,
            position=volume.protein.position, orientation=volume.protein.orientation)

        self.remove_assembly(volume.name)
        self.add_assembly(name=volume.name, position=position)
        self.add_assembly_protein(_protein)

    def add_surfactant(self, surfactant, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                       position=Vector3(), random_seed=0):
        """
        Add a surfactant assembly to the scene

        @param surfactant: Description of the surfactant
        @param atom_radius_multiplier: Representation of the protein (Atoms, atoms and sticks, etc)
        @param representation: Multiplies atom radius by the specified value
        @param position: Position of the volume in the scene
        @param random_seed: Random seed used to define the shape of the branches
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

        head_name = surfactant.name + '_' + self.NAME_SURFACTANT_HEAD
        branch_name = surfactant.name + '_' + self.NAME_COLLAGEN + '_'

        protein_sp_d = AssemblyProtein(
            assembly_name=surfactant.name,
            name=head_name, shape=shape,
            source=surfactant.head_source,
            occurrences=nb_branches,
            assembly_params=Vector2(collagen_size * (nb_collagens + 1) - 9.0, 0.0),
            atom_radius_multiplier=atom_radius_multiplier,
            random_seed=random_seed,
            representation=representation,
            orientation=Quaternion(-0.624, -0.417, 0.0, 0.661))

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
                    assembly_params=Vector2(collagen_size * (i + 1) - 7.0, 0.0),
                    random_seed=random_seed,
                    representation=representation))

        self.remove_assembly(surfactant.name)
        self.add_assembly(
            name=surfactant.name, position=position)

        for collagen in collagens:
            self.add_assembly_protein(collagen)
        self.add_assembly_protein(protein_sp_d)

    def add_assembly(self, name, clipping_planes=list(), position=Vector3(), orientation=Quaternion()):
        """
        Add an assembly to the scene

        @param name: Name of the assembly
        @param clipping_planes: List of clipping planes to apply to the virus assembly
        @param position: Position of the scene in the scene
        @param orientation: Orientation of the assembly in the scene
        """
        assert isinstance(clipping_planes, list)
        assert isinstance(position, Vector3)
        assert isinstance(orientation, Quaternion)

        clipping_planes_values = list()
        for plane in clipping_planes:
            for i in range(4):
                clipping_planes_values.append(plane[i])

        params = dict()
        params['name'] = name
        params['position'] = position.to_list()
        params['orientation'] = orientation.to_list()
        params['clippingPlanes'] = clipping_planes_values
        result = self._client.rockets_client.request(method='add-assembly', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def set_protein_color_scheme(self, assembly_name, name, color_scheme, palette_name='', palette_size=256,
                                 palette=list(), chain_ids=list()):
        """
        Set a color scheme to a protein

        @param assembly_name: Name of the assembly containing the protein
        @param name: Name of the protein
        @param color_scheme: Color scheme
        @param palette_name: Name of the Seaborn color palette
        @param palette_size: Size of the Seaborn color palette
        @param palette: Seaborn palette (overrides the palette_name and palette size if specified)
        @param chain_ids: Ids of the chains to which the color scheme should be applied
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(palette, list)
        assert isinstance(chain_ids, list)

        p = list()
        if len(palette) == 0 and palette_name != '':
            palette = sns.color_palette(palette_name, palette_size)

        for color in palette:
            for i in range(3):
                p.append(color[i])

        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = name
        params['colorScheme'] = color_scheme
        params['palette'] = p
        params['chainIds'] = chain_ids
        result = self._client.rockets_client.request(method='set-protein-color-scheme', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def set_protein_amino_acid_sequence_as_string(self, assembly_name, name, amino_acid_sequence):
        """
        Displays a specified amino acid sequence on the protein

        @param assembly_name: Name of the assembly containing the protein
        @param name: Name of the protein
        @param amino_acid_sequence: String containing the amino acid sequence
        @return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = name
        params['sequence'] = amino_acid_sequence
        result = self._client.rockets_client.request(method='set-protein-amino-acid-sequence-as-string', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def set_protein_amino_acid_sequence_as_range(self, assembly_name, name, amino_acid_range):
        """
        Displays a specified amino acid range on the protein

        @param assembly_name: Name of the assembly containing the protein
        @param name: Name of the protein
        @param amino_acid_range: Tuple containing the amino acid range
        @return: Result of the call to the BioExplorer backend
        """
        assert len(amino_acid_range) == 2
        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = name
        params['range'] = amino_acid_range
        result = self._client.rockets_client.request(method='set-protein-amino-acid-sequence-as-range', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def get_protein_amino_acid_information(self, assembly_name, name):
        """
        Returns amino acid information of the protein

        @param assembly_name: Name of the assembly containing the protein
        @param name: Name of the protein
        @return: Result of the call to the BioExplorer backend
        """
        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = name
        result = self._client.rockets_client.request(method='get-protein-amino-acid-information', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result['contents'].split()

    def add_rna_sequence(self, assembly_name, name, rna_sequence):
        """
        Add an RNA sequence object to an assembly
        @param assembly_name: Name of the assembly
        @param name: Name of the RNA sequence
        @param rna_sequence: Description of the RNA sequence
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(rna_sequence, RNASequence)
        t_range = Vector2(0.0, 2.0 * math.pi)
        if rna_sequence.t_range is None:
            ''' Defaults '''
            if rna_sequence.shape == self.RNA_SHAPE_TORUS:
                t_range = Vector2(0.0, 2.0 * math.pi)
            elif rna_sequence.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                t_range = Vector2(0.0, 4.0 * math.pi)
        else:
            t_range = rna_sequence.t_range

        shape_params = [1.0, 1.0, 1.0]
        if rna_sequence.shape_params is None:
            ''' Defaults '''
            if rna_sequence.shape == self.RNA_SHAPE_TORUS:
                shape_params = Vector3(0.5, 10.0, 0.0)
            elif rna_sequence.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                shape_params = Vector3(2.5, 2.0, 2.2)

        else:
            shape_params = rna_sequence.shape_params

        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = name
        params['contents'] = ''.join(open(rna_sequence.source).readlines())
        params['shape'] = rna_sequence.shape
        params['assemblyParams'] = rna_sequence.assembly_params.to_list()
        params['range'] = t_range.to_list()
        params['params'] = shape_params.to_list()
        result = self._client.rockets_client.request(method='add-rna-sequence', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def add_membrane(self, assembly_name, name, membrane, shape, position_randomization_type, assembly_params,
                     random_seed):
        """
        Add a membrane to the scene

        @param assembly_name: Name of the assembly
        @param name: Name of the cell
        @param membrane: Description of the membrane
        @param shape: Shape of the membrane
        @param position_randomization_type: Type of randomisation for the elements of the membrane
        @param assembly_params: Size of the membrane
        @param random_seed: Seed used to randomise position the elements in the membrane
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(membrane, Membrane)
        assert isinstance(assembly_params, Vector2)

        contents = ['', '', '', '']
        for i in range(len(membrane.sources)):
            contents[i] = ''.join(open(membrane.sources[i]).readlines())

        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = name
        params['content1'] = contents[0]
        params['content2'] = contents[1]
        params['content3'] = contents[2]
        params['content4'] = contents[3]
        params['shape'] = shape
        params['assemblyParams'] = assembly_params.to_list()
        params['atomRadiusMultiplier'] = membrane.atom_radius_multiplier
        params['loadBonds'] = membrane.load_bonds
        params['loadNonPolymerChemicals'] = membrane.load_non_polymer_chemicals
        params['representation'] = membrane.representation
        params['chainIds'] = membrane.chain_ids
        params['recenter'] = membrane.recenter
        params['occurrences'] = membrane.number_of_instances
        params['randomSeed'] = random_seed
        params['locationCutoffAngle'] = membrane.location_cutoff_angle
        params['positionRandomizationType'] = position_randomization_type
        params['orientation'] = membrane.orientation.to_list()
        result = self._client.rockets_client.request(method='add-membrane', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def add_protein(self, name, protein, representation=REPRESENTATION_ATOMS, conformation_index=0,
                    atom_radius_multiplier=1.0, position=Vector3(), orientation=Quaternion()):
        """
        Add a protein to the scene

        @param name: Name of the protein
        @param protein: Description of the protein
        @param representation: Representation of the protein (Atoms, atoms and sticks, etc)
        @param conformation_index: Index of the source to be used for the protein conformation
        @param atom_radius_multiplier: Multiplies atom radius by the specified value
        @param position: Position of the protein in the scene
        @param orientation: Orientation of the protein in the scene
        @return: Result of the call to the BioExplorer backend
        """
        assert conformation_index < len(protein.sources)
        assert isinstance(protein, Protein)

        self.remove_assembly(name)
        self.add_assembly(name=name)
        _protein = AssemblyProtein(
            assembly_name=name, name=name,
            source=protein.sources[conformation_index],
            shape=self.ASSEMBLY_SHAPE_PLANAR, load_hydrogen=protein.load_hydrogen,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=protein.load_bonds,
            load_non_polymer_chemicals=protein.load_non_polymer_chemicals,
            representation=representation, position=position,
            orientation=orientation)
        return self.add_assembly_protein(_protein)

    def add_assembly_protein(self, protein):
        """
        Add an protein to an assembly

        @param protein: Description of the protein
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(protein, AssemblyProtein)

        params = dict()
        params['assemblyName'] = protein.assembly_name
        params['name'] = protein.name
        params['contents'] = protein.contents
        params['shape'] = protein.shape
        params['assemblyParams'] = protein.assembly_params.to_list()
        params['atomRadiusMultiplier'] = protein.atom_radius_multiplier
        params['loadBonds'] = protein.load_bonds
        params['loadNonPolymerChemicals'] = protein.load_non_polymer_chemicals
        params['loadHydrogen'] = protein.load_hydrogen
        params['representation'] = protein.representation
        params['chainIds'] = protein.chain_ids
        params['recenter'] = protein.recenter
        params['occurrences'] = protein.occurrences
        params['allowedOccurrences'] = protein.allowed_occurrences
        params['randomSeed'] = protein.random_seed
        params['locationCutoffAngle'] = protein.location_cutoff_angle
        params['positionRandomizationType'] = protein.position_randomization_type
        params['position'] = protein.position.to_list()
        params['orientation'] = protein.orientation.to_list()
        result = self._client.rockets_client.request(method='add-protein', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def add_mesh(self, name, mesh, position=Vector3(), orientation=Quaternion(), scale=Vector3()):
        """
        Add a mesh to the scene
        @param name: Name of the mesh in the scene
        @param mesh: Description of the mesh
        @param position: Position of the mesh in the scene
        @param orientation: Orientation of the mesh in the scene
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(mesh, Mesh)

        self.remove_assembly(name)
        self.add_assembly(name=name)
        _mesh = AssemblyMesh(
            assembly_name=name, name=name, mesh_source=mesh.mesh_source, protein_source=mesh.protein_source,
            density=mesh.density, surface_offset=mesh.surface_offset,
            atom_radius_multiplier=mesh.atom_radius_multiplier, representation=mesh.representation,
            random_seed=mesh.random_seed, position=position, orientation=orientation, scale=scale)
        return self.add_assembly_mesh(_mesh)

    def add_assembly_mesh(self, mesh):
        """
        Add an mesh to an assembly

        @param mesh: Description of the mesh
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(mesh, AssemblyMesh)

        params = dict()
        params['assemblyName'] = mesh.assembly_name
        params['name'] = mesh.name
        params['meshContents'] = mesh.mesh_contents
        params['proteinContents'] = mesh.protein_contents
        params['recenter'] = mesh.recenter
        params['density'] = mesh.density
        params['surfaceOffset'] = mesh.surface_offset
        params['atomRadiusMultiplier'] = mesh.atom_radius_multiplier
        params['representation'] = mesh.representation
        params['randomSeed'] = mesh.random_seed
        params['position'] = mesh.position.to_list()
        params['orientation'] = mesh.orientation.to_list()
        params['scale'] = mesh.scale.to_list()
        result = self._client.rockets_client.request(method='add-mesh', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def add_glycans(self, glycans):
        """
        Add glycans to an protein in an assembly

        @param glycans: Description of the glycans
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(glycans, Sugars)

        params = dict()
        params['assemblyName'] = glycans.assembly_name
        params['name'] = glycans.name
        params['contents'] = glycans.contents
        params['proteinName'] = glycans.protein_name
        params['atomRadiusMultiplier'] = glycans.atom_radius_multiplier
        params['addSticks'] = glycans.add_sticks
        params['recenter'] = glycans.recenter
        params['chainIds'] = glycans.chain_ids
        params['siteIndices'] = glycans.site_indices
        params['allowedOccurrences'] = glycans.allowed_occurrences
        params['orientation'] = glycans.orientation.to_list()
        result = self._client.rockets_client.request(method='add-glycans', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def add_multiple_glycans(
            self, assembly_name, glycan_type, protein_name, paths, chain_ids=list(), indices=list(),
            allowed_occurrences=list(), index_offset=0, add_sticks=True, atom_radius_multiplier=1.0):
        """
        Add glycans to a protein in a assembly

        @param assembly_name: Name of the assembly
        @param glycan_type: Type of glycans
        @param protein_name: Name of the protein
        @param paths: Paths to PDB files with various glycan structures
        @param chain_ids: IDs of the chains to be loaded
        @param indices: Indices of the glycosylation sites where glycans should be added
        @param allowed_occurrences: List of occurrences of the protein in the assembly, where glycans should be added
        @param index_offset: Offset applied to the indices. This is because not all amino acid sequences start at the
                             same index in the description of the protein in the PDB file.
        @param add_sticks: Defines if sticks should be added between atoms
        @param atom_radius_multiplier: Multiplies atom radius by the specified value
        """
        assert isinstance(chain_ids, list)
        assert isinstance(indices, list)
        assert isinstance(allowed_occurrences, list)

        for path_index in range(len(paths)):
            path = paths[path_index]
            site_indices = list()
            if indices is not None:
                for index in range(len(indices)):
                    if index % len(paths) == path_index:
                        site_indices.append(indices[index] + index_offset)

            occurrences = list()
            if allowed_occurrences is not None:
                occurrences = allowed_occurrences

            _glycans = Sugars(
                assembly_name=assembly_name,
                name=assembly_name + '_' + protein_name + '_' + glycan_type + '_' + str(path_index), source=path,
                protein_name=assembly_name + '_' + protein_name, chain_ids=chain_ids,
                atom_radius_multiplier=atom_radius_multiplier, add_sticks=add_sticks, recenter=True,
                site_indices=site_indices, allowed_occurrences=occurrences, orientation=Quaternion())
            self.add_glycans(_glycans)

    def add_sugars(self, sugars):
        """
        Add sugars to a protein in an assembly
        @param sugars: Description of the sugars
        @return: Result of the call to the BioExplorer backend
        """
        assert isinstance(sugars, Sugars)

        params = dict()
        params['assemblyName'] = sugars.assembly_name
        params['name'] = sugars.name
        params['contents'] = sugars.contents
        params['proteinName'] = sugars.protein_name
        params['atomRadiusMultiplier'] = sugars.atom_radius_multiplier
        params['addSticks'] = sugars.add_sticks
        params['recenter'] = sugars.recenter
        params['chainIds'] = sugars.chain_ids
        params['siteIndices'] = sugars.site_indices
        params['allowedOccurrences'] = sugars.allowed_occurrences
        params['orientation'] = sugars.orientation.to_list()
        result = self._client.rockets_client.request(method='add-sugars', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)
        return result

    def set_image_quality(self, image_quality):
        """
        Set image quality using hard-coded presets

        @param image_quality: Quality of the image (IMAGE_QUALITY_LOW or IMAGE_QUALITY_HIGH)
        @return: Result of the call to the BioExplorer backend
        """
        if image_quality == self.IMAGE_QUALITY_HIGH:
            self._client.set_renderer(background_color=[96 / 255, 125 / 255, 139 / 255], current='bio_explorer',
                                      samples_per_pixel=1, subsampling=4, max_accum_frames=128)
            params = self._client.BioExplorerRendererParams()
            params.gi_samples = 3
            params.gi_weight = 0.25
            params.gi_distance = 20
            params.shadows = 1.0
            params.soft_shadows = 1.0
            params.fog_start = 1300
            params.fog_thickness = 1300
            params.max_bounces = 3
            return self._client.set_renderer_params(params)
        else:
            return self._client.set_renderer(
                background_color=Vector3(), current='basic', samples_per_pixel=1, subsampling=4, max_accum_frames=16)

    def get_material_ids(self, model_id):
        if self._client is None:
            return

        params = dict()
        params['modelId'] = model_id
        return self._client.rockets_client.request('get-material-ids', params)

    def set_materials(self, model_ids, material_ids, diffuse_colors, specular_colors,
                      specular_exponents=list(), opacities=list(), reflection_indices=list(),
                      refraction_indices=list(), glossinesses=list(), shading_modes=list(), emissions=list(),
                      user_parameters=list()):
        """
        Set a list of material on a specified list of models

        :param int model_ids: IDs of the models
        :param int material_ids: IDs of the materials
        :param list diffuse_colors: List of diffuse colors (3 values between 0 and 1)
        :param list specular_colors: List of specular colors (3 values between 0 and 1)
        :param list specular_exponents: List of diffuse exponents
        :param list opacities: List of opacities
        :param list reflection_indices: List of reflection indices (value between 0 and 1)
        :param list refraction_indices: List of refraction indices
        :param list glossinesses: List of glossinesses (value between 0 and 1)
        :param list shading_modes: List of shading modes (SHADING_MODE_NONE, SHADING_MODE_BASIC, SHADING_MODE_DIFFUSE,
        SHADING_MODE_ELECTRON, SHADING_MODE_CARTOON, SHADING_MODE_ELECTRON_TRANSPARENCY, SHADING_MODE_PERLIN or
        SHADING_MODE_DIFFUSE_TRANSPARENCY)
        :param list emissions: List of light emission intensities
        :param list user_parameters: List of convenience parameter used by some of the shaders
        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        params = dict()
        params['modelIds'] = model_ids
        params['materialIds'] = material_ids

        dc = list()
        for diffuse in diffuse_colors:
            for k in range(3):
                dc.append(diffuse[k])
        params['diffuseColors'] = dc

        sc = list()
        for specular in specular_colors:
            for k in range(3):
                sc.append(specular[k])
        params['specularColors'] = sc

        params['specularExponents'] = specular_exponents
        params['reflectionIndices'] = reflection_indices
        params['opacities'] = opacities
        params['refractionIndices'] = refraction_indices
        params['emissions'] = emissions
        params['glossinesses'] = glossinesses
        params['shadingModes'] = shading_modes
        params['userParameters'] = user_parameters
        return self._client.rockets_client.request("set-materials", params=params)

    def set_materials_from_palette(self, model_ids, material_ids, palette, shading_mode, specular_exponent,
                                   user_parameter=1.0, glossiness=1.0, emission=0.0, opacity=1.0, reflection_index=0.0,
                                   refraction_index=1.0):
        colors = list()
        shading_modes = list()
        user_parameters = list()
        glossinesses = list()
        specular_exponents = list()
        emissions = list()
        opacities = list()
        reflection_indices = list()
        refraction_indices = list()
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
        self.set_materials(
            model_ids=model_ids, material_ids=material_ids, diffuse_colors=colors, specular_colors=colors,
            specular_exponents=specular_exponents, user_parameters=user_parameters, glossinesses=glossinesses,
            shading_modes=shading_modes, emissions=emissions, opacities=opacities,
            reflection_indices=reflection_indices, refraction_indices=refraction_indices)

    def apply_default_color_scheme(self, shading_mode, user_parameter=0.03, specular_exponent=5.0, glossiness=0.5):
        from ipywidgets import IntProgress
        from IPython.display import display

        if self._url is not None:
            ''' Refresh connection to Brayns to make sure we get all current models '''
            self._client = Client(self._url)

        glycans_colors = [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0.2, 0.2, 0.7]]

        progress = IntProgress(value=0, min=0, max=len(self._client.scene.models), orientation='horizontal')
        display(progress)

        i = 0
        for model in self._client.scene.models:
            model_id = model['id']
            model_name = model['name']

            material_ids = self.get_material_ids(model_id)['ids']
            nb_materials = len(material_ids)

            if self.NAME_MEMBRANE in model_name:
                palette = sns.color_palette('gist_heat', nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id], material_ids=material_ids, palette=palette,
                    shading_mode=shading_mode, user_parameter=user_parameter, glossiness=glossiness,
                    specular_exponent=specular_exponent)

            if self.NAME_TRANS_MEMBRANE in model_name:
                palette = list()
                for p in range(nb_materials):
                    palette.append([1, 1, 1])
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                opacity=0.5, reflection_index=0.0, refraction_index=1.1,
                                                shading_mode=self.SHADING_MODE_DIFFUSE,
                                                specular_exponent=30)

            if self.NAME_PROTEIN_S_CLOSED in model_name or self.NAME_PROTEIN_S_OPEN in model_name or \
                    self.NAME_PROTEIN_E in model_name or self.NAME_PROTEIN_M in model_name:
                palette = sns.color_palette('Greens', nb_materials)
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_GLUCOSE in model_name:
                palette = sns.color_palette('Blues', nb_materials)
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_LACTOFERRIN in model_name:
                palette = sns.color_palette('afmhot', nb_materials)
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_DEFENSIN in model_name:
                palette = sns.color_palette('plasma_r', nb_materials)
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_GLYCAN_HYBRID in model_name:
                palette = list()
                for p in range(nb_materials):
                    palette.append(glycans_colors[2])
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_GLYCAN_COMPLEX in model_name:
                palette = list()
                for p in range(nb_materials):
                    palette.append(glycans_colors[1])
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_GLYCAN_HIGH_MANNOSE in model_name:
                palette = list()
                for p in range(nb_materials):
                    palette.append(glycans_colors[0])
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_GLYCAN_O_GLYCAN in model_name:
                palette = list()
                for p in range(nb_materials):
                    palette.append(glycans_colors[3])
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if 'Lymphocyte' in model_name:
                palette = list()
                for p in range(nb_materials):
                    palette.append([1, 1, 1])
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            if self.NAME_SURFACTANT_HEAD in model_name or \
                    self.NAME_COLLAGEN in model_name:
                palette = sns.color_palette('OrRd_r', nb_materials)
                emission = 0
                if self.NAME_COLLAGEN in model_name:
                    emission = 0.1
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                shading_mode=shading_mode, emission=emission,
                                                user_parameter=user_parameter, glossiness=glossiness,
                                                specular_exponent=specular_exponent)

            i += 1
            progress.value = i

    def build_fields(self, voxel_size):
        """
        Build fields acceleration structures and creates according data handler

        :param float voxel_size: Voxel size
        :param str filename: Octree filename
        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        params = dict()
        params['voxelSize'] = voxel_size
        return self._client.rockets_client.request('build-fields', params)

    def import_fields_from_file(self, filename):
        """
        Imports fields acceleration structures from file

        :param str filename: Octree filename
        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        params = dict()
        params['filename'] = filename
        return self._client.rockets_client.request('import-fields-from-file', params)

    def export_fields_to_file(self, model_id, filename):
        """
        Exports fields acceleration structures to file

        :param int model_id: id of the model containing the fields
        :param str filename: Octree filename
        :return: Result of the request submission
        :rtype: str
        """

        assert isinstance(model_id, int)
        if self._client is None:
            return

        params = dict()
        params['modelId'] = model_id
        params['filename'] = filename
        return self._client.rockets_client.request('export-fields-to-file', params)

    def add_grid(self, min_value, max_value, interval, radius=1.0, opacity=0.5, show_axis=True, colored=True,
                 position=Vector3()):
        """
        Adds a reference grid to the scene

        :param float min_value: Minimum value for all axis
        :param float max_value: Maximum value for all axis
        :param float interval: Interval at which lines should appear on the grid
        :param float radius: Radius of grid lines
        :param float opacity: Opacity of the grid
        :param bool show_axis: Shows axis if True
        :param bool colored: Colors the grid it True. X in red, Y in green, Z in blue
        :param Vector3 position: Position of the grid
        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        assert isinstance(position, Vector3)
        params = dict()
        params['minValue'] = min_value
        params['maxValue'] = max_value
        params['steps'] = interval
        params['radius'] = radius
        params['planeOpacity'] = opacity
        params['showAxis'] = show_axis
        params['useColors'] = colored
        params['position'] = position.to_list()
        return self._client.rockets_client.request('add-grid', params)


''' Internal classes '''


class AssemblyProtein:

    def __init__(self, assembly_name, name, source, assembly_params=Vector2(),
                 shape=BioExplorer.ASSEMBLY_SHAPE_PLANAR, atom_radius_multiplier=1.0,
                 load_bonds=False, representation=BioExplorer.REPRESENTATION_ATOMS,
                 load_non_polymer_chemicals=False, load_hydrogen=True,
                 chain_ids=list(), recenter=True, occurrences=1, random_seed=0,
                 location_cutoff_angle=0.0,
                 position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_CIRCULAR,
                 position=Vector3(), orientation=Quaternion(),
                 allowed_occurrences=list()):
        self.assembly_name = assembly_name
        self.name = name
        self.contents = ''.join(open(source).readlines())
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
        self.location_cutoff_angle = location_cutoff_angle
        self.position_randomization_type = position_randomization_type
        self.position = position
        self.orientation = orientation


class AssemblyMesh:

    def __init__(self, assembly_name, name, mesh_source, protein_source, recenter=True, density=1, surface_offset=0,
                 atom_radius_multiplier=1.0, representation=BioExplorer.REPRESENTATION_ATOMS, random_seed=0,
                 position=Vector3(), orientation=Quaternion(), scale=Vector3()):
        assert isinstance(position, Vector3)
        assert isinstance(orientation, Quaternion)
        assert isinstance(scale, Vector3)
        self.assembly_name = assembly_name
        self.name = name
        self.mesh_contents = ''.join(open(mesh_source).readlines())
        self.protein_contents = ''.join(open(protein_source).readlines())
        self.recenter = recenter
        self.density = density
        self.surface_offset = surface_offset
        self.atom_radius_multiplier = atom_radius_multiplier
        self.representation = representation
        self.random_seed = random_seed
        self.position = position
        self.orientation = orientation
        self.scale = scale


''' External classes '''


class Membrane:

    def __init__(self, sources,
                 atom_radius_multiplier=1.0, load_bonds=False, representation=BioExplorer.REPRESENTATION_ATOMS,
                 load_non_polymer_chemicals=False, chain_ids=list(), recenter=True, number_of_instances=1,
                 location_cutoff_angle=0.0,
                 position=Vector3(), orientation=Quaternion()):
        """
        A membrane is an assembly of proteins with a given size and shape

        @param sources: Full paths of the PDB files containing the building blocks of the membrane
        @param atom_radius_multiplier: Multiplies atom radius by the specified value
        @param load_bonds: Defines if bonds should be loaded
        @param representation: Representation of the protein (Atoms, atoms and sticks, etc)
        @param load_non_polymer_chemicals: Defines if non-polymer chemical should be loaded
        @param chain_ids: IDs of the protein chains to be loaded
        @param recenter: Defines if proteins should be recentered when loaded from PDB files
        @param number_of_instances: Number of instances of proteins defining the membrane
        @param location_cutoff_angle:
        @param position: Position of the membrane in the assembly
        @param orientation: Orientation of the membrane in the assembly
        """
        self.sources = sources
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.representation = representation
        self.chain_ids = chain_ids
        self.recenter = recenter
        self.number_of_instances = number_of_instances
        self.location_cutoff_angle = location_cutoff_angle
        self.position = position
        self.orientation = orientation


class Sugars:

    def __init__(self, assembly_name, name, source, protein_name,
                 atom_radius_multiplier=1.0, add_sticks=False,
                 recenter=True, chain_ids=list(), site_indices=list(),
                 allowed_occurrences=list(), orientation=Quaternion()):
        assert isinstance(chain_ids, list)
        assert isinstance(site_indices, list)
        assert isinstance(allowed_occurrences, list)
        assert isinstance(orientation, Quaternion)
        self.assembly_name = assembly_name
        self.name = name
        self.contents = ''.join(open(source).readlines())
        self.protein_name = protein_name
        self.atom_radius_multiplier = atom_radius_multiplier
        self.add_sticks = add_sticks
        self.recenter = recenter
        self.chain_ids = chain_ids
        self.site_indices = site_indices
        self.allowed_occurrences = allowed_occurrences
        self.orientation = orientation


class RNASequence:

    def __init__(self, source, shape, assembly_params,
                 t_range=Vector2(), shape_params=Vector3()):
        assert isinstance(t_range, Vector2)
        assert isinstance(shape_params, Vector3)
        assert isinstance(assembly_params, Vector2)
        self.source = source
        self.shape = shape
        self.assembly_params = assembly_params
        self.t_range = t_range
        self.shape_params = shape_params


class Surfactant:

    def __init__(self, name, surfactant_protein, head_source, branch_source):
        self.surfactant_protein = surfactant_protein
        self.name = name
        self.head_source = head_source
        self.branch_source = branch_source


class Cell:

    def __init__(self, name, size, shape, membrane, receptor):
        assert isinstance(size, Vector2)
        assert isinstance(membrane, Membrane)
        assert isinstance(receptor, Protein)
        self.name = name
        self.shape = shape
        self.size = size
        self.membrane = membrane
        self.receptor = receptor


class Volume:

    def __init__(self, name, size, protein):
        assert isinstance(size, Vector2)
        assert isinstance(protein, Protein)
        self.name = name
        self.shape = BioExplorer.ASSEMBLY_SHAPE_CUBIC
        self.size = size
        self.protein = protein


class Protein:

    def __init__(self, sources, number_of_instances=1, assembly_params=Vector2(),
                 load_bonds=False, load_hydrogen=False,
                 load_non_polymer_chemicals=False, cutoff_angle=0.0, position=Vector3(),
                 orientation=Quaternion(), instance_indices=list()):
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert isinstance(position, Vector3)
        assert isinstance(orientation, Quaternion)
        assert isinstance(instance_indices, list)
        self.sources = sources
        self.number_of_instances = number_of_instances
        self.assembly_params = assembly_params
        self.load_bonds = load_bonds
        self.load_hydrogen = load_hydrogen
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.cutoff_angle = cutoff_angle
        self.position = position
        self.orientation = orientation
        self.instance_indices = instance_indices


class Mesh:

    def __init__(self, mesh_source, protein_source, density=1, surface_offset=0.0, atom_radius_multiplier=1.0,
                 representation=BioExplorer.REPRESENTATION_ATOMS, random_seed=0, recenter=True, position=Vector3(),
                 orientation=Quaternion(), scale=Vector3()):
        assert isinstance(position, Vector3)
        assert isinstance(orientation, Quaternion)
        assert isinstance(scale, Vector3)
        self.mesh_source = mesh_source
        self.protein_source = protein_source
        self.density = density
        self.surface_offset = surface_offset
        self.atom_radius_multiplier = atom_radius_multiplier
        self.representation = representation
        self.recenter = recenter
        self.random_seed = random_seed
        self.position = position
        self.orientation = orientation
        self.scale = scale


class Virus:

    def __init__(self, name, assembly_params, protein_s=None, protein_e=None, protein_m=None,
                 membrane=None, rna_sequence=None):
        assert isinstance(assembly_params, Vector2)
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
        self.name = name
        self.protein_s = protein_s
        self.protein_e = protein_e
        self.protein_m = protein_m
        self.membrane = membrane
        self.rna_sequence = rna_sequence
        self.assembly_params = assembly_params
