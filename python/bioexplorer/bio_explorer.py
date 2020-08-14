#!/usr/bin/env python
# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, Cyrille Favreau <cyrille.favreau@epfl.ch>
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

    NAME_SURFACTANT_HEAD = 'Head'
    NAME_COLLAGEN = 'Collagen'
    NAME_GLUCOSE = 'Glucose'

    NAME_LACTOFERRIN = 'Lactoferrin'
    NAME_DEFENSIN = 'Defensin'

    NAME_GLYCAN_HIGH_MANNOSE = 'High-mannose'
    NAME_GLYCAN_O_GLYCAN = 'O-glycan'
    NAME_GLYCAN_HYBRID = 'Hybrid'
    NAME_GLYCAN_COMPLEX = 'Complex'

    SURFACTANT_PROTEIN_A = 0
    SURFACTANT_PROTEIN_D = 1

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
        """Return a pretty-print of the class"""
        return "Blue Brain BioExplorer"

    def get_client(self):
        return self._client

    def version(self):
        if self._client is None:
            return __version__

        result = self._client.rockets_client.request(method='version')
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result['contents']

    def reset(self):
        if self._client is None:
            return

        ids = list()
        for model in self._client.scene.models:
            ids.append(model['id'])
        self._client.remove_model(array=ids)

    def export_to_cache(self, filename):
        if self._client is None:
            return

        params = dict()
        params['filename'] = filename
        result = self._client.rockets_client.request(method='export-to-cache', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])

    def export_to_xyzr(self, filename):
        if self._client is None:
            return

        params = dict()
        params['filename'] = filename
        result = self._client.rockets_client.request(method='export-to-xyzr', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])

    def remove_assembly(self, name):
        if self._client is None:
            return

        params = dict()
        params['name'] = name
        params['position'] = Vector3().to_list()
        params['clippingPlanes'] = list()
        result = self._client.rockets_client.request(method='remove-assembly', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])

    def add_virus(self, virus,
                  atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                  clipping_planes=list(), delay_between_additions=0):

        shape = self.ASSEMBLY_SHAPE_SPHERICAL
        _protein_s = virus.protein_s

        import time
        self.remove_assembly(virus.name)
        time.sleep(delay_between_additions)
        self.add_assembly(
            name=virus.name, position=virus.position,
            clipping_planes=clipping_planes)

        if virus.protein_s is not None:
            radius = virus.protein_s.assembly_params.x + virus.assembly_params.x
            _protein_s_open = Protein(
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
                position=_protein_s.position,
                orientation=_protein_s.orientation,
                allowed_occurrences=_protein_s.instance_indices[0])
            time.sleep(delay_between_additions)
            self.add_protein(_protein_s_open)

            _protein_s_closed = Protein(
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
                position=_protein_s.position,
                orientation=_protein_s.orientation,
                allowed_occurrences=_protein_s.instance_indices[1])
            time.sleep(delay_between_additions)
            self.add_protein(_protein_s_closed)

            palette = 'Greens'
            for protein_name in [self.NAME_PROTEIN_S_OPEN, self.NAME_PROTEIN_S_CLOSED]:
                self.set_protein_color_scheme(
                    assembly_name=virus.name,
                    protein_name=virus.name + '_' + protein_name,
                    color_scheme=self.COLOR_SCHEME_CHAINS,
                    palette_name=palette, palette_size=7)

        if virus.protein_m is not None:
            radius = virus.protein_m.assembly_params.x + virus.assembly_params.x
            _protein_m = Protein(
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
                position=virus.protein_m.position, orientation=virus.protein_m.orientation)

            time.sleep(delay_between_additions)
            self.add_protein(_protein_m)

            palette = 'Greens'
            self.set_protein_color_scheme(
                assembly_name=virus.name,
                protein_name=virus.name + '_' + self.NAME_PROTEIN_M,
                color_scheme=self.COLOR_SCHEME_CHAINS,
                palette_name=palette, palette_size=30)

        if virus.protein_e is not None:
            radius = virus.protein_e.assembly_params.x + virus.assembly_params.x
            _protein_e = Protein(
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
                position=virus.protein_e.position, orientation=virus.protein_e.orientation)
            time.sleep(delay_between_additions)
            self.add_protein(_protein_e)

            palette = 'Greens'
            self.set_protein_color_scheme(
                assembly_name=virus.name,
                protein_name=virus.name + '_' + self.NAME_PROTEIN_E,
                color_scheme=self.COLOR_SCHEME_CHAINS,
                palette_name=palette, palette_size=30)

        if virus.membrane is not None:
            contents = list()
            for path in virus.membrane.sources:
                contents.append(''.join(open(path).readlines()))
            _membrane = MembraneAssembly(
                assembly_name=virus.name, name=virus.name + '_' + self.NAME_MEMBRANE,
                shape=shape, contents=contents,
                occurrences=virus.membrane.number_of_instances,
                load_non_polymer_chemicals=True,
                assembly_params=virus.assembly_params,
                atom_radius_multiplier=atom_radius_multiplier,
                load_bonds=False,
                representation=representation,
                random_seed=4,
                location_cutoff_angle=0.0,
                position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL)

            time.sleep(delay_between_additions)
            self.add_membrane(_membrane)

            palette = 'inferno'
            for i in range(len(virus.membrane.sources)):
                self.set_protein_color_scheme(
                    assembly_name=virus.name,
                    protein_name=virus.name + '_' + self.NAME_MEMBRANE + '_' + str(i),
                    color_scheme=self.COLOR_SCHEME_CHAINS,
                    palette_name=palette, palette_size=30)

        if virus.transmembrane is not None:
            time.sleep(delay_between_additions)
            _trans_membrane = Mesh(
                assembly_name=virus.name,
                name=virus.name + '_' + self.NAME_TRANS_MEMBRANE,
                contents=''.join(open(virus.transmembrane.source).readlines()),
                shape=shape,
                occurrences=virus.transmembrane.number_of_instances,
                assembly_params=virus.transmembrane.assembly_params,
                recenter=True, random_seed=1, position_randomization_type=0,
                location_cutoff_angle=virus.transmembrane.cutoff_angle,
                position=virus.transmembrane.position,
                orientation=virus.transmembrane.orientation)
            self.add_mesh(_trans_membrane)

        if virus.rna_sequence is not None:
            time.sleep(delay_between_additions)
            _rna_sequence = RNASequence(
                assembly_name=virus.name,
                name=virus.name + '_' + self.NAME_RNA_SEQUENCE,
                contents=''.join(open(virus.rna_sequence.source).readlines()),
                assembly_params=virus.rna_sequence.assembly_params,
                radius=virus.rna_sequence.radius,
                t_range=virus.rna_sequence.t_range,
                shape=virus.rna_sequence.shape,
                shape_params=virus.rna_sequence.shape_params)
            self.add_rna_sequence(_rna_sequence)

    def add_cell(self, cell, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                 position=Vector3(), clipping_planes=list(), delay_between_additions=0):

        _receptor = Protein(
            assembly_name=cell.name,
            name=cell.name + '_' + self.NAME_RECEPTOR, shape=cell.shape,
            source=cell.receptor.source,
            occurrences=cell.receptor.number_of_instances,
            assembly_params=cell.size,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=True, representation=representation, random_seed=1,
            location_cutoff_angle=0.0,
            position=cell.receptor.position,
            orientation=cell.receptor.orientation)

        contents = list()
        for path in cell.membrane.sources:
            contents.append(''.join(open(path).readlines()))
        _membrane = MembraneAssembly(
            assembly_name=cell.name, name=cell.name + '_' + self.NAME_MEMBRANE,
            shape=cell.shape, contents=contents,
            occurrences=cell.membrane.number_of_instances,
            load_non_polymer_chemicals=True, assembly_params=cell.size,
            atom_radius_multiplier=atom_radius_multiplier,
            load_bonds=False, representation=representation, random_seed=40,
            location_cutoff_angle=0.0,
            position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL)

        import time
        self.remove_assembly(cell.name)
        time.sleep(delay_between_additions)
        self.add_assembly(name=cell.name, position=position, clipping_planes=clipping_planes)
        time.sleep(delay_between_additions)
        self.add_protein(_receptor)
        time.sleep(delay_between_additions)
        self.add_membrane(_membrane)

        palette = 'OrRd_r'
        self.set_protein_color_scheme(
            assembly_name=cell.name,
            protein_name=cell.name + '_' + self.NAME_RECEPTOR,
            color_scheme=self.COLOR_SCHEME_CHAINS,
            palette_name=palette, palette_size=7)

        palette = 'inferno'
        for i in range(len(cell.membrane.sources)):
            self.set_protein_color_scheme(
                assembly_name=cell.name,
                protein_name=cell.name + '_' + self.NAME_MEMBRANE + '_' + str(i),
                color_scheme=self.COLOR_SCHEME_CHAINS,
                palette_name=palette, palette_size=7)

    def add_assembly(self, name, position=Vector3(), clipping_planes=list()):
        if self._client is None:
            return

        clipping_planes_values = list()
        for plane in clipping_planes:
            for i in range(4):
                clipping_planes_values.append(plane[i])

        params = dict()
        params['name'] = name
        params['position'] = position.to_list()
        params['clippingPlanes'] = clipping_planes_values
        result = self._client.rockets_client.request(method='add-assembly', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def apply_transformations(self, protein_, transformations):
        if self._client is None:
            return

        transformations_as_floats = list()
        for transformation in transformations:
            for i in range(3):
                transformations_as_floats.append(transformation.translation[i])
            for i in range(3):
                transformations_as_floats.append(transformation.rotation_center[i])
            for i in range(4):
                transformations_as_floats.append(transformation.rotation[i])
            for i in range(3):
                transformations_as_floats.append(transformation.scale[i])
        params = dict()
        params['assemblyName'] = protein.assembly_name
        params['name'] = protein.name
        params['transformations'] = transformations_as_floats
        result = self._client.rockets_client.request(method='apply-transformations', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def set_protein_color_scheme(self, assembly_name, protein_name, color_scheme, palette_name='', palette_size=256,
                                 palette=list(), chain_ids=list()):
        if self._client is None:
            return

        p = list()
        if len(palette) == 0 and palette_name != '':
            palette = sns.color_palette(palette_name, palette_size)

        for color in palette:
            for i in range(3):
                p.append(color[i])

        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = protein_name
        params['colorScheme'] = color_scheme
        params['palette'] = p
        params['chainIds'] = chain_ids
        result = self._client.rockets_client.request(method='set-protein-color-scheme', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def set_protein_amino_acid_sequence_as_string(self, assembly_name, protein_name, amino_acid_sequence):
        if self._client is None:
            return

        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = protein_name
        params['sequence'] = amino_acid_sequence
        result = self._client.rockets_client.request(method='set-protein-amino-acid-sequence-as-string', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def set_protein_amino_acid_sequence_as_range(self, assembly_name, protein_name, amino_acid_range):
        if self._client is None:
            return

        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = protein_name
        params['range'] = amino_acid_range
        result = self._client.rockets_client.request(method='set-protein-amino-acid-sequence-as-range', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def show_amino_acid_on_protein(self, assembly_name, protein_name, sequence_id=0, palette_name='Set1',
                                   palette_size=2):
        from ipywidgets import IntRangeSlider, Label
        from IPython.display import display

        sequences = self.get_protein_amino_acid_sequences(assembly_name, protein_name)
        if sequence_id >= len(sequences):
            raise RuntimeError('Invalid sequence Id')
        sequence_as_list = sequences[0].split(',')

        value_range = [int(sequence_as_list[0]), int(sequence_as_list[1])]
        irs = IntRangeSlider(value=[value_range[0], value_range[1]], min=value_range[0], max=value_range[1])
        lbl = Label(value="AA sequence")

        def update_slider(v):
            self.set_protein_amino_acid_sequence_as_range(assembly_name, protein_name, v['new'])
            self.set_protein_color_scheme(assembly_name, protein_name, self.COLOR_SCHEME_AMINO_ACID_SEQUENCE,
                                          palette_name, palette_size)
            lbl.value = sequence_as_list[2][v['new'][0] - value_range[0]:v['new'][1] - value_range[0]]

        irs.observe(update_slider, 'value')
        display(irs)
        display(lbl)

    def get_protein_amino_acid_information(self, assembly_name, protein_name):
        if self._client is None:
            return

        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = protein_name
        result = self._client.rockets_client.request(method='get-protein-amino-acid-information', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result['contents'].split()

    def add_rna_sequence(self, rna_sequence):
        if self._client is None:
            return

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
        params['assemblyName'] = rna_sequence.assembly_name
        params['name'] = rna_sequence.name
        params['contents'] = rna_sequence.contents
        params['shape'] = rna_sequence.shape
        params['assemblyParams'] = rna_sequence.assembly_params.to_list()
        params['range'] = t_range.to_list()
        params['params'] = shape_params.to_list()
        result = self._client.rockets_client.request(method='add-rna-sequence', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_membrane(self, membrane):
        if self._client is None:
            return

        params = dict()
        params['assemblyName'] = membrane.assembly_name
        params['name'] = membrane.name
        params['content1'] = membrane.content1
        params['content2'] = membrane.content2
        params['content3'] = membrane.content3
        params['content4'] = membrane.content4
        params['shape'] = membrane.shape
        params['assemblyParams'] = membrane.assembly_params.to_list()
        params['atomRadiusMultiplier'] = membrane.atom_radius_multiplier
        params['loadBonds'] = membrane.load_bonds
        params['loadNonPolymerChemicals'] = membrane.load_non_polymer_chemicals
        params['representation'] = membrane.representation
        params['chainIds'] = membrane.chain_ids
        params['recenter'] = membrane.recenter
        params['occurrences'] = membrane.occurrences
        params['randomSeed'] = membrane.random_seed
        params['locationCutoffAngle'] = membrane.location_cutoff_angle
        params['positionRandomizationType'] = membrane.position_randomization_type
        params['orientation'] = membrane.orientation.to_list()
        result = self._client.rockets_client.request(method='add-membrane', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_protein(self, protein):
        if self._client is None:
            return

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

    def add_mesh(self, mesh):
        if self._client is None:
            return

        params = dict()
        params['assemblyName'] = mesh.assembly_name
        params['name'] = mesh.name
        params['contents'] = mesh.contents
        params['shape'] = mesh.shape
        params['assemblyParams'] = mesh.assembly_params.to_list()
        params['recenter'] = mesh.recenter
        params['occurrences'] = mesh.occurrences
        params['randomSeed'] = mesh.random_seed
        params['locationCutoffAngle'] = mesh.location_cutoff_angle
        params['positionRandomizationType'] = mesh.position_randomization_type
        params['position'] = mesh.position.to_list()
        params['orientation'] = mesh.orientation.to_list()
        result = self._client.rockets_client.request(method='add-mesh', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_glycans(self, glycans):
        if self._client is None:
            return

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

    def add_multiple_glycans(
            self, assembly_name, glycan_type, protein_name, paths, chain_ids=list(), indices=list(),
            allowed_occurrences=list(), index_offset=0, add_sticks=False, radius_multiplier=1.0):

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
                assembly_name=assembly_name, name=assembly_name + '_' + protein_name + '_' + glycan_type,
                contents=''.join(open(path).readlines()),
                protein_name=assembly_name + '_' + protein_name, chain_ids=chain_ids,
                atom_radius_multiplier=radius_multiplier,
                add_sticks=add_sticks, recenter=True, site_indices=site_indices,
                allowed_occurrences=occurrences, orientation=Quaternion())
            self.add_glycans(_glycans)

    def add_glucoses(self, sugars_):
        if self._client is None:
            return

        params = dict()
        params['assemblyName'] = sugars_.assembly_name
        params['name'] = sugars_.name
        params['contents'] = sugars_.contents
        params['proteinName'] = sugars_.protein_name
        params['atomRadiusMultiplier'] = sugars_.atom_radius_multiplier
        params['addSticks'] = sugars_.add_sticks
        params['recenter'] = sugars_.recenter
        params['chainIds'] = sugars_.chain_ids
        params['siteIndices'] = sugars_.site_indices
        params['allowedOccurrences'] = sugars_.allowed_occurrences
        params['orientation'] = sugars_.orientation
        result = self._client.rockets_client.request(method='add-glucoses', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_surfactant(self, surfactant, head_pdb_filename, branch_pdb_filename,
                       representation, position=Vector3(), radius_multiplier=1.0,
                       random_seed=0):

        shape = self.ASSEMBLY_SHAPE_SPHERICAL
        nb_branches = 4
        if surfactant == self.SURFACTANT_PROTEIN_A:
            shape = self.ASSEMBLY_SHAPE_FAN
            nb_branches = 6

        nb_collagens = 2
        collagen_size = 16.0

        head_name = surfactant.name + '_' + self.NAME_SURFACTANT_HEAD
        branch_name = surfactant.name + '_' + self.NAME_COLLAGEN + '_'

        protein_sp_d = Protein(
            assembly_name=surfactant.name,
            name=head_name, shape=shape,
            pdb_filename=head_pdb_filename,
            occurrences=nb_branches,
            assembly_params=Vector2(collagen_size * (nb_collagens + 1) - 9.0, 0.0),
            atom_radius_multiplier=radius_multiplier,
            random_seed=random_seed,
            representation=representation,
            orientation=Quaternion(-0.624, -0.417, 0.0, 0.661))

        collagens = list()
        for i in range(nb_collagens):
            collagens.append(
                Protein(
                    assembly_name=surfactant.name, shape=shape,
                    atom_radius_multiplier=radius_multiplier,
                    name=branch_name + str(i),
                    pdb_filename=branch_pdb_filename,
                    occurrences=nb_branches,
                    assembly_params=Vector2(collagen_size * (i + 1) - 7.0, 0.0),
                    random_seed=random_seed,
                    representation=representation))

        self.remove_assembly(surfactant.name)
        self.add_assembly(name=surfactant.name, position=position)
        for collagen in collagens:
            self.add_protein(collagen)
        self.add_protein(protein_sp_d)

        palette = 'Reds'
        self.set_protein_color_scheme(assembly_name=surfactant.name,
                                      protein_name=head_name,
                                      color_scheme=self.COLOR_SCHEME_CHAINS, palette_name=palette, palette_size=5)
        for i in range(nb_collagens):
            self.set_protein_color_scheme(assembly_name=surfactant.name,
                                          protein_name=branch_name + str(i),
                                          color_scheme=self.COLOR_SCHEME_RESIDUES,
                                          palette_name=palette, palette_size=5)

    def set_image_quality(self, image_quality=IMAGE_QUALITY_LOW):
        if self._client is None:
            return

        if image_quality == self.IMAGE_QUALITY_HIGH:
            self._client.set_renderer(
                background_color=[96 / 255, 125 / 255, 139 / 255],
                current='bio_explorer',
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
            self._client.set_renderer_params(params)
        else:
            self._client.set_renderer(
                background_color=Vector3(),
                current='basic',
                samples_per_pixel=1, subsampling=4, max_accum_frames=16)

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

    def add_grid(self, min_value, max_value, interval, radius=1.0, opacity=0.5, show_axis=True, colored=True):
        """
        Adds a reference grid to the scene

        :param float min_value: Minimum value for all axis
        :param float max_value: Maximum value for all axis
        :param float interval: Interval at which lines should appear on the grid
        :param float radius: Radius of grid lines
        :param float opacity: Opacity of the grid
        :param bool show_axis: Shows axis if True
        :param bool colored: Colors the grid it True. X in red, Y in green, Z in blue
        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        params = dict()
        params['minValue'] = min_value
        params['maxValue'] = max_value
        params['steps'] = interval
        params['radius'] = radius
        params['planeOpacity'] = opacity
        params['showAxis'] = show_axis
        params['useColors'] = colored
        return self._client.rockets_client.request('add-grid', params)

    def set_camera(self, origin, direction, up):
        """
        Sets the camera using origin, direction and up vectors

        :param list origin: Origin of the camera
        :param list direction: Direction in which the camera is looking
        :param list up: Up vector
        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        params = dict()
        params['origin'] = origin
        params['direction'] = direction
        params['up'] = up
        return self._client.rockets_client.request('set-odu-camera', params)

    def get_camera(self):
        """
        Gets the origin, direction and up vector of the camera

        :return: A JSon representation of the origin, direction and up vectors
        :rtype: str
        """
        if self._client is None:
            return

        return self._client.rockets_client.request('get-odu-camera')

    def export_frames_to_disk(self, path, animation_frames, camera_definitions, image_format='png',
                              quality=100, samples_per_pixel=1, start_frame=0):
        """
        Exports frames to disk. Frames are named using a 6 digit representation of the frame number

        :param str path: Folder into which frames are exported
        :param list animation_frames: List of animation frames
        :param list camera_definitions: List of camera definitions (origin, direction and up)
        :param str image_format: Image format (the ones supported par Brayns: PNG, JPEG, etc)
        :param float quality: Quality of the exported image (Between 0 and 100)
        :param int samples_per_pixel: Number of samples per pixels
        :param int start_frame: Optional value if the rendering should start at a specific frame.
        This is used to resume the rendering of a previously canceled sequence)
        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        params = dict()
        params['path'] = path
        params['format'] = image_format
        params['quality'] = quality
        params['spp'] = samples_per_pixel
        params['startFrame'] = start_frame
        params['animationInformation'] = animation_frames
        values = list()
        for camera_definition in camera_definitions:
            # Origin
            for i in range(3):
                values.append(camera_definition[0][i])
            # Direction
            for i in range(3):
                values.append(camera_definition[1][i])
            # Up
            for i in range(3):
                values.append(camera_definition[2][i])
            # Aperture radius
            values.append(camera_definition[3])
            # Focus distance
            values.append(camera_definition[4])
        params['cameraInformation'] = values
        return self._client.rockets_client.request('export-frames-to-disk', params)

    def get_export_frames_progress(self):
        """
        Queries the progress of the last export of frames to disk request

        :return: Dictionary with the result: "frameNumber" with the number of
        the last written-to-disk frame, and "done", a boolean flag stating wether
        the exporting is finished or is still in progress
        :rtype: dict
        """
        if self._client is None:
            return

        return self._client.rockets_client.request('get-export-frames-progress')

    def cancel_frames_export(self):
        """
        Cancel the exports of frames to disk

        :return: Result of the request submission
        :rtype: str
        """
        if self._client is None:
            return

        params = dict()
        params['path'] = '/tmp'
        params['format'] = 'png'
        params['quality'] = 100
        params['spp'] = 1
        params['startFrame'] = 0
        params['animationInformation'] = []
        params['cameraInformation'] = []
        return self._client.rockets_client.request('export-frames-to-disk', params)


class Transformation(object):
    def __init__(self, translation, rotation_center, rotation, scale):
        if len(translation) != 3:
            raise RuntimeError('Invalid translation. List of 3 floats expected')
        if len(rotation_center) != 3:
            raise RuntimeError('Invalid rotation center. List of 3 floats expected')
        if len(rotation) != 4:
            raise RuntimeError('Invalid rotation. List of 4 floats expected')
        if len(scale) != 3:
            raise RuntimeError('Invalid scale. List of 3 floats expected')
        self.translation = translation
        self.rotation_center = rotation_center
        self.rotation = rotation
        self.scale = scale


class Mesh(object):

    def __init__(self, assembly_name, name, source, assembly_params,
                 shape=BioExplorer.ASSEMBLY_SHAPE_PLANAR,
                 recenter=True, occurrences=1, random_seed=0,
                 location_cutoff_angle=0.0,
                 position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_CIRCULAR,
                 position=Vector3(), orientation=Quaternion()):
        self.assembly_name = assembly_name
        self.name = name
        self.contents = ''.join(open(source).readlines())
        self.shape = shape
        self.assembly_params = assembly_params
        self.recenter = recenter
        self.occurrences = occurrences
        self.random_seed = random_seed
        self.location_cutoff_angle = location_cutoff_angle
        self.position_randomization_type = position_randomization_type
        self.position = position
        self.orientation = orientation


class MembraneAssembly(object):

    def __init__(self, assembly_name, name, contents, shape, assembly_params,
                 atom_radius_multiplier=1, load_bonds=False, representation=BioExplorer.REPRESENTATION_ATOMS,
                 load_non_polymer_chemicals=False,
                 chain_ids=list(), recenter=True, occurrences=1, random_seed=0,
                 location_cutoff_angle=0.0,
                 position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_CIRCULAR,
                 orientation=Quaternion()):
        self.assembly_name = assembly_name
        self.name = name
        self.content1 = contents[0]
        self.content2 = ''
        if len(contents) > 1:
            self.content2 = contents[1]
        self.content3 = ''
        if len(contents) > 2:
            self.content3 = contents[2]
        self.content4 = ''
        if len(contents) > 3:
            self.content4 = contents[3]

        self.shape = shape
        self.assembly_params = assembly_params
        self.atom_radius_multiplier = atom_radius_multiplier
        self.load_bonds = load_bonds
        self.load_non_polymer_chemicals = load_non_polymer_chemicals
        self.representation = representation
        self.chain_ids = chain_ids
        self.recenter = recenter
        self.occurrences = occurrences
        self.random_seed = random_seed
        self.location_cutoff_angle = location_cutoff_angle
        self.position_randomization_type = position_randomization_type
        self.orientation = orientation


class Protein:

    def __init__(self, assembly_name, name, source, assembly_params=Vector2(),
                 shape=BioExplorer.ASSEMBLY_SHAPE_PLANAR, atom_radius_multiplier=1,
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


class Sugars(object):

    def __init__(self, assembly_name, name, contents, protein_name,
                 atom_radius_multiplier=1.0, add_sticks=False,
                 recenter=True, chain_ids=list(), site_indices=list(),
                 allowed_occurrences=list(), orientation=Quaternion()):
        self.assembly_name = assembly_name
        self.name = name
        self.contents = contents
        self.protein_name = protein_name
        self.atom_radius_multiplier = atom_radius_multiplier
        self.add_sticks = add_sticks
        self.recenter = recenter
        self.chain_ids = chain_ids
        self.site_indices = site_indices
        self.allowed_occurrences = allowed_occurrences
        self.orientation = orientation


class RNASequence(object):

    def __init__(self, assembly_name, name, source, shape, assembly_params,
                 t_range=None, shape_params=None):
        self.assembly_name = assembly_name
        self.name = name
        self.contents = ''.join(open(source).readlines())
        self.shape = shape
        self.assembly_params = assembly_params
        self.t_range = t_range
        self.shape_params = shape_params


class Surfactant(object):

    def __init__(self, surfactant_protein, name, position, random_seed):
        assert isinstance(position, Vector3)
        self.surfactant_protein = surfactant_protein
        self.name = name
        self.position = position
        self.random_seed = random_seed


class Cell(object):

    def __init__(self, name, size, shape, membrane, receptor,
                 position=Vector3(), random_seed=0):
        assert isinstance(position, Vector3)
        assert isinstance(size, Vector2)
        assert isinstance(membrane, Membrane)
        assert isinstance(receptor, SurfaceReceptor)
        self.name = name
        self.position = position
        self.shape = shape
        self.size = size
        self.membrane = membrane
        self.receptor = receptor
        self.random_seed = random_seed


class Membrane:
    """
    A membrane is a selective barrier. It allows some things to pass through but stops others. Such things may be molecules, ions, or other small particles. Biological membranes include cell membranes (outer coverings of cells or organelles that allow passage of certain constituents)
    """

    def __init__(self, sources, number_of_instances):
        """
        Create a new Membrane instance

        :param list sources: List of PDB filename containing the description of individual proteins of the membrane
        :param int number_of_instances: Total number of instances of the proteins
        """
        self.sources = sources
        self.number_of_instances = number_of_instances


class SurfaceReceptor:
    """
    A Receptor is a chemical structure, composed of protein, that receive and transduce signals that may be integrated into biological systems
    """

    def __init__(self, source, number_of_instances,
                 position=Vector3(), orientation=Quaternion()):
        """
        Create a new surface receptor instance

        :param string source: PDB filename containing the description of the surface receptor protein
        :param int number_of_instances: Total number of instances
        :param floats position: x,y and z coordinates of the protein. Coordinates are relative to the surface
        :param floats orientation: Quaternion defining the relative orientation of the protein on to the surface
        """
        assert isinstance(position, Vector3)
        assert isinstance(orientation, Quaternion)
        self.source = source
        self.number_of_instances = number_of_instances
        self.position = position
        self.orientation = orientation


class VirusProtein:

    def __init__(self, sources, number_of_instances, assembly_params,
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


class Virus:

    def __init__(self, name, assembly_params, protein_s=None, protein_e=None, protein_m=None,
                 membrane=None, transmembrane=None, rna_sequence=None,
                 position=Vector3()):
        assert isinstance(assembly_params, Vector2)
        assert isinstance(position, Vector3)
        if protein_s is not None:
            assert isinstance(protein_s, VirusProtein)
        if protein_e is not None:
            assert isinstance(protein_e, VirusProtein)
        if protein_m is not None:
            assert isinstance(protein_m, VirusProtein)
        if membrane is not None:
            assert isinstance(membrane, Membrane)
        if rna_sequence is not None:
            assert isinstance(rna_sequence, RNASequence)
        if transmembrane is not None:
            assert isinstance(transmembrane, Mesh)
        self.name = name
        self.protein_s = protein_s
        self.protein_e = protein_e
        self.protein_m = protein_m
        self.transmembrane = transmembrane
        self.membrane = membrane
        self.rna_sequence = rna_sequence
        self.assembly_params = assembly_params
        self.position = position
