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

    NAME_SURFACTANT_D = 'SP-D'
    NAME_SURFACTANT_A = 'SP-A'
    NAME_COLLAGEN = 'Collagen'
    NAME_GLUCOSE = 'Glucose'

    NAME_GLYCAN_HIGH_MANNOSE = 'High-mannose'
    NAME_GLYCAN_O_GLYCAN = 'O-glycan'
    NAME_GLYCAN_HYBRID = 'Hybrid'
    NAME_GLYCAN_COMPLEX = 'Complex'

    def __init__(self, url):
        """
        Create a new Steps instance
        """
        self._client = Client(url)

        if __version__ != self.version():
            raise RuntimeError(
                'Wrong version of the back-end. Use version ' + BIO_EXPLORER_VERSION + \
                ' for this version of the BioExplorer python library')

    def __str__(self):
        """Return a pretty-print of the class"""
        return "Virus Explorer for Brayns"

    def version(self):
        result = self._client.rockets_client.request(method='version')
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result['contents']

    def reset(self):
        ids = list()
        for model in self._client.scene.models:
            ids.append(model['id'])
        self._client.remove_model(array=ids)

    def export_to_cache(self, filename):
        params = dict()
        params['filename'] = filename
        result = self._client.rockets_client.request(method='export-to-cache', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])

    def export_to_xyzr(self, filename):
        params = dict()
        params['filename'] = filename
        result = self._client.rockets_client.request(method='export-to-xyzr', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])

    def remove_assembly(self, name):
        params = dict()
        params['name'] = name
        params['position'] = [0, 0, 0]
        params['clippingPlanes'] = list()
        result = self._client.rockets_client.request(method='remove-assembly', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])

    def add_virus(self, name, protein_s, protein_e, protein_m, trans_membrane, membrane,
                  rna_sequence=None, atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                  position=[0, 0, 0], clipping_planes=list(), delay_between_additions=0,
                  load_non_polymer_chemicals=False):

        shape = BioExplorer.ASSEMBLY_SHAPE_SPHERICAL

        _protein_s_open = ProteinDescriptor(assembly_name=name, name=name + '_' + self.NAME_PROTEIN_S_OPEN,
                                            contents=''.join(open(protein_s[0]).readlines()), shape=shape,
                                            load_hydrogen=protein_s[2],
                                            occurrences=protein_s[3], assembly_params=protein_s[4],
                                            atom_radius_multiplier=atom_radius_multiplier, load_bonds=True,
                                            load_non_polymer_chemicals=load_non_polymer_chemicals,
                                            representation=representation, random_seed=1,
                                            location_cutoff_angle=protein_s[5], position=protein_s[6],
                                            orientation=protein_s[7], allowed_occurrences=protein_s[8])

        allowed_occurrences = list()
        for i in range(protein_s[2]):
            if i not in protein_s[7]:
                allowed_occurrences.append(i)

        _protein_s_closed = ProteinDescriptor(assembly_name=name, name=name + '_' + self.NAME_PROTEIN_S_CLOSED,
                                              contents=''.join(open(protein_s[1]).readlines()), shape=shape,
                                              load_hydrogen=protein_s[2],
                                              occurrences=protein_s[3], assembly_params=protein_s[4],
                                              atom_radius_multiplier=atom_radius_multiplier, load_bonds=True,
                                              load_non_polymer_chemicals=load_non_polymer_chemicals,
                                              representation=representation, random_seed=1,
                                              location_cutoff_angle=protein_s[5], position=protein_s[6],
                                              orientation=protein_s[7], allowed_occurrences=allowed_occurrences)

        _protein_m = ProteinDescriptor(assembly_name=name, name=name + '_' + self.NAME_PROTEIN_M,
                                       contents=''.join(open(protein_m[0]).readlines()), shape=shape,
                                       load_hydrogen=protein_m[1],
                                       occurrences=protein_m[2], assembly_params=protein_m[3],
                                       atom_radius_multiplier=atom_radius_multiplier, load_bonds=True,
                                       load_non_polymer_chemicals=load_non_polymer_chemicals,
                                       representation=representation, random_seed=3, location_cutoff_angle=protein_m[4],
                                       position=protein_m[5], orientation=protein_m[6])

        _protein_e = ProteinDescriptor(assembly_name=name, name=name + '_' + self.NAME_PROTEIN_E,
                                       contents=''.join(open(protein_e[0]).readlines()), shape=shape,
                                       load_hydrogen=protein_e[1],
                                       occurrences=protein_e[2], assembly_params=protein_e[3],
                                       atom_radius_multiplier=atom_radius_multiplier, load_bonds=True,
                                       load_non_polymer_chemicals=load_non_polymer_chemicals,
                                       representation=representation, random_seed=3, location_cutoff_angle=protein_e[4],
                                       position=protein_e[5], orientation=protein_e[6])

        contents = list()
        for path in membrane[0]:
            contents.append(''.join(open(path).readlines()))
        _membrane = MembraneDescriptor(assembly_name=name, name=name + '_' + self.NAME_MEMBRANE,
                                       shape=shape, contents=contents, occurrences=membrane[1],
                                       load_non_polymer_chemicals=True, assembly_params=membrane[2],
                                       atom_radius_multiplier=atom_radius_multiplier,
                                       load_bonds=False, representation=representation, random_seed=40,
                                       location_cutoff_angle=membrane[3],
                                       position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL)

        _trans_membrane = MeshDescriptor(recenter=True, assembly_name=name, name=name + '_' + self.NAME_TRANS_MEMBRANE,
                                         contents=''.join(open(trans_membrane[0]).readlines()), shape=shape,
                                         occurrences=trans_membrane[1], assembly_params=trans_membrane[2],
                                         random_seed=1, position_randomization_type=0,
                                         location_cutoff_angle=trans_membrane[3],
                                         position=trans_membrane[4], orientation=trans_membrane[5])

        if rna_sequence is not None:
            import math
            _rna_sequence = RNASequenceDescriptor(
                assembly_name=name,
                name=name + '_' + self.VIRUS_NAME_RNA_SEQUENCE, contents=''.join(open(rna_sequence[0]).readlines()),
                assembly_params=rna_sequence[1], radius=rna_sequence[2],
                t_range=rna_sequence[3],
                shape=rna_sequence[4], shape_params=rna_sequence[5])

        import time
        self.remove_assembly(name)
        time.sleep(delay_between_additions)
        self.add_assembly(name=name, position=position, clipping_planes=clipping_planes)
        time.sleep(delay_between_additions)
        self.add_protein(_protein_s_open)
        time.sleep(delay_between_additions)
        self.add_protein(_protein_s_closed)
        time.sleep(delay_between_additions)
        self.add_protein(_protein_m)
        time.sleep(delay_between_additions)
        self.add_protein(_protein_e)
        time.sleep(delay_between_additions)
        self.add_mesh(_trans_membrane)
        time.sleep(delay_between_additions)
        self.add_membrane(_membrane)
        if rna_sequence is not None:
            self.add_rna_sequence(_rna_sequence)

        palette = 'Greens'
        for protein_name in [self.NAME_PROTEIN_S_OPEN, self.NAME_PROTEIN_S_CLOSED]:
            self.set_protein_color_scheme(
                assembly_name=name,
                protein_name=name + '_' + protein_name,
                color_scheme=self.COLOR_SCHEME_CHAINS,
                palette_name=palette, palette_size=4)

        self.set_protein_color_scheme(
            assembly_name=name,
            protein_name=name + '_' + self.NAME_PROTEIN_E,
            color_scheme=self.COLOR_SCHEME_RESIDUES,
            palette_name=palette, palette_size=30)

        self.set_protein_color_scheme(
            assembly_name=name,
            protein_name=name + '_' + self.NAME_PROTEIN_M,
            color_scheme=self.COLOR_SCHEME_RESIDUES,
            palette_name=palette, palette_size=30)

        palette = 'inferno'
        for i in range(len(membrane[0])):
            self.set_protein_color_scheme(
                assembly_name=name,
                protein_name=name + '_' + self.NAME_MEMBRANE + '_' + str(i),
                color_scheme=self.COLOR_SCHEME_CHAINS,
                palette_name=palette, palette_size=7)

    def add_cell(self, name, receptor, membrane, shape=ASSEMBLY_SHAPE_PLANAR,
                 atom_radius_multiplier=1.0, representation=REPRESENTATION_ATOMS,
                 position=[0, 0, 0], clipping_planes=list(), delay_between_additions=0):

        _receptor = ProteinDescriptor(assembly_name=name, name=name + '_' + self.NAME_RECEPTOR, shape=shape,
                                      contents=''.join(open(receptor[0]).readlines()),
                                      occurrences=receptor[1], assembly_params=receptor[2],
                                      atom_radius_multiplier=atom_radius_multiplier,
                                      load_bonds=True, representation=representation, random_seed=1,
                                      location_cutoff_angle=receptor[3], position=receptor[4], orientation=receptor[5])

        contents = list()
        for path in membrane[0]:
            contents.append(''.join(open(path).readlines()))
        _membrane = MembraneDescriptor(assembly_name=name, name=name + '_' + self.NAME_MEMBRANE,
                                       shape=shape, contents=contents, occurrences=membrane[1],
                                       load_non_polymer_chemicals=True, assembly_params=membrane[2],
                                       atom_radius_multiplier=atom_radius_multiplier,
                                       load_bonds=False, representation=representation, random_seed=40,
                                       location_cutoff_angle=membrane[3],
                                       position_randomization_type=self.POSITION_RANDOMIZATION_TYPE_RADIAL)

        import time
        self.remove_assembly(name)
        time.sleep(delay_between_additions)
        self.add_assembly(name=name, position=position, clipping_planes=clipping_planes)
        time.sleep(delay_between_additions)
        self.add_protein(_receptor)
        time.sleep(delay_between_additions)
        self.add_membrane(_membrane)

        palette = 'OrRd_r'
        self.set_protein_color_scheme(
            assembly_name=name,
            protein_name=name + '_' + self.NAME_RECEPTOR,
            color_scheme=self.COLOR_SCHEME_CHAINS,
            palette_name=palette, palette_size=7)

        palette = 'inferno'
        self.set_protein_color_scheme(
            assembly_name=name,
            protein_name=name + '_' + self.NAME_MEMBRANE,
            color_scheme=self.COLOR_SCHEME_CHAINS,
            palette_name=palette, palette_size=7)

    def add_assembly(self, name, position=[0, 0, 0], clipping_planes=list()):
        clipping_planes_values = list()
        for plane in clipping_planes:
            for i in range(4):
                clipping_planes_values.append(plane[i])

        params = dict()
        params['name'] = name
        params['position'] = position
        params['clippingPlanes'] = clipping_planes_values
        result = self._client.rockets_client.request(method='add-assembly', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def apply_transformations(self, protein_descriptor, transformations):
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
        params['assemblyName'] = protein_descriptor.assembly_name
        params['name'] = protein_descriptor.name
        params['transformations'] = transformations_as_floats
        result = self._client.rockets_client.request(method='apply-transformations', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def set_protein_color_scheme(self, assembly_name, protein_name, color_scheme, palette_name='', palette_size=256,
                                 palette=list(), chain_ids=list()):
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
        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = protein_name
        params['sequence'] = amino_acid_sequence
        result = self._client.rockets_client.request(method='set-protein-amino-acid-sequence-as-string', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def set_protein_amino_acid_sequence_as_range(self, assembly_name, protein_name, amino_acid_range):
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
        params = dict()
        params['assemblyName'] = assembly_name
        params['name'] = protein_name
        result = self._client.rockets_client.request(method='get-protein-amino-acid-information', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        return result['contents'].split()

    def add_rna_sequence(self, rna_sequence_descriptor):

        t_range = [0.0, 2.0 * math.pi]
        if rna_sequence_descriptor.t_range is None:
            ''' Defaults '''
            if rna_sequence_descriptor.shape == self.RNA_SHAPE_TORUS:
                t_range = [0.0, 2.0 * math.pi]
            elif rna_sequence_descriptor.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                t_range = [0.0, 4.0 * math.pi]
        else:
            t_range = rna_sequence_descriptor.t_range

        shape_params = [1.0, 1.0, 1.0]
        if rna_sequence_descriptor.shape_params is None:
            ''' Defaults '''
            if rna_sequence_descriptor.shape == self.RNA_SHAPE_TORUS:
                shape_params = [0.5, 10.0, 0.0]
            elif rna_sequence_descriptor.shape == self.RNA_SHAPE_TREFOIL_KNOT:
                shape_params = [2.5, 2.0, 2.2]

        else:
            shape_params = rna_sequence_descriptor.shape_params

        params = dict()
        params['assemblyName'] = rna_sequence_descriptor.assembly_name
        params['name'] = rna_sequence_descriptor.name
        params['contents'] = rna_sequence_descriptor.contents
        params['shape'] = rna_sequence_descriptor.shape
        params['assemblyRadius'] = rna_sequence_descriptor.assembly_params
        params['radius'] = rna_sequence_descriptor.radius
        params['range'] = t_range
        params['params'] = shape_params
        result = self._client.rockets_client.request(method='add-rna-sequence', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_membrane(self, membrane_descriptor):
        params = dict()
        params['assemblyName'] = membrane_descriptor.assembly_name
        params['name'] = membrane_descriptor.name
        params['content1'] = membrane_descriptor.content1
        params['content2'] = membrane_descriptor.content2
        params['content3'] = membrane_descriptor.content3
        params['content4'] = membrane_descriptor.content4
        params['shape'] = membrane_descriptor.shape
        params['assemblyParams'] = membrane_descriptor.assembly_params
        params['atomRadiusMultiplier'] = membrane_descriptor.atom_radius_multiplier
        params['loadBonds'] = membrane_descriptor.load_bonds
        params['loadNonPolymerChemicals'] = membrane_descriptor.load_non_polymer_chemicals
        params['representation'] = membrane_descriptor.representation
        params['chainIds'] = membrane_descriptor.chain_ids
        params['recenter'] = membrane_descriptor.recenter
        params['occurrences'] = membrane_descriptor.occurrences
        params['randomSeed'] = membrane_descriptor.random_seed
        params['locationCutoffAngle'] = membrane_descriptor.location_cutoff_angle
        params['positionRandomizationType'] = membrane_descriptor.position_randomization_type
        params['orientation'] = membrane_descriptor.orientation
        result = self._client.rockets_client.request(method='add-membrane', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_protein(self, protein_descriptor):
        params = dict()
        params['assemblyName'] = protein_descriptor.assembly_name
        params['name'] = protein_descriptor.name
        params['contents'] = protein_descriptor.contents
        params['shape'] = protein_descriptor.shape
        params['assemblyParams'] = protein_descriptor.assembly_params
        params['atomRadiusMultiplier'] = protein_descriptor.atom_radius_multiplier
        params['loadBonds'] = protein_descriptor.load_bonds
        params['loadNonPolymerChemicals'] = protein_descriptor.load_non_polymer_chemicals
        params['loadHydrogen'] = protein_descriptor.load_hydrogen
        params['representation'] = protein_descriptor.representation
        params['chainIds'] = protein_descriptor.chain_ids
        params['recenter'] = protein_descriptor.recenter
        params['occurrences'] = protein_descriptor.occurrences
        params['allowedOccurrences'] = protein_descriptor.allowed_occurrences
        params['randomSeed'] = protein_descriptor.random_seed
        params['locationCutoffAngle'] = protein_descriptor.location_cutoff_angle
        params['positionRandomizationType'] = protein_descriptor.position_randomization_type
        params['position'] = protein_descriptor.position
        params['orientation'] = protein_descriptor.orientation
        result = self._client.rockets_client.request(method='add-protein', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_mesh(self, mesh_descriptor):
        params = dict()
        params['assemblyName'] = mesh_descriptor.assembly_name
        params['name'] = mesh_descriptor.name
        params['contents'] = mesh_descriptor.contents
        params['shape'] = mesh_descriptor.shape
        params['assemblyParams'] = mesh_descriptor.assembly_params
        params['recenter'] = mesh_descriptor.recenter
        params['occurrences'] = mesh_descriptor.occurrences
        params['randomSeed'] = mesh_descriptor.random_seed
        params['locationCutoffAngle'] = mesh_descriptor.location_cutoff_angle
        params['positionRandomizationType'] = mesh_descriptor.position_randomization_type
        params['position'] = mesh_descriptor.position
        params['orientation'] = mesh_descriptor.orientation
        result = self._client.rockets_client.request(method='add-mesh', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_glycans(self, glycans_descriptor):
        params = dict()
        params['assemblyName'] = glycans_descriptor.assembly_name
        params['name'] = glycans_descriptor.name
        params['contents'] = glycans_descriptor.contents
        params['proteinName'] = glycans_descriptor.protein_name
        params['atomRadiusMultiplier'] = glycans_descriptor.atom_radius_multiplier
        params['addSticks'] = glycans_descriptor.add_sticks
        params['recenter'] = glycans_descriptor.recenter
        params['chainIds'] = glycans_descriptor.chain_ids
        params['siteIndices'] = glycans_descriptor.site_indices
        params['allowedOccurrences'] = glycans_descriptor.allowed_occurrences
        params['orientation'] = glycans_descriptor.orientation
        result = self._client.rockets_client.request(method='add-glycans', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def add_multiple_glycans(self, assembly_name, glycan_type, protein_name, paths, chain_ids=list(), indices=list(),
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

            _glycans = SugarsDescriptor(
                assembly_name=assembly_name, name=assembly_name + '_' + protein_name + '_' + glycan_type,
                contents=''.join(open(path).readlines()),
                protein_name=assembly_name + '_' + protein_name, chain_ids=chain_ids,
                atom_radius_multiplier=radius_multiplier,
                add_sticks=add_sticks, recenter=True, site_indices=site_indices,
                allowed_occurrences=occurrences, orientation=[0, 0, 0, 1])
            self.add_glycans(_glycans)

    def add_glucoses(self, sugars_descriptor):
        params = dict()
        params['assemblyName'] = sugars_descriptor.assembly_name
        params['name'] = sugars_descriptor.name
        params['contents'] = sugars_descriptor.contents
        params['proteinName'] = sugars_descriptor.protein_name
        params['atomRadiusMultiplier'] = sugars_descriptor.atom_radius_multiplier
        params['addSticks'] = sugars_descriptor.add_sticks
        params['recenter'] = sugars_descriptor.recenter
        params['chainIds'] = sugars_descriptor.chain_ids
        params['siteIndices'] = sugars_descriptor.site_indices
        params['allowedOccurrences'] = sugars_descriptor.allowed_occurrences
        params['orientation'] = sugars_descriptor.orientation
        result = self._client.rockets_client.request(method='add-glucoses', params=params)
        if not result['status']:
            raise RuntimeError(result['contents'])
        self._client.set_renderer(accumulation=True)

    def _add_surfactant(self, assembly_name, surfactant_name, head_protein, collagen_protein, representation,
                        nb_branches, shape, position=None,
                        radius_multiplier=1.0, random_seed=0):

        if position is None:
            position = [0, 0, 0]
        nb_collagens = 2
        collagen_size = 16

        head_name = assembly_name + '_' + surfactant_name
        branch_name = assembly_name + '_' + self.NAME_COLLAGEN + '_'

        protein_sp_d = ProteinDescriptor(assembly_name=assembly_name, recenter=True,
                                         name=head_name, shape=shape,
                                         contents=''.join(open(head_protein).readlines()), occurrences=nb_branches,
                                         assembly_params=[collagen_size * (nb_collagens + 1) - 9, 0.0],
                                         atom_radius_multiplier=radius_multiplier, random_seed=random_seed,
                                         location_cutoff_angle=0.0, representation=representation,
                                         orientation=[-0.624, -0.417, 0.0, 0.661])

        collagens = list()
        contents = ''.join(open(collagen_protein).readlines())
        for i in range(nb_collagens):
            collagens.append(ProteinDescriptor(recenter=True, assembly_name=assembly_name, shape=shape,
                                               atom_radius_multiplier=radius_multiplier,
                                               name=branch_name + str(i),
                                               contents=contents, occurrences=nb_branches,
                                               assembly_params=[collagen_size * (i + 1) - 7, 0.0],
                                               random_seed=random_seed,
                                               location_cutoff_angle=0.0, representation=representation))

        self.remove_assembly(assembly_name)
        self.add_assembly(name=assembly_name, position=position)
        for collagen in collagens:
            self.add_protein(collagen)
        self.add_protein(protein_sp_d)

        palette = 'Reds'
        self.set_protein_color_scheme(assembly_name=assembly_name,
                                      protein_name=head_name,
                                      color_scheme=self.COLOR_SCHEME_CHAINS, palette_name=palette, palette_size=5)
        for i in range(nb_collagens):
            self.set_protein_color_scheme(assembly_name=assembly_name,
                                          protein_name=branch_name + str(i),
                                          color_scheme=self.COLOR_SCHEME_RESIDUES,

                                          palette_name=palette, palette_size=5)

    def add_surfactant_d(self, name, head_protein, collagen_protein, representation, position=None,
                         radius_multiplier=1.0,
                         random_seed=0):

        self._add_surfactant(name, self.NAME_SURFACTANT_D, head_protein, collagen_protein, representation,
                             4, self.ASSEMBLY_SHAPE_SPHERICAL, position, radius_multiplier, random_seed)

    def add_surfactant_a(self, name, head_protein, collagen_protein, representation, position=None,
                         radius_multiplier=1.0, random_seed=0):

        self._add_surfactant(name, self.NAME_SURFACTANT_A, head_protein, collagen_protein, representation,
                             6, self.ASSEMBLY_SHAPE_FAN, position, radius_multiplier, random_seed)

    def set_image_quality(self, image_quality=IMAGE_QUALITY_LOW):
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
                background_color=[0, 0, 0],
                current='basic',
                samples_per_pixel=1, subsampling=4, max_accum_frames=16)

    def get_material_ids(self, model_id):
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
                                   user_parameter=1.0, glossiness=1.0, emission=0.0):
        colors = list()
        shading_modes = list()
        user_parameters = list()
        glossinesses = list()
        specular_exponents = list()
        emissions = list()
        for color in palette:
            colors.append(color)
            shading_modes.append(shading_mode)
            user_parameters.append(user_parameter)
            specular_exponents.append(specular_exponent)
            glossinesses.append(glossiness)
            emissions.append(emission)
        self.set_materials(
            model_ids=model_ids, material_ids=material_ids, diffuse_colors=colors, specular_colors=colors,
            specular_exponents=specular_exponents, user_parameters=user_parameters, glossinesses=glossinesses,
            shading_modes=shading_modes, emissions=emissions)

    def apply_default_color_scheme(self, shading_mode, user_parameter=0.03, specular_exponent=5.0, glossiness=0.5):
        from ipywidgets import IntProgress
        from IPython.display import display

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
                palette = sns.color_palette('Greys_r', nb_materials)
                self.set_materials_from_palette(
                    model_ids=[model_id], material_ids=material_ids, palette=palette,
                    shading_mode=shading_mode, user_parameter=user_parameter, glossiness=glossiness,
                    specular_exponent=specular_exponent)

            if self.NAME_TRANS_MEMBRANE in model_name:
                palette = list()
                for p in range(nb_materials):
                    palette.append([1, 1, 1])
                self.set_materials_from_palette(model_ids=[model_id], material_ids=material_ids, palette=palette,
                                                opacity=0.7, reflection_index=0, refraction_index=1.1,
                                                shading_mode=self.ce.SHADING_MODE_DIFFUSE,
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

            if self.NAME_SURFACTANT_D in model_name or self.NAME_SURFACTANT_A in model_name or \
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
        params = dict()
        params['minValue'] = min_value
        params['maxValue'] = max_value
        params['steps'] = interval
        params['radius'] = radius
        params['planeOpacity'] = opacity
        params['showAxis'] = show_axis
        params['useColors'] = colored
        return self._client.rockets_client.request('add-grid', params)


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


class MeshDescriptor(object):

    def __init__(self, assembly_name, name, contents, assembly_params, shape=BioExplorer.ASSEMBLY_SHAPE_PLANAR,
                 recenter=True, occurrences=1, random_seed=0, location_cutoff_angle=0,
                 position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_CIRCULAR,
                 position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0]):
        self.assembly_name = assembly_name
        self.name = name
        self.contents = contents
        self.shape = shape
        self.assembly_params = assembly_params
        self.recenter = recenter
        self.occurrences = occurrences
        self.random_seed = random_seed
        self.location_cutoff_angle = location_cutoff_angle
        self.position_randomization_type = position_randomization_type
        self.position = position
        self.orientation = orientation


class MembraneDescriptor(object):

    def __init__(self, assembly_name, name, contents, shape, assembly_params,
                 atom_radius_multiplier=1, load_bonds=False, representation=BioExplorer.REPRESENTATION_ATOMS,
                 load_non_polymer_chemicals=False,
                 chain_ids=list(), recenter=True, occurrences=1, random_seed=0,
                 location_cutoff_angle=0.0,
                 position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_CIRCULAR,
                 orientation=[0, 0, 0, 1]):
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


class ProteinDescriptor(object):

    def __init__(self, assembly_name, name, contents, assembly_params, shape=BioExplorer.ASSEMBLY_SHAPE_PLANAR,
                 atom_radius_multiplier=1, load_bonds=False, representation=BioExplorer.REPRESENTATION_ATOMS,
                 load_non_polymer_chemicals=False, load_hydrogen=True,
                 chain_ids=list(), recenter=True, occurrences=1, random_seed=0,
                 location_cutoff_angle=0.0,
                 position_randomization_type=BioExplorer.POSITION_RANDOMIZATION_TYPE_CIRCULAR,
                 position=[0, 0, 0], orientation=[0, 0, 0, 1], allowed_occurrences=list()):
        self.assembly_name = assembly_name
        self.name = name
        self.contents = contents
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


class SugarsDescriptor(object):

    def __init__(self, assembly_name, name, contents, protein_name, atom_radius_multiplier=1.0, add_sticks=False,
                 recenter=True, chain_ids=list(), site_indices=list(), allowed_occurrences=list(),
                 orientation=[0, 0, 0, 1]):
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


class RNASequenceDescriptor(object):

    def __init__(self, assembly_name, name, contents, shape, assembly_params, radius, t_range=None,
                 shape_params=None):
        self.assembly_name = assembly_name
        self.name = name
        self.contents = contents
        self.shape = shape
        self.assembly_params = assembly_params
        self.radius = radius
        self.t_range = t_range
        self.shape_params = shape_params
