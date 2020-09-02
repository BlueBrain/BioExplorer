#!/usr/bin/env python
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

from bioexplorer import BioExplorer, RNASequence, Protein, Virus, Surfactant, Membrane, Cell, Sugars, \
    Volume, Vector2, Vector3, Quaternion

be = BioExplorer('localhost:5000')
print('BioExplorer version ' + be.version())
be.reset()

# Model settings
protein_radius_multiplier = 2.0
protein_representation = BioExplorer.REPRESENTATION_ATOMS
protein_load_hydrogen = False

# Virus configuration
nb_protein_s = 62
nb_protein_e = 42
nb_protein_m = 50
add_rna = False
add_glycans = True
glycan_add_sticks = True

# Cell parameters
cell_size = 1600
cell_height = 80

# Resources
resource_folder = 'test_files/'
pdb_folder = resource_folder + 'pdb/'
rna_folder = resource_folder + 'rna/'
obj_folder = resource_folder + 'obj/'
glycan_folder = pdb_folder + 'glycans/'

complex_paths = [glycan_folder + 'complex/5.pdb', glycan_folder + 'complex/15.pdb', glycan_folder + 'complex/25.pdb',
                 glycan_folder + 'complex/35.pdb']
high_mannose_paths = [glycan_folder + 'high-mannose/1.pdb', glycan_folder + 'high-mannose/2.pdb',
                      glycan_folder + 'high-mannose/3.pdb', glycan_folder + 'high-mannose/4.pdb']
hybrid_paths = [glycan_folder + 'hybrid/20.pdb']
o_glycan_paths = [glycan_folder + 'o-glycan/12.pdb']

surfactant_head_source = pdb_folder + 'surfactant/1pw9.pdb'
surfactant_branch_source = pdb_folder + 'surfactant/1k6f.pdb'

glucose_path = pdb_folder + 'glucose.pdb'
lactoferrins_path = pdb_folder + 'immune/1b0l.pdb'
defensins_path = pdb_folder + 'immune/1ijv.pdb'


def add_virus(name, position, open_conformation_indices=list()):
    closed_conformation_indices = list()
    for i in range(nb_protein_s):
        if i not in open_conformation_indices:
            closed_conformation_indices.append(i)

    virus_protein_s = Protein(
        sources=[
            pdb_folder + '6vyb.pdb',         # Open conformation
            pdb_folder + 'sars-cov-2-v1.pdb' # Closed conformation
        ],
        load_hydrogen=protein_load_hydrogen, number_of_instances=nb_protein_s, assembly_params=Vector2(11.5, 0.0),
        cutoff_angle=0.999, orientation=Quaternion(0.087, 0.0, 0.996, 0.0),
        instance_indices=[open_conformation_indices, closed_conformation_indices])

    virus_protein_m = Protein(
        sources=[pdb_folder + 'QHD43419a.pdb'], load_hydrogen=protein_load_hydrogen, number_of_instances=nb_protein_m,
        assembly_params=Vector2(2.0, 0.0), cutoff_angle=0.999, orientation=Quaternion(0.99, 0.0, 0.0, 0.135))

    virus_protein_e = Protein(
        sources=[pdb_folder + 'QHD43418a.pdb'], load_hydrogen=protein_load_hydrogen, number_of_instances=nb_protein_e,
        assembly_params=Vector2(3.0, 0.0), cutoff_angle=0.9999, orientation=Quaternion(0.705, 0.705, -0.04, -0.04))

    virus_membrane = Membrane(
        sources=[pdb_folder + 'membrane/popc.pdb'],
        number_of_instances=15000
    )

    rna_sequence = None
    if add_rna:
        import math

        rna_sequence = RNASequence(
            source=rna_folder + 'sars-cov-2.rna',
            assembly_params=Vector2(11.0, 0.5),
            t_range=Vector2(0, 30.5 * math.pi), shape=be.RNA_SHAPE_TREFOIL_KNOT,
            shape_params=Vector3(1.51, 1.12, 1.93))

    coronavirus = Virus(
        name=name, protein_s=virus_protein_s, protein_e=virus_protein_e, protein_m=virus_protein_m,
        membrane=virus_membrane, rna_sequence=rna_sequence, assembly_params=Vector2(45.0, 1.5))

    clip_planes = list()
    if add_rna:
        clip_planes.append([0, 0, -1, 15])
    be.add_virus(
        virus=coronavirus, position=position, representation=protein_representation,
        atom_radius_multiplier=protein_radius_multiplier, clipping_planes=clip_planes)

    if add_glycans:
        ''' High-mannose '''
        indices = [61, 122, 234, 603, 709, 717, 801, 1074]
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE, protein_name=be.NAME_PROTEIN_S_CLOSED,
            paths=high_mannose_paths, indices=indices, add_sticks=glycan_add_sticks,
            allowed_occurrences=closed_conformation_indices)
        if len(open_conformation_indices) > 0:
            be.add_multiple_glycans(
                assembly_name=name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE, protein_name=be.NAME_PROTEIN_S_OPEN,
                paths=high_mannose_paths, indices=indices, index_offset=19, add_sticks=glycan_add_sticks,
                allowed_occurrences=open_conformation_indices)

        ''' Complex '''
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_COMPLEX, protein_name=be.NAME_PROTEIN_S_CLOSED,
            paths=complex_paths, indices=[17, 74, 149, 165, 282, 331, 343, 616, 1098, 1134, 1158, 1173, 1194],
            add_sticks=glycan_add_sticks, allowed_occurrences=closed_conformation_indices)
        if len(open_conformation_indices) > 0:
            be.add_multiple_glycans(
                assembly_name=name, glycan_type=be.NAME_GLYCAN_COMPLEX, protein_name=be.NAME_PROTEIN_S_OPEN,
                paths=complex_paths, indices=[17, 74, 149, 165, 282, 331, 343, 1098, 1134, 1158, 1173, 1194],
                index_offset=19, add_sticks=glycan_add_sticks, allowed_occurrences=open_conformation_indices)

        ''' Hybrid '''
        indices = [657]
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_HYBRID, protein_name=be.NAME_PROTEIN_S_CLOSED,
            paths=hybrid_paths, indices=indices, add_sticks=glycan_add_sticks,
            allowed_occurrences=closed_conformation_indices)
        if len(open_conformation_indices) > 0:
            be.add_multiple_glycans(
                assembly_name=name, glycan_type=be.NAME_GLYCAN_HYBRID, protein_name=be.NAME_PROTEIN_S_OPEN,
                paths=hybrid_paths, indices=indices, index_offset=19, add_sticks=glycan_add_sticks,
                allowed_occurrences=open_conformation_indices)

        ''' O-Glycans '''
        for index in [323, 325]:
            o_glycan_name = name + '_' + be.NAME_GLYCAN_O_GLYCAN + '_' + str(index)
            o_glycan = Sugars(
                assembly_name=name, name=o_glycan_name, source=o_glycan_paths[0],
                protein_name=name + '_' + be.NAME_PROTEIN_S_CLOSED, add_sticks=glycan_add_sticks,
                site_indices=[index])
            be.add_sugars(o_glycan)

        ''' High-mannose glycans on Protein M '''
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE, protein_name=be.NAME_PROTEIN_M,
            paths=high_mannose_paths, add_sticks=glycan_add_sticks)

        ''' Complex glycans on Protein E '''
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_COMPLEX, protein_name=be.NAME_PROTEIN_E, paths=complex_paths,
            add_sticks=glycan_add_sticks)


def add_cell(name, size, height, position=Vector3()):
    ace2_receptor = Protein(
        sources=[pdb_folder + '6m1d.pdb'], number_of_instances=20, position=Vector3(0.0, 6.0, 0.0))
    membrane = Membrane(
        sources=[pdb_folder + 'membrane/popc.pdb'], number_of_instances=1000000)
    cell = Cell(
        name=name, size=Vector2(size, height), shape=be.ASSEMBLY_SHAPE_SINUSOIDAL, membrane=membrane,
        receptor=ace2_receptor)
    be.add_cell(
        cell=cell, position=position, representation=protein_representation)

    if add_glycans:
        be.add_multiple_glycans(
            add_sticks=glycan_add_sticks, assembly_name=name, glycan_type=be.NAME_GLYCAN_COMPLEX,
            protein_name=be.NAME_RECEPTOR, paths=complex_paths, indices=[62, 99, 112, 331, 441, 699])
        be.add_multiple_glycans(
            add_sticks=glycan_add_sticks, assembly_name=name, glycan_type=be.NAME_GLYCAN_HYBRID,
            protein_name=be.NAME_RECEPTOR, paths=hybrid_paths, indices=[555])

        indices = [
            [164, Quaternion(0.707, 0.0, 0.707, 0.0)],
            [739, Quaternion(0.707, 0.0, 0.707, 0.0)]
        ]
        for index in indices:
            o_glycan_name = name + '_' + be.NAME_GLYCAN_O_GLYCAN + '_' + str(index[0])
            o_glycan = Sugars(
                assembly_name=name, name=o_glycan_name, source=o_glycan_paths[0],
                protein_name=name + '_' + be.NAME_RECEPTOR, add_sticks=glycan_add_sticks,
                chain_ids=[2, 4], site_indices=[index[0]], orientation=index[1])
            be.add_sugars(o_glycan)


def add_surfactant_d(name, position, random_seed):
    surfactant_d = Surfactant(
        name=name, surfactant_protein=be.SURFACTANT_PROTEIN_D, head_source=surfactant_head_source,
        branch_source=surfactant_branch_source)
    be.add_surfactant(
        surfactant=surfactant_d, representation=protein_representation, position=position, random_seed=random_seed)


def add_surfactant_a(name, position, random_seed):
    surfactant_a = Surfactant(
        name=name, surfactant_protein=be.SURFACTANT_PROTEIN_A, head_source=surfactant_head_source,
        branch_source=surfactant_branch_source)
    be.add_surfactant(
        surfactant=surfactant_a, representation=protein_representation, position=position, random_seed=random_seed)


def add_glucose(size, number):
    protein = Protein(sources=[glucose_path], load_non_polymer_chemicals=True, number_of_instances=number)
    volume = Volume(name=be.NAME_GLUCOSE, size=Vector2(size, size), protein=protein)
    be.add_volume(volume=volume, representation=be.REPRESENTATION_ATOMS,
                  position=Vector3(0.0, size / 2.0 - 200.0, 0.0))


def add_lactoferrins(size, number):
    lactoferrins = Protein(sources=[lactoferrins_path], load_non_polymer_chemicals=True,
                           number_of_instances=number)
    lactoferrins_volume = Volume(name=be.NAME_LACTOFERRIN, size=Vector2(size, size), protein=lactoferrins)
    be.add_volume(volume=lactoferrins_volume, representation=be.REPRESENTATION_ATOMS,
                  position=Vector3(0.0, size / 2.0 - 200.0, 0.0))


def add_defensins(size, number):
    defensins = Protein(sources=[defensins_path], load_non_polymer_chemicals=True, number_of_instances=number)
    defensins_volume = Volume(name=be.NAME_DEFENSIN, size=Vector2(size, size), protein=defensins)
    be.add_volume(volume=defensins_volume, representation=be.REPRESENTATION_ATOMS,
                  position=Vector3(0.0, size / 2.0 - 200.0, 0.0))


def test_high_glucose():
    # Suspend image streaming
    be.core_api().set_application_parameters(image_stream_fps=0)

    # Build full model
    add_virus(name='Coronavirus 1', position=Vector3(-289.5, -97, -97.5), open_conformation_indices=[1])
    add_virus(name='Coronavirus 2', position=Vector3(-79.5, -102, 229.5), open_conformation_indices=[1])
    add_virus(name='Coronavirus 3', position=Vector3(296.5, -125, 225.5), open_conformation_indices=[1])
    add_virus(name='Coronavirus 4', position=Vector3(4.5, 100, 7.5))
    add_virus(name='Coronavirus 5', position=Vector3(204.5, -100, 27.5), open_conformation_indices=[1])
    add_virus(name='Coronavirus 6', position=Vector3(54.5, -100, -257.5), open_conformation_indices=[1])

    add_cell(name='Cell 1', size=cell_size, height=cell_height, position=Vector3(4.5, -186.0, 7.0))

    add_surfactant_d(name='Surfactant-D 1', position=Vector3(74.0, 24.0, -45.0), random_seed=1)
    add_surfactant_d(name='Surfactant-D 2', position=Vector3(104.0, 175.0, -89.0), random_seed=2)
    add_surfactant_d(name='Surfactant-D 3', position=Vector3(-260.0, 50.0, 0.0), random_seed=6)

    add_surfactant_a(name='Surfactant-A 1', position=Vector3(-100.0, 150.0, 0.0), random_seed=2)

    add_glucose(cell_size, 360000)
    add_lactoferrins(cell_size, 50)
    add_defensins(cell_size, 100)

    # Apply default materials
    be.apply_default_color_scheme(be.SHADING_MODE_BASIC)

    # Set rendering settings
    be.core_api().set_renderer(background_color=[96 / 255, 125 / 255, 139 / 255], current='bio_explorer',
                               samples_per_pixel=1, subsampling=4, max_accum_frames=64)
    params = be.core_api().BioExplorerRendererParams()
    params.shadows = 0.75
    params.soft_shadows = 1.0
    be.core_api().set_renderer_params(params)

    # Restore image streaming
    be.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
