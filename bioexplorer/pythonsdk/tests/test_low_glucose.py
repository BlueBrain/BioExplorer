#!/usr/bin/env python
"""Test low glucose scenario"""

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
from bioexplorer import BioExplorer, RNASequence, Protein, AssemblyProtein, Virus, Surfactant, \
    Membrane, Cell, Sugars, Volume, Vector2, Vector3, Quaternion

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value

# Model settings
PROTEIN_RADIUS_MULTIPLIER = 1.0
PROTEIN_REPRESENTATION = BioExplorer.REPRESENTATION_ATOMS
PROTEIN_LOAD_HYDROGEN = False

# Virus configuration
NB_PROTEIN_S = 62
NB_PROTEIN_E = 42
NB_PROTEIN_M = 50
NB_DEFENSINS = 2
ADD_RNA = False

# Glycans
ADD_GLYCANS = True

# Cell configuration
CELL_SIZE = 1600
CELL_HEIGHT = 80

# Resources
RESOURCE_FOLDER = 'tests/test_files/'
PDB_FOLDER = RESOURCE_FOLDER + 'pdb/'
RNA_FOLDER = RESOURCE_FOLDER + 'rna/'
OBJ_FOLDER = RESOURCE_FOLDER + 'obj/'
GLYCAN_FOLDER = PDB_FOLDER + 'glycans/'

COMPLEX_PATHS = [
    GLYCAN_FOLDER + 'complex/5.pdb',
    GLYCAN_FOLDER + 'complex/15.pdb',
    GLYCAN_FOLDER + 'complex/25.pdb',
    GLYCAN_FOLDER + 'complex/35.pdb'
]
HIGH_MANNOSE_PATHS = [
    GLYCAN_FOLDER + 'high-mannose/1.pdb',
    GLYCAN_FOLDER + 'high-mannose/2.pdb',
    GLYCAN_FOLDER + 'high-mannose/3.pdb',
    GLYCAN_FOLDER + 'high-mannose/4.pdb'
]
HYBRID_PATHS = [GLYCAN_FOLDER + 'hybrid/20.pdb']
O_GLYCAN_PATHS = [GLYCAN_FOLDER + 'o-glycan/12.pdb']

SURFACTANT_HEAD_SOURCE = PDB_FOLDER + 'surfactant/1pw9.pdb'
SURFACTANT_BRANCH_SOURCE = PDB_FOLDER + 'surfactant/1k6f.pdb'

GLUCOSE_PATH = PDB_FOLDER + 'glucose.pdb'
LACTOFERRINS_PATH = PDB_FOLDER + 'immune/1b0l.pdb'
DEFENSINS_PATH = PDB_FOLDER + 'immune/1ijv.pdb'
LYMPHOCYTE_PATH = OBJ_FOLDER + 'lymphocyte.obj'


def add_virus(bioexplorer, name, position, open_conformation_indices=list()):
    closed_conformation_indices = list()
    for i in range(NB_PROTEIN_S):
        if i not in open_conformation_indices:
            closed_conformation_indices.append(i)

    virus_protein_s = Protein(
        sources=[
            PDB_FOLDER + '6vyb.pdb',         # Open conformation
            PDB_FOLDER + 'sars-cov-2-v1.pdb'  # Closed conformation
        ],
        load_hydrogen=PROTEIN_LOAD_HYDROGEN, occurences=NB_PROTEIN_S,
        assembly_params=Vector2(11.5, 0.0), cutoff_angle=0.999,
        orientation=Quaternion(0.0, 1.0, 0.0, 0.0),
        instance_indices=[open_conformation_indices, closed_conformation_indices])

    virus_protein_m = Protein(
        sources=[PDB_FOLDER + 'QHD43419a.pdb'], load_hydrogen=PROTEIN_LOAD_HYDROGEN,
        occurences=NB_PROTEIN_M, assembly_params=Vector2(2.0, 0.0),
        cutoff_angle=0.999, orientation=Quaternion(0.99, 0.0, 0.0, 0.135))

    virus_protein_e = Protein(
        sources=[PDB_FOLDER + 'QHD43418a.pdb'], load_hydrogen=PROTEIN_LOAD_HYDROGEN,
        occurences=NB_PROTEIN_E, assembly_params=Vector2(3.0, 0.0),
        cutoff_angle=0.9999, orientation=Quaternion(0.705, 0.705, -0.04, -0.04))

    virus_membrane = Membrane(
        sources=[PDB_FOLDER + 'membrane/popc.pdb'],
        occurences=15000
    )

    rna_sequence = None
    if ADD_RNA:
        rna_sequence = RNASequence(
            source=RNA_FOLDER + 'sars-cov-2.rna',
            assembly_params=Vector2(11.0, 0.5),
            t_range=Vector2(0, 30.5 * math.pi), shape=bioexplorer.RNA_SHAPE_TREFOIL_KNOT,
            shape_params=Vector3(1.51, 1.12, 1.93))

    coronavirus = Virus(
        name=name, protein_s=virus_protein_s, protein_e=virus_protein_e, protein_m=virus_protein_m,
        membrane=virus_membrane, rna_sequence=rna_sequence, assembly_params=Vector3(45.0, 1.5, 0.0))

    clip_planes = list()
    if ADD_RNA:
        clip_planes.append([0, 0, -1, 15])
    bioexplorer.add_virus(
        virus=coronavirus, position=position, representation=PROTEIN_REPRESENTATION,
        atom_radius_multiplier=PROTEIN_RADIUS_MULTIPLIER, clipping_planes=clip_planes)

    if ADD_GLYCANS:
        # High-mannose
        indices_closed = [61, 122, 234, 603, 709, 717, 801, 1074]
        indices_open = [61, 122, 234, 709, 717, 801, 1074]
        bioexplorer.add_multiple_glycans(
            assembly_name=name, glycan_type=bioexplorer.NAME_GLYCAN_HIGH_MANNOSE,
            protein_name=bioexplorer.NAME_PROTEIN_S_CLOSED, paths=HIGH_MANNOSE_PATHS,
            indices=indices_closed, representation=PROTEIN_REPRESENTATION)
        if open_conformation_indices:
            bioexplorer.add_multiple_glycans(
                assembly_name=name, glycan_type=bioexplorer.NAME_GLYCAN_HIGH_MANNOSE,
                protein_name=bioexplorer.NAME_PROTEIN_S_OPEN, paths=HIGH_MANNOSE_PATHS,
                indices=indices_open, representation=PROTEIN_REPRESENTATION)

        # Complex
        indices_closed = [17, 74, 149, 165, 282, 331,
                          343, 616, 657, 1098, 1134, 1158, 1173, 1194]
        indices_open = [17, 74, 149, 165, 282, 331, 343, 657, 1098, 1134, 1158, 1173, 1194]
        bioexplorer.add_multiple_glycans(
            assembly_name=name, glycan_type=bioexplorer.NAME_GLYCAN_COMPLEX,
            protein_name=bioexplorer.NAME_PROTEIN_S_CLOSED, paths=COMPLEX_PATHS,
            indices=indices_closed, representation=PROTEIN_REPRESENTATION)
        if open_conformation_indices:
            bioexplorer.add_multiple_glycans(
                assembly_name=name, glycan_type=bioexplorer.NAME_GLYCAN_COMPLEX,
                protein_name=bioexplorer.NAME_PROTEIN_S_OPEN, paths=COMPLEX_PATHS,
                indices=indices_open, representation=PROTEIN_REPRESENTATION)

        # O-Glycans
        for index in [323, 325]:
            o_glycan_name = name + '_' + bioexplorer.NAME_GLYCAN_O_GLYCAN + '_' + str(index)
            o_glycan = Sugars(
                assembly_name=name, name=o_glycan_name, source=O_GLYCAN_PATHS[0],
                protein_name=name + '_' + bioexplorer.NAME_PROTEIN_S_CLOSED,
                representation=PROTEIN_REPRESENTATION,
                site_indices=[index])
            bioexplorer.add_sugars(o_glycan)

        # High-mannose glycans on Protein M
        indices = [5]
        high_mannose_glycans = Sugars(
            orientation=Quaternion(0.707, 0.0, 0.0, 0.707),
            assembly_name=name, name=bioexplorer.NAME_GLYCAN_HIGH_MANNOSE,
            protein_name=name + '_' + bioexplorer.NAME_PROTEIN_M, source=HIGH_MANNOSE_PATHS[0],
            site_indices=indices,
            representation=PROTEIN_REPRESENTATION
        )
        bioexplorer.add_glycans(high_mannose_glycans)

        # Complex glycans on Protein E
        indices = [48, 66]
        complex_glycans = Sugars(
            orientation=Quaternion(0.707, 0.0, 0.0, 0.707),
            assembly_name=name, name=bioexplorer.NAME_GLYCAN_COMPLEX,
            protein_name=name + '_' + bioexplorer.NAME_PROTEIN_E, source=COMPLEX_PATHS[0],
            site_indices=indices,
            representation=PROTEIN_REPRESENTATION
        )
        bioexplorer.add_glycans(complex_glycans)

    for i in range(NB_DEFENSINS):
        defensin = AssemblyProtein(
            assembly_name=name, name=name + '_' + bioexplorer.NAME_DEFENSIN + '_' + str(i),
            source=DEFENSINS_PATH, assembly_params=Vector2(2.0, 0.0),
            shape=bioexplorer.ASSEMBLY_SHAPE_SPHERICAL,
            atom_radius_multiplier=PROTEIN_RADIUS_MULTIPLIER, representation=PROTEIN_REPRESENTATION,
            random_seed=100 + i + 1)
        bioexplorer.add_assembly_protein(defensin)


def add_cell(bioexplorer, name, size, height, position=Vector3()):
    ace2_receptor = Protein(
        sources=[PDB_FOLDER + '6m18.pdb'], occurences=20, position=Vector3(0.0, 6.0, 0.0))
    membrane = Membrane(
        sources=[PDB_FOLDER + 'membrane/popc.pdb'], occurences=1200000)
    cell = Cell(
        name=name, size=Vector2(size, height), shape=bioexplorer.ASSEMBLY_SHAPE_SINUSOIDAL,
        membrane=membrane, receptor=ace2_receptor)
    bioexplorer.add_cell(
        cell=cell, position=position, representation=PROTEIN_REPRESENTATION)

    if ADD_GLYCANS:
        bioexplorer.add_multiple_glycans(
            representation=PROTEIN_REPRESENTATION, assembly_name=name,
            glycan_type=bioexplorer.NAME_GLYCAN_COMPLEX,
            protein_name=bioexplorer.NAME_RECEPTOR, paths=COMPLEX_PATHS,
            indices=[53, 90, 103, 322, 432, 690])
        bioexplorer.add_multiple_glycans(
            representation=PROTEIN_REPRESENTATION, assembly_name=name,
            glycan_type=bioexplorer.NAME_GLYCAN_HYBRID, protein_name=bioexplorer.NAME_RECEPTOR,
            paths=HYBRID_PATHS, indices=[546])

        indices = [
            [155, Quaternion(0.707, 0.0, 0.707, 0.0)],
            [730, Quaternion(0.707, 0.0, 0.707, 0.0)]
        ]
        for index in indices:
            o_glycan_name = name + '_' + bioexplorer.NAME_GLYCAN_O_GLYCAN + '_' + str(index[0])
            o_glycan = Sugars(
                assembly_name=name, name=o_glycan_name, source=O_GLYCAN_PATHS[0],
                protein_name=name + '_' + bioexplorer.NAME_RECEPTOR,
                representation=PROTEIN_REPRESENTATION, chain_ids=[2, 4], site_indices=[index[0]],
                orientation=index[1])
            bioexplorer.add_sugars(o_glycan)


def add_surfactant_d(bioexplorer, name, position, random_seed):
    surfactant_d = Surfactant(
        name=name, surfactant_protein=bioexplorer.SURFACTANT_PROTEIN_D,
        head_source=SURFACTANT_HEAD_SOURCE, branch_source=SURFACTANT_BRANCH_SOURCE)
    bioexplorer.add_surfactant(
        surfactant=surfactant_d, representation=PROTEIN_REPRESENTATION, position=position,
        random_seed=random_seed)


def add_surfactant_a(bioexplorer, name, position, random_seed):
    surfactant_a = Surfactant(
        name=name, surfactant_protein=bioexplorer.SURFACTANT_PROTEIN_A,
        head_source=SURFACTANT_HEAD_SOURCE, branch_source=SURFACTANT_BRANCH_SOURCE)
    bioexplorer.add_surfactant(
        surfactant=surfactant_a, representation=PROTEIN_REPRESENTATION, position=position,
        random_seed=random_seed)


def add_glucose(bioexplorer, size, number):
    protein = Protein(sources=[GLUCOSE_PATH], load_non_polymer_chemicals=True,
                      occurences=number)
    volume = Volume(
        name=bioexplorer.NAME_GLUCOSE, size=Vector2(size, size), protein=protein)
    bioexplorer.add_volume(
        volume=volume, representation=bioexplorer.REPRESENTATION_ATOMS,
        position=Vector3(0.0, size / 2.0 - 200.0, 0.0))


def add_lactoferrins(bioexplorer, size, number):
    lactoferrins = Protein(
        sources=[LACTOFERRINS_PATH], load_non_polymer_chemicals=True, occurences=number)
    lactoferrins_volume = Volume(
        name=bioexplorer.NAME_LACTOFERRIN, size=Vector2(size, size), protein=lactoferrins)
    bioexplorer.add_volume(
        volume=lactoferrins_volume, representation=bioexplorer.REPRESENTATION_ATOMS,
        position=Vector3(0.0, size / 2.0 - 200.0, 0.0))


def add_defensins(bioexplorer, size, number):
    defensins = Protein(sources=[DEFENSINS_PATH],
                        load_non_polymer_chemicals=True, occurences=number)
    defensins_volume = Volume(name=bioexplorer.NAME_DEFENSIN,
                              size=Vector2(size, size), protein=defensins)
    bioexplorer.add_volume(volume=defensins_volume, representation=bioexplorer.REPRESENTATION_ATOMS,
                           position=Vector3(0.0, size / 2.0 - 200.0, 0.0))


def test_low_glucose():
    try:
        # Connect to BioExplorer server
        bio_explorer = BioExplorer('localhost:5000')
        core = bio_explorer.core_api()
        print('BioExplorer version ' + bio_explorer.version())
        bio_explorer.reset()

        # Suspend image streaming
        core.set_application_parameters(image_stream_fps=0)

        # Build full model
        add_virus(
            bio_explorer, name='Coronavirus 1', position=Vector3(-5.0, 19.0, -36.0))
        add_virus(
            bio_explorer, name='Coronavirus 2', position=Vector3(73.0, 93.0, -115.0))
        add_virus(
            bio_explorer, name='Coronavirus 3', position=Vector3(-84.0, 110.0, 75.0))
        add_virus(
            bio_explorer, name='Coronavirus 4', position=Vector3(-70.0, -100.0, 230.0),
            open_conformation_indices=[1])
        add_virus(
            bio_explorer, name='Coronavirus 5', position=Vector3(200.0, 20.0, -150.0))

        add_cell(
            bio_explorer, name='Cell 1', size=CELL_SIZE, height=CELL_HEIGHT,
            position=Vector3(4.5, -186, 7.0))

        add_surfactant_d(
            bio_explorer, name='Surfactant-D 1', position=Vector3(74.0, 24.0, -45.0), random_seed=1)
        add_surfactant_d(
            bio_explorer, name='Surfactant-D 2', position=Vector3(-30.0, 91.0, 20.0), random_seed=2)
        add_surfactant_d(
            bio_explorer, name='Surfactant-D 3', position=Vector3(-165.0, 140.0, 105.0),
            random_seed=1)
        add_surfactant_d(
            bio_explorer, name='Surfactant-D 4', position=Vector3(-260.0, 50.0, 0.0), random_seed=6)

        add_surfactant_a(
            bio_explorer, name='Surfactant-A 1', position=Vector3(200.0, 50.0, 150.0),
            random_seed=2)

        add_glucose(
            bio_explorer, CELL_SIZE, 120000)
        add_lactoferrins(
            bio_explorer, CELL_SIZE, 150)
        add_defensins(
            bio_explorer, CELL_SIZE, 300)

        # BBP only
        #
        # core.add_model(name='Emile', path=LYMPHOCYTE_PATH)
        # models = core.scene.models
        # lymphocyte_model_id = models[len(models) - 1]['id']
        # transformation = {'rotation': [0.707, 0.707, 0.0, 0.0],
        #                   'rotation_center': [0.0, 0.0, 0.0],
        #                   'scale': [2.0, 2.0, 2.0],
        #                   'translation': [-935.0, 0.0, 0.0]}
        # core.update_model(
        #     id=lymphocyte_model_id, transformation=transformation)

        # Apply default materials
        bio_explorer.apply_default_color_scheme(bio_explorer.SHADING_MODE_BASIC)

        # Set rendering settings
        core.set_renderer(
            background_color=[96 / 255, 125 / 255, 139 / 255], current='bio_explorer',
            samples_per_pixel=1, subsampling=4, max_accum_frames=64)
        params = core.BioExplorerRendererParams()
        params.shadows = 0.75
        params.soft_shadows = 1.0
        core.set_renderer_params(params)

        # Restore image streaming
        core.set_application_parameters(image_stream_fps=20)
    except Exception as ex:
        print(ex)
        raise


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
