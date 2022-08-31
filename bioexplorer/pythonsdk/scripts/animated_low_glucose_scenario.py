#!/usr/bin/env python
"""Animated low glucose scenario"""

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

from bioexplorer import BioExplorer, Protein, Membrane, Surfactant, Cell, \
    Sugars, AnimationParams, Volume, Vector2, Vector3, Quaternion, \
    MovieScenario
import math
import sys

resource_folder = './tests/test_files/'

# --------------------------------------------------------------------------------
# Movie settings
# --------------------------------------------------------------------------------
image_k = 4
image_samples_per_pixels = 64
image_projection = 'perspective'
image_output_folder = '/tmp'

# --------------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------------
scenario = 'low_glucose'
# Scene
scene_size = Vector3(800.0, 800.0, 800.0)

# Proteins
protein_radius_multiplier = 1.0
protein_representation = BioExplorer.REPRESENTATION_ATOMS_AND_STICKS
protein_load_hydrogen = False

# Glycans
add_glycans = True
glycan_radius_multiplier = 1.0
glycan_representation = BioExplorer.REPRESENTATION_ATOMS_AND_STICKS

# Viruses
nb_protein_s = 62
nb_protein_e = 42
nb_protein_m = 50
add_rna = True
landing_distance = 50.0

# Immune system
nb_glucoses = 120000
nb_lactoferrins = 150
nb_defensins = 300

# Cells
cell_nb_receptors = 100
lipid_density = 1.0

# --------------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------------
image_folder = resource_folder + 'images/'
pdb_folder = resource_folder + 'pdb/'
rna_folder = resource_folder + 'rna/'
obj_folder = resource_folder + 'obj/'
glycan_folder = pdb_folder + 'glycans/'
membrane_folder = pdb_folder + 'membrane/'

complex_paths = [glycan_folder + 'complex/33.pdb', glycan_folder + 'complex/34.pdb',
                 glycan_folder + 'complex/35.pdb', glycan_folder + 'complex/36.pdb']
high_mannose_paths = [glycan_folder + 'high-mannose/1.pdb',
                      glycan_folder + 'high-mannose/2.pdb',
                      glycan_folder + 'high-mannose/3.pdb',
                      glycan_folder + 'high-mannose/4.pdb']
hybrid_paths = [glycan_folder + 'hybrid/24.pdb']
o_glycan_paths = [glycan_folder + 'o-glycan/12.pdb']

glucose_path = pdb_folder + 'glucose.pdb'
lactoferrin_path = pdb_folder + 'immune/1b0l.pdb'
defensin_path = pdb_folder + 'immune/1ijv.pdb'

surfactant_head_source = pdb_folder + 'surfactant/1pw9.pdb'
surfactant_branch_source = pdb_folder + 'surfactant/1k6f.pdb'

lymphocyte_path = obj_folder + 'clipped_lymphocyte.obj'

# --------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------
ROTATION_MODE_LINEAR = 0
ROTATION_MODE_SINUSOIDAL = 1


class LowGlucoseScenario(MovieScenario):

    def __init__(self, hostname, port, projection, output_folder, image_k=4,
                 image_samples_per_pixel=64, log_level=1, shaders=list(['bio_explorer'])):
        super().__init__(hostname, port, projection, output_folder,
                         image_k, image_samples_per_pixel, log_level, False, shaders)

    def _get_transformation(self, start_frame, end_frame, frame, data):
        '''Progress'''
        progress = (frame - start_frame) * 1.0 / (end_frame - start_frame)
        progress = max(0.0, progress)
        progress = min(1.0, progress)

        '''Position'''
        start_pos = data[0].to_list()
        end_pos = data[2].to_list()
        pos = start_pos
        for i in range(3):
            pos[i] += (end_pos[i] - start_pos[i]) * progress

        '''Rotation'''
        start_rot = data[1]
        end_rot = data[3]
        rot = Quaternion.slerp(start_rot, end_rot, progress)
        if data[4] == ROTATION_MODE_SINUSOIDAL:
            rot = Quaternion.slerp(start_rot, end_rot, math.cos((progress - 0.5) * math.pi))

        return [Vector3(pos[0], pos[1], pos[2]), rot, progress * 100.0]

    def _add_viruses(self, frame):
        virus_sequences = [
            [[0, 2599], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6]],
            [[0, 2599], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6]],
            [[0, 2599], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6]],
            [[0, 2999], [3000, 3099], [3100, 3299], [3300, 3750], [1e6, 1e6], [1e6, 1e6]],
            [[0, 599], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1600, 3750]],
        ]
        virus_flights_in = [
            [Vector3(-35.0, 300.0, -70.0), Quaternion(0.519, 0.671, 0.528, -0.036),
             Vector3(-5.0, 45.0, -33.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            [Vector3(153.0, 300.0, -200.0), Quaternion(0.456, 0.129, -0.185, -0.860),
             Vector3(73.0, 93.0, -130.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            # Virus used for SP-D zoom
            [Vector3(-100.0, 300.0, 20.0), Quaternion(0.087, 0.971, -0.147, -0.161),
             Vector3(-79.0, 108.0, 80.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            # Virus getting inside cell
            [Vector3(224.9, 300.0, -220.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             Vector3(211.5, -104.9, -339.2), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            # Virus used for detailed view of the Spike
            [Vector3(200.0, 20.0, -150.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(200.0, 20.0, -150.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR]
        ]

        virus_flights_out = [
            # Unused
            [Vector3(-5.0, 45.0, -33.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(-105.0, 45.0, -33.0), Quaternion(0.0, 1.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            # Unused
            [Vector3(73.0, 93.0, -130.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(-33.0, 93.0, -130.0), Quaternion(0.0, 0.0, 1.0, 0.0),
             ROTATION_MODE_LINEAR],
            # Virus used for SP-D zoom
            [Vector3(-84.0, 110.0, 75.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(-100.0, -100.0, 51.2), Quaternion(0.087, 0.971, -0.147, -0.161),
             ROTATION_MODE_LINEAR],
            # Unused
            [Vector3(0.0, 0.0, 0.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(0.0, 0.0, 0.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            # Virus used for detailed view of the Spike
            [Vector3(200.0, 20.0, -150.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(300.0, -100.0, -100.0), Quaternion(0.456, 0.129, -0.185, -0.860),
             ROTATION_MODE_LINEAR]
        ]

        for virus_index in range(len(virus_sequences)):
            name = 'sars-cov-2 ' + str(virus_index)
            current_sequence = 0
            sequences = virus_sequences[virus_index]
            for i in range(len(sequences)):
                if frame >= sequences[i][0]:
                    current_sequence = i

            '''Initialize position and rotation to end-of-flight values'''
            start_frame = sequences[current_sequence][0]
            end_frame = sequences[current_sequence][1]
            progress_in_sequence = (frame - start_frame) / (end_frame - start_frame)
            morphing_step = 0.0

            if current_sequence == 0:
                '''Flying'''
                pos, rot, progress = self._get_transformation(start_frame, end_frame,
                                                              frame, virus_flights_in[virus_index])
                self._log(3, '-   Virus %d is flying in... (%.01f pct)' % (virus_index, progress))
            elif current_sequence == 1:
                '''Landing'''
                pos = virus_flights_in[virus_index][2]
                rot = virus_flights_in[virus_index][3]
                pos.y -= landing_distance * progress_in_sequence
                self._log(3, '-   Virus %d is landing...' % virus_index)
            elif current_sequence == 2:
                '''Merging into cell'''
                pos = virus_flights_in[virus_index][2]
                rot = virus_flights_in[virus_index][3]
                morphing_step = (frame - start_frame) / (end_frame - start_frame)
                pos.y -= landing_distance
                self._log(3, '-   Virus %d is merging in (%.01f pct)' %
                          (virus_index, morphing_step * 100.0))
            elif current_sequence == 3:
                '''Inside cell'''
                self._log(3, '-   Virus %d is inside cell' % virus_index)
                '''Virus is not added to the scene'''
                self._be.remove_assembly(name=name)
                continue
            elif current_sequence == 4:
                '''Merging out of cell'''
                pos = virus_flights_out[virus_index][0]
                rot = virus_flights_out[virus_index][1]
                morphing_step = 1.0 - (frame - start_frame) / (end_frame - start_frame)
                self._log(3, '-   Virus %d is merging out (%.01f pct)' %
                          (virus_index, morphing_step * 100.0))
            else:
                '''Flying out'''
                pos, rot, progress = self._get_transformation(start_frame, end_frame,
                                                              frame, virus_flights_out[virus_index])
                self._log(3, '-   Virus %d is flying out... (%.01f pct)' % (virus_index, progress))

            self._be.add_sars_cov_2(
                name=name, resource_folder=resource_folder,
                representation=protein_representation, position=pos, rotation=rot,
                add_glycans=add_glycans, add_rna_sequence=add_rna,
                animation_params=AnimationParams(
                    1, 5 * frame + 2 * virus_index, 0.25,
                    frame + 2 * virus_index + 1, 0.1, morphing_step)
            )

    def _add_cell(self, frame):
        name = 'Cell'
        receptor_name = name + '_' + BioExplorer.NAME_RECEPTOR
        nb_receptors = cell_nb_receptors
        size = Vector3(scene_size.x * 2.0, scene_size.y / 10.0, scene_size.x * 2.0)
        position = Vector3(4.5, -186.0, 7.0)
        random_seed = 10

        ace2_receptor = Protein(
            name=receptor_name,
            source=pdb_folder + '6m18.pdb', occurrences=nb_receptors,
            transmembrane_params=Vector2(-6.0, 5.0),
            animation_params=AnimationParams(
                random_seed, frame + 1, 0.025, frame + 2, 0.2)
        )

        membrane = Membrane(
            lipid_sources=[
                membrane_folder + 'segA.pdb',
                membrane_folder + 'segB.pdb',
                membrane_folder + 'segC.pdb',
                membrane_folder + 'segD.pdb'
            ],
            animation_params=AnimationParams(
                random_seed, frame + 1, 0.025, frame + 2, 0.2)
        )

        cell = Cell(
            name=name,
            shape=BioExplorer.ASSEMBLY_SHAPE_SINUSOID,
            shape_params=size,
            membrane=membrane, proteins=[ace2_receptor],
        )

        self._check(self._be.add_cell(
            cell=cell, position=position,
            representation=protein_representation))

        '''Modify receptor position when attached virus enters the cell'''
        receptors_instances = [37]
        receptors_sequences = [[3000, 3099]]

        for i in range(len(receptors_instances)):
            instance_index = receptors_instances[i]
            sequence = receptors_sequences[i]
            start_frame = sequence[0]
            end_frame = sequence[1]
            if frame >= start_frame:
                if frame > end_frame:
                    '''Send receptor to outter space'''
                    self._check(self._be.set_protein_instance_transformation(
                        assembly_name=name, name=receptor_name,
                        instance_index=instance_index, position=Vector3(0.0, 1e6, 0.0)
                    ))
                else:
                    '''Current receptor transformation'''
                    transformation = self._be.get_protein_instance_transformation(
                        assembly_name=name, name=receptor_name,
                        instance_index=instance_index
                    )
                    p = transformation['position'].split(',')
                    q = transformation['rotation'].split(',')
                    pos = Vector3(float(p[0]), float(p[1]), float(p[2]))
                    q2 = Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

                    '''Bend receptor'''
                    progress = (frame - start_frame) * 1.0 / (end_frame - start_frame)
                    q1 = Quaternion(axis=[0, 1, 0], angle=-math.pi * progress)
                    rot = q2 * q1

                    pos.x += landing_distance * progress * 0.3
                    pos.y -= landing_distance * progress * 0.3

                    self._check(self._be.set_protein_instance_transformation(
                        assembly_name=name, name=receptor_name,
                        instance_index=instance_index, position=pos, rotation=rot
                    ))

        '''Glycans'''
        if nb_receptors != 0 and add_glycans:
            self._be.add_multiple_glycans(
                representation=glycan_representation, assembly_name=name,
                glycan_type=BioExplorer.NAME_GLYCAN_COMPLEX,
                protein_name=BioExplorer.NAME_RECEPTOR, paths=complex_paths,
                indices=[53, 90, 103, 322, 432, 690],
                animation_params=AnimationParams(0, 0, 0.0, frame + 3, 0.2)
            )
            self._be.add_multiple_glycans(
                representation=glycan_representation, assembly_name=name,
                glycan_type=BioExplorer.NAME_GLYCAN_HYBRID,
                protein_name=BioExplorer.NAME_RECEPTOR, paths=hybrid_paths,
                indices=[546],
                animation_params=AnimationParams(0, 0, 0.0, frame + 4, 0.2)
            )

            indices = [[155, Quaternion(0.707, 0.0, 0.707, 0.0)],
                       [730, Quaternion(0.707, 0.0, 0.707, 0.0)]]
            count = 0
            for index in indices:
                o_glycan_name = name + '_' + BioExplorer.NAME_GLYCAN_O_GLYCAN + '_' + str(index[0])
                o_glycan = Sugars(
                    assembly_name=name, name=o_glycan_name, source=o_glycan_paths[0],
                    protein_name=name + '_' + BioExplorer.NAME_RECEPTOR, representation=glycan_representation,
                    chain_ids=[2, 4], site_indices=[index[0]], rotation=index[1],
                    animation_params=AnimationParams(0, 0, 0.0, frame + count + 5, 0.2)
                )
                self._check(self._be.add_sugars(o_glycan))
                count += 1

    def _add_surfactant_d(self, name, position, rotation, animation_params):
        surfactant_d = Surfactant(
            name=name, surfactant_protein=BioExplorer.SURFACTANT_PROTEIN_D,
            head_source=surfactant_head_source,
            branch_source=surfactant_branch_source)
        self._check(self._be.add_surfactant(
            surfactant=surfactant_d, representation=protein_representation,
            position=position, rotation=rotation, animation_params=animation_params))

    def _add_surfactant_a(self, name, position, rotation, animation_params):
        surfactant_a = Surfactant(
            name=name, surfactant_protein=BioExplorer.SURFACTANT_PROTEIN_A,
            head_source=surfactant_head_source,
            branch_source=surfactant_branch_source)
        self._check(self._be.add_surfactant(
            surfactant=surfactant_a, representation=protein_representation,
            position=position, rotation=rotation, animation_params=animation_params))

    def _add_surfactants_d(self, frame):
        spd_sequences = [[0, 3750], [0, 2600], [0, 2600], [0, 3750]]
        spd_random_seeds = [1, 1, 1, 2]

        spd_flights = [
            [Vector3(300,  124.0, 0.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             Vector3(74.0,  24.0, -45.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_SINUSOIDAL],
            # SP-D is used for the head focus on 3rd virus spike
            [Vector3(-50,  250.0, 20.0), Quaternion(0.087, 0.971, -0.147, -0.161),
             Vector3(-11.0,  108.0, 20.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            # SP-D attaching to lymphocyte
            [Vector3(-200.0, 100.0, 100.0), Quaternion(0.519, 0.671, 0.528, -0.036),
             Vector3(-135.0, 135.0, 140.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            [Vector3(100.0,  0.0, -80.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             Vector3(-260.0,  50.0, 150.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_SINUSOIDAL]
        ]

        for surfactant_index in range(len(spd_sequences)):
            name = 'Surfactant-D ' + str(surfactant_index)
            sequence = spd_sequences[surfactant_index]
            pos, rot, progress = self._get_transformation(
                start_frame=sequence[0], end_frame=sequence[1],
                frame=frame, data=spd_flights[surfactant_index])
            self._log(3, '-   ' + name + ' (%.01f pct)' % progress)
            self._add_surfactant_d(
                name=name, position=pos, rotation=rot,
                animation_params=AnimationParams(spd_random_seeds[surfactant_index])
            )

    def _add_surfactants_a(self, frame):
        spa_sequences = [[0, 3750]]
        spa_random_seeds = [2]
        spa_frames = [
            [Vector3(300.0,  -50.0, -50.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             Vector3(100.0,  50.0, 150.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_SINUSOIDAL],
        ]

        for surfactant_index in range(len(spa_frames)):
            name = 'Surfactant-A ' + str(surfactant_index)
            sequence = spa_sequences[surfactant_index]
            pos, rot, progress = self._get_transformation(
                start_frame=sequence[0], end_frame=sequence[1],
                frame=frame, data=spa_frames[surfactant_index])
            self._log(3, '-   ' + name + ' (%.01f pct)' % progress)
            self._add_surfactant_a(
                name=name, position=pos, rotation=rot,
                animation_params=AnimationParams(spa_random_seeds[surfactant_index])
            )

    def _add_glucose(self, frame):
        glucose = Protein(
            name=BioExplorer.NAME_GLUCOSE,
            source=glucose_path,
            load_non_polymer_chemicals=True, load_bonds=True, load_hydrogen=True,
            occurrences=nb_glucoses,
            animation_params=AnimationParams(
                100, frame + 20, scene_size.y / 600.0, frame + 21, 0.3)
        )
        volume = Volume(
            name=BioExplorer.NAME_GLUCOSE,
            shape=BioExplorer.ASSEMBLY_SHAPE_CUBE,
            shape_params=scene_size,
            protein=glucose
        )
        self._check(self._be.add_volume(
            volume=volume, representation=protein_representation,
            position=Vector3(0.0, scene_size.y / 2.0 - 200.0, 0.0)
        ))

    def _add_lactoferrins(self, frame):
        lactoferrin = Protein(
            name=BioExplorer.NAME_LACTOFERRIN,
            source=lactoferrin_path,
            load_non_polymer_chemicals=True, load_bonds=True, load_hydrogen=True,
            occurrences=nb_lactoferrins,
            animation_params=AnimationParams(
                101, frame + 30, scene_size.y / 400.0, frame + 31, 0.3)
        )
        lactoferrins_volume = Volume(
            name=BioExplorer.NAME_LACTOFERRIN,
            shape=BioExplorer.ASSEMBLY_SHAPE_CUBE,
            shape_params=scene_size,
            protein=lactoferrin
        )
        self._check(self._be.add_volume(
            volume=lactoferrins_volume, representation=protein_representation,
            position=Vector3(0.0, scene_size.y / 2.0 - 200.0, 0.0)
        ))

    def _add_defensins(self, frame):
        defensin = Protein(
            name=BioExplorer.NAME_DEFENSIN,
            source=defensin_path,
            load_non_polymer_chemicals=True, load_bonds=True, load_hydrogen=True,
            occurrences=nb_defensins,
            animation_params=AnimationParams(
                102, frame + 40, scene_size.y / 400.0, frame + 41, 0.3)
        )
        defensins_volume = Volume(
            name=BioExplorer.NAME_DEFENSIN,
            shape=BioExplorer.ASSEMBLY_SHAPE_CUBE,
            shape_params=scene_size,
            protein=defensin
        )
        self._check(self._be.add_volume(
            volume=defensins_volume, representation=protein_representation,
            position=Vector3(0.0, scene_size.y / 2.0 - 200.0, 0.0)
        ))

    def _add_lymphocyte(self, frame):
        if frame < 1400:
            '''Lymphocyte is not in the field of view'''
            return
        '''Protein animation params'''
        params = [0, 0, 0.0, frame + 2, 0.2]

        name = 'Emile'
        lymphocyte_sequence = [0, 3750]
        lymphocyte_frames = [Vector3(-700, 100.0, 0.0), Quaternion(1.0, 0.0, 0.0, 0.0),
                             Vector3(-200.0, 100.0, 0.0), Quaternion(1.0, 0.0, 0.0, 0.0),
                             ROTATION_MODE_LINEAR]

        pdb_lipids = [
            membrane_folder + 'segA.pdb',
            membrane_folder + 'segB.pdb',
            membrane_folder + 'segC.pdb',
            membrane_folder + 'segD.pdb'
        ]

        membrane = Membrane(
            lipid_sources=pdb_lipids, lipid_density=lipid_density,
            load_non_polymer_chemicals=True, load_bonds=True,
            animation_params=AnimationParams(0, 1, 0.025, 2, 0.5))

        pos, rot, progress = self._get_transformation(
            start_frame=lymphocyte_sequence[0], end_frame=lymphocyte_sequence[1],
            frame=frame, data=lymphocyte_frames)
        self._log(3, '-   ' + name + ' (%.01f pct)' % progress)

        clip_planes = [
            [-1.0, 0.0, 0.0, scene_size.x / 2.0 + pos.x],
            [0.0, 1.0, 0.0, scene_size.y / 2.0 - pos.y],
            [0.0, 0.0, 1.0, scene_size.z / 2.0],
            [0.0, 0.0, -1.0, scene_size.z / 2.0],
        ]

        scale = Vector3(1.0, 1.0, 1.0)
        cell = Cell(
            name=name,
            shape=BioExplorer.ASSEMBLY_SHAPE_MESH,
            shape_mesh_source=lymphocyte_path,
            shape_params=scale,
            membrane=membrane,
            proteins=list())

        self._check(self._be.add_cell(
            cell=cell,
            position=pos, rotation=rot,
            clipping_planes=clip_planes
        ))

        for i in range(len(pdb_lipids)):
            self._check(self._be.set_protein_color_scheme(
                assembly_name=name, name=BioExplorer.NAME_MEMBRANE + '_' + str(i),
                color_scheme=BioExplorer.COLOR_SCHEME_CHAINS,
                palette_name='OrRd', palette_size=5))

    def _set_materials(self):
        '''Default materials'''
        self._be.apply_default_color_scheme(
            shading_mode=BioExplorer.SHADING_MODE_DIFFUSE, specular_exponent=50.0)

    def _set_clipping_planes(self):
        '''Clipping planes'''
        clip_planes = [
            [1.0, 0.0, 0.0, scene_size.x * 1.5 + 5],
            [-1.0, 0.0, 0.0, scene_size.x * 1.5 + 5],
            [0.0, 0.0, 1.0, scene_size.z + 5],
            [0.0, 0.0, -1.0, scene_size.z + 5]
        ]
        cps = self._core.get_clip_planes()
        ids = list()
        if cps:
            for cp in cps:
                ids.append(cp['id'])
        self._core.remove_clip_planes(ids)
        for plane in clip_planes:
            self._core.add_clip_plane(plane)

    def set_rendering_settings(self, renderer):
        if renderer == 'bio_explorer':
            self._core.set_renderer(
                background_color=[96 / 255, 125 / 255, 139 / 255],
                current=renderer, head_light=False,
                samples_per_pixel=1, subsampling=1, max_accum_frames=1)
            params = self._core.BioExplorerRendererParams()
            params.exposure = 1.0
            params.gi_samples = 1
            params.gi_weight = 0.3
            params.gi_distance = 5000
            params.shadows = 0.8
            params.soft_shadows = 0.05
            params.fog_start = 1000
            params.fog_thickness = 300
            params.max_bounces = 1
            params.use_hardware_randomizer = True
            self._core.set_renderer_params(params)

            '''Lights'''
            self._core.clear_lights()
            self._core.add_light_directional(
                angularDiameter=0.5, color=[1, 1, 1], direction=[-0.7, -0.4, -1],
                intensity=1.0, is_visible=False)

    def build_frame(self, frame):
        self._log(2, '- Resetting scene...')
        self._be.reset_scene()

        self._log(2, '- Building surfactants...')
        self._add_surfactants_d(frame)
        self._add_surfactants_a(frame)

        self._log(2, '- Building glucose...')
        self._add_glucose(frame)

        self._log(2, '- Building lactoferrins...')
        self._add_lactoferrins(frame)

        self._log(2, '- Building defensins...')
        self._add_defensins(frame)

        self._log(2, '- Building viruses...')
        self._add_viruses(frame)

        self._log(2, '- Building cell...')
        self._add_cell(frame)

        self._log(2, '- Building lymphocyte...')
        self._add_lymphocyte(frame)

        self._log(2, '- Setting materials...')
        self._set_materials()

        self._log(2, '- Showing models...')
        self._be.set_models_visibility(True)
        self._core.set_renderer()

    def render_movie(self, start_frame=0, end_frame=0, frame_step=1, frame_list=list()):
        aperture_ratio = 0.0
        cameras_key_frames = [
            {  # Virus overview (on 5th virus)
                'apertureRadius': aperture_ratio * 0.02,
                'direction': [0.0, 0.0, -1.0],
                'focusDistance': 139.56,
                'origin': [199.840, 20.634, 34.664],
                'up': [0.0, 1.0, 0.0]
            },
            {  # Protein S (on 5th virus)
                'apertureRadius': aperture_ratio * 0.02,
                'direction': [0.0, 0.0, -1.0],
                'focusDistance': 23.60,
                'origin': [195.937, 74.319, -111.767],
                'up': [0.0, 1.0, 0.0]
            },
            {  # Protein M and E
                'apertureRadius': aperture_ratio * 0.02,
                'direction': [-0.047, -0.298, -0.953],
                'focusDistance': 54.56,
                'origin': [208.156, 55.792, -59.805],
                'up': [0.003, 0.954, -0.298]
            },
            {  # Overview SPA
                'apertureRadius': aperture_ratio * 0.001,
                'direction': [-0.471, -0.006, -0.882],
                'focusDistance': 444.63,
                'origin': [238.163, 46.437, 372.585],
                'up': [0.0, 1.0, 0.0]
            },
            {  # Overview SPD
                'apertureRadius': aperture_ratio * 0.001,
                'focusDistance': 444.63,
                'direction': [-0.471, -0.005, -0.881],
                'origin': [238.163, 46.436, 372.584],
                'up': [0.0, 1.0, 0.0]
            },
            {  # Zoom SPD (on 3rd virus)
                'apertureRadius': aperture_ratio * 0.02,
                'focusDistance': 31.86,
                'direction': [-0.821, 0.202, -0.533],
                'origin': [-9.827, 110.720, 60.944],
                'up': [0.178, 0.979, 0.098]
            },
            {  # Overview scene
                'apertureRadius': aperture_ratio * 0.0,
                'focusDistance': 1.0,
                'direction': [-1.0, 0.0, 0.0],
                'origin': [1008.957, 29.057, 113.283],
                'up': [0.0, 1.0, 0.0]
            },
            {  # Cell view
                'apertureRadius': aperture_ratio * 0.0,
                'direction': [0.0, 0.0, -1.0],
                'focusDistance': 1.0,
                'origin': [150.0, -170.0, 400.0],
                'up': [0.0, 1.0, 0.0]
            }
        ]

        '''Double the frames to make it smoother'''
        key_frames = list()
        for cameras_key_frame in cameras_key_frames:
            key_frames.append(cameras_key_frame)
            key_frames.append(cameras_key_frame)

        '''Clipping planes'''
        self._set_clipping_planes()

        super().render_movie(key_frames, 250, 150, start_frame, end_frame, frame_step, frame_list)


def main(argv):
    args = MovieScenario.parse_arguments(argv)

    scenario = LowGlucoseScenario(
        hostname=args.hostname,
        port=args.port,
        projection=args.projection,
        output_folder=args.export_folder,
        image_k=args.image_resolution_k,
        image_samples_per_pixel=args.image_samples_per_pixel,
        log_level=args.log_level,
        shaders=args.shaders)

    scenario.set_rendering_settings('bio_explorer')
    scenario.render_movie(
        start_frame=args.from_frame,
        end_frame=args.to_frame,
        frame_step=args.frame_step,
        frame_list=args.frame_list)


if __name__ == "__main__":
    main(sys.argv[1:])
