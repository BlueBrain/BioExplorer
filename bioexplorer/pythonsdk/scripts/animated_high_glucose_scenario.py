#!/usr/bin/env python
"""Animated high glucose scenario"""

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

from bioexplorer import BioExplorer, RNASequence, Protein, \
    AssemblyProtein, Virus, Surfactant, ParametricMembrane, Cell, Sugars, \
    Volume, Vector2, Vector3, Quaternion
from mediamaker import MovieMaker
import math
from datetime import datetime, timedelta
import time
import sys
import argparse

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
scenario = 'high_glucose'
# Scene
scene_size = 800.0

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
add_rna = False
landing_distance = 50.0

# Immune system
nb_glucoses = 360000
nb_lactoferrins = 50
nb_defensins = 100
nb_defensins_on_virus = 2

# Cell
cell_nb_receptors = 100
cell_nb_lipids = 1200000

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

# --------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------
ROTATION_MODE_LINEAR = 0
ROTATION_MODE_SINUSOIDAL = 1


class HighGlucoseScenario():

    def __init__(self, hostname, port, projection, output_folder, image_k=4,
                 image_samples_per_pixels=64, log_level=1):
        self._log_level = log_level
        self._hostname = hostname
        self._url = hostname + ':' + str(port)
        self._be = BioExplorer(self._url)
        self._core = self._be.core_api()
        self._image_size = [1920, 1080]
        self._image_samples_per_pixels = image_samples_per_pixels
        self._image_projection = projection
        self._image_output_folder = output_folder
        self._prepare_movie(projection, image_k)
        self._log(1, '================================================================================')
        self._log(1, '- Version          : ' + self._be.version())
        self._log(1, '- URL              : ' + self._url)
        self._log(1, '- Projection       : ' + projection)
        self._log(1, '- Frame size       : ' + str(self._image_size))
        self._log(1, '- Export folder    : ' + self._image_output_folder)
        self._log(1, '- Samples per pixel: ' + str(self._image_samples_per_pixels))
        self._log(1, '================================================================================')

    def _log(self, level, message):
        if level <= self._log_level:
            print('[' + str(datetime.now()) + '] ' + message)

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
        virus_radii = [45.0, 44.0, 45.0, 43.0, 44.0, 43.0]
        virus_sequences = [
            [[-1000, 2499], [2500, 2599], [2600, 2799], [2800, 2999], [3000, 3099], [3100, 3750]],
            # Virus used for the ACE2 close-up
            [[0, 2100], [2200, 2299], [2300, 2499], [2500, 3049], [3050, 3149], [3150, 3750]],
            [[-800, 2549], [2550, 2649], [2650, 2849], [2850, 3199], [3200, 3299], [3300, 3750]],
            [[-1400, 3750], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6], [1e6, 1e6]],
            [[-400, 2599], [2600, 2699], [2700, 2899], [2900, 3119], [3120, 3219], [3220, 3750]],
            [[0, 2649], [2650, 2749], [2750, 2849], [2850, 3199], [3200, 3299], [3300, 3750]],
        ]
        virus_flights_in = [
            [Vector3(-250.0, 100.0, -70.0), Quaternion(0.519, 0.671, 0.528, -0.036),
             Vector3(-337.3, -92.3, -99.2), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            [Vector3(-50.0, 300.0, 250.0), Quaternion(0.456, 0.129, -0.185, -0.860),
             Vector3(-74.9, -99.0, 228.8), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            [Vector3(150.0, 100.0, 50.0), Quaternion(0.087, 0.971, -0.147, -0.161),
             Vector3(187.5, -110.4, 51.2), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            [Vector3(40.0, 250.0, -50), Quaternion(0.0, 0.0, 0.0, 1.0),
             Vector3(4.5,  100.0, 7.5), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            [Vector3(60.0, 100.0, -240.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             Vector3(73.9, -117.1, -190.4), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR],
            [Vector3(200.0, 100.0, 300.0), Quaternion(-0.866, 0.201, 0.308, -0.336),
             Vector3(211.5, -104.9, 339.2), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_LINEAR]
        ]

        virus_flights_out = [
            [Vector3(-250.0, -150.0, -70.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(-270.0, 200.0, -99.2), Quaternion(0.519, 0.671, 0.528, -0.036),
             ROTATION_MODE_LINEAR],
            [Vector3(-50.0, -150.0, 250.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(-75.0, 200.0, 228.8), Quaternion(0.456, 0.129, -0.185, -0.860),
             ROTATION_MODE_LINEAR],
            [Vector3(150.0, -150.0, 50.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(187.0, 200.0, 51.2), Quaternion(0.087, 0.971, -0.147, -0.161),
             ROTATION_MODE_LINEAR],
            [Vector3(40.0, -150.0, -50.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(60.0,  200.0, -30.0), Quaternion(0.0, 0.0, 0.0, 1.0),
             ROTATION_MODE_LINEAR],
            [Vector3(60.0, -150.0, -240.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(74.0, 200.0, -220.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             ROTATION_MODE_LINEAR],
            [Vector3(200.0, -150.0, 300.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             Vector3(210.0, 200.0, 330.0), Quaternion(-0.866, 0.201, 0.308, -0.336),
             ROTATION_MODE_LINEAR]
        ]

        for virus_index in range(len(virus_sequences)):
            name = 'Coronavirus ' + str(virus_index)
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

            self._be.add_coronavirus(
                name=name, resource_folder=resource_folder,
                representation=protein_representation, position=pos, rotation=rot,
                add_glycans=add_glycans,
                assembly_params=[virus_radii[virus_index], 5 * frame + 2 * virus_index,
                                 0.25, frame + 2 * virus_index + 1, 0.1, morphing_step]
            )

    def _add_cell(self, frame):

        name = 'Cell'
        nb_receptors = cell_nb_receptors
        size = scene_size * 2.0
        height = scene_size / 10.0
        position = Vector3(4.5, -186.0, 7.0)
        random_seed = 10

        nb_lipids = cell_nb_lipids
        ace2_receptor = Protein(
            sources=[pdb_folder + '6m18.pdb'], occurences=nb_receptors,
            position=Vector3(0.0, 6.0, 0.0))

        membrane = ParametricMembrane(
            sources=[
                membrane_folder + 'segA.pdb',
                membrane_folder + 'segB.pdb',
                membrane_folder + 'segC.pdb',
                membrane_folder + 'segD.pdb'
            ],
            occurences=cell_nb_lipids
        )

        cell = Cell(
            name=name, size=size, extra_parameters=[height],
            shape=BioExplorer.ASSEMBLY_SHAPE_SINUSOIDAL,
            membrane=membrane, receptor=ace2_receptor,
            random_position_seed=frame + 1, random_position_strength=0.025,
            random_rotation_seed=frame + 2, random_rotation_strength=0.2
        )

        self._be.add_cell(
            cell=cell, position=position,
            representation=protein_representation,
            random_seed=random_seed)

        '''Modify receptor position when attached virus enters the cell'''
        receptors_instances = [90, 23, 24, 98, 37, 44]
        receptors_sequences = [[1000, 1099], [2200, 2299], [
            1200, 1299], [1600, 1699], [2000, 2099], [-1, -1]]

        for i in range(len(receptors_instances)):
            instance_index = receptors_instances[i]
            sequence = receptors_sequences[i]
            start_frame = sequence[0]
            end_frame = sequence[1]
            if frame >= start_frame:
                if frame > end_frame:
                    '''Send receptor to outter space'''
                    status = self._be.set_protein_instance_transformation(
                        assembly_name=name, name=name + '_' + BioExplorer.NAME_RECEPTOR,
                        instance_index=instance_index, position=Vector3(0.0, 1e6, 0.0)
                    )
                else:
                    '''Current receptor transformation'''
                    transformation = self._be.get_protein_instance_transformation(
                        assembly_name=name, name=name + '_' + BioExplorer.NAME_RECEPTOR,
                        instance_index=instance_index
                    )
                    p = transformation['position'].split(',')
                    q = transformation['rotation'].split(',')
                    pos = Vector3(float(p[0]), float(p[1]), float(p[2]))
                    q2 = Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

                    '''Bend receptor'''
                    progress = (frame - start_frame) * 1.0 / (end_frame - start_frame)
                    q1 = Quaternion(axis=[0, 1, 0], angle=math.pi * progress)
                    rot = q2 * q1

                    pos.x += landing_distance * progress * 0.3
                    pos.y -= landing_distance * progress * 0.3

                    status = self._be.set_protein_instance_transformation(
                        assembly_name=name, name=name + '_' + BioExplorer.NAME_RECEPTOR,
                        instance_index=instance_index, position=pos, rotation=rot
                    )

        '''Glycans'''
        if nb_receptors != 0 and add_glycans:
            self._be.add_multiple_glycans(
                representation=glycan_representation, assembly_name=name,
                glycan_type=BioExplorer.NAME_GLYCAN_COMPLEX,
                protein_name=BioExplorer.NAME_RECEPTOR, paths=complex_paths,
                indices=[53, 90, 103, 322, 432, 690],
                assembly_params=[0, 0, 0.0, frame + 3, 0.2]
            )
            self._be.add_multiple_glycans(
                representation=glycan_representation, assembly_name=name,
                glycan_type=BioExplorer.NAME_GLYCAN_HYBRID,
                protein_name=BioExplorer.NAME_RECEPTOR, paths=hybrid_paths,
                indices=[546],
                assembly_params=[0, 0, 0.0, frame + 4, 0.2])

            indices = [[155, Quaternion(0.707, 0.0, 0.707, 0.0)],
                       [730, Quaternion(0.707, 0.0, 0.707, 0.0)]]
            count = 0
            for index in indices:
                o_glycan_name = name + '_' + BioExplorer.NAME_GLYCAN_O_GLYCAN + '_' + str(index[0])
                o_glycan = Sugars(
                    assembly_name=name, name=o_glycan_name, source=o_glycan_paths[0],
                    protein_name=name + '_' + BioExplorer.NAME_RECEPTOR, representation=glycan_representation,
                    chain_ids=[2, 4], site_indices=[index[0]], rotation=index[1],
                    assembly_params=[0, 0, 0.0, frame + count + 5, 0.2])
                self._be.add_sugars(o_glycan)
                count += 1

    def _add_surfactant_d(self, name, position, rotation, random_seed):
        surfactant_d = Surfactant(
            name=name, surfactant_protein=BioExplorer.SURFACTANT_PROTEIN_D,
            head_source=surfactant_head_source,
            branch_source=surfactant_branch_source)
        self._be.add_surfactant(
            surfactant=surfactant_d, representation=protein_representation,
            position=position, rotation=rotation, random_seed=random_seed)

    def _add_surfactant_a(self, name, position, rotation, random_seed):
        surfactant_a = Surfactant(
            name=name, surfactant_protein=BioExplorer.SURFACTANT_PROTEIN_A,
            head_source=surfactant_head_source,
            branch_source=surfactant_branch_source)
        self._be.add_surfactant(
            surfactant=surfactant_a, representation=protein_representation,
            position=position, rotation=rotation, random_seed=random_seed)

    def _add_glucose_to_surfactant_head(self, name):
        for index in [321, 323]:
            glucose_name = name + '_' + BioExplorer.NAME_GLUCOSE + '_' + str(index)
            glucose = Sugars(
                assembly_name=name, name=glucose_name, source=glucose_path,
                protein_name=name + '_' + BioExplorer.NAME_SURFACTANT_HEAD,
                representation=glycan_representation, site_indices=[index])
            self._be.add_sugars(glucose)

    def _add_surfactants_d(self, frame):
        spd_sequences = [[-1550, 3750], [0, 3750], [0, 3750]]
        spd_random_seeds = [1, 2, 6]
        spd_flights = [
            [Vector3(-340.0, 0.0, -100.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             Vector3(74.0 + (74.0 + 340), 24.0 + (24.0 - 0.0), -45.0 +
                     (-45.0 + 100)), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_SINUSOIDAL],
            [Vector3(-200, 0.0, -200.0), Quaternion(0.087, 0.971, -0.147, -0.161),
             Vector3(304.0, 75.0, -100.0), Quaternion(1.0, 0.0, 0.0, 0.0),
             ROTATION_MODE_SINUSOIDAL],
            [Vector3(-460.0, 50.0, 0.0), Quaternion(0.519, 0.671, 0.528, -0.036),
             Vector3(160.0, -50.0, -50.0), Quaternion(1.0, 0.0, 0.0, 0.0),
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
                random_seed=spd_random_seeds[surfactant_index])
            self._add_glucose_to_surfactant_head(name=name)

    def _add_surfactants_a(self, frame):
        spa_sequences = [[0, 3750]]
        spa_random_seeds = [2]
        spa_frames = [
            [Vector3(-400.0, -100.0, 100.0), Quaternion(-0.095, 0.652, -0.326, 0.677),
             Vector3(250.0, -50.0, 100.0), Quaternion(1.0, 0.0, 0.0, 0.0),
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
                random_seed=spa_random_seeds[surfactant_index])
            self._add_glucose_to_surfactant_head(name=name)

    def _add_glucose(self, frame):
        glucose = Protein(
            sources=[glucose_path], load_non_polymer_chemicals=True,
            occurences=nb_glucoses)
        volume = Volume(
            name=BioExplorer.NAME_GLUCOSE, size=scene_size, protein=glucose,
            random_position_seed=frame + 20, random_position_stength=scene_size / 600.0,
            random_rotation_seed=frame + 21, random_rotation_stength=0.3
        )
        status = self._be.add_volume(
            volume=volume, representation=protein_representation,
            position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0),
            random_seed=100)

    def _add_lactoferrins(self, frame):
        lactoferrin = Protein(
            sources=[lactoferrin_path], load_non_polymer_chemicals=True,
            occurences=nb_lactoferrins)
        lactoferrins_volume = Volume(
            name=BioExplorer.NAME_LACTOFERRIN, size=scene_size, protein=lactoferrin,
            random_position_seed=frame + 30, random_position_stength=scene_size / 400.0,
            random_rotation_seed=frame + 31, random_rotation_stength=0.3
        )
        status = self._be.add_volume(
            volume=lactoferrins_volume, representation=protein_representation,
            position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0),
            random_seed=101)

    def _add_defensins(self, frame):
        defensin = Protein(
            sources=[defensin_path], load_non_polymer_chemicals=True,
            occurences=nb_defensins)
        defensins_volume = Volume(
            name=BioExplorer.NAME_DEFENSIN, size=scene_size, protein=defensin,
            random_position_seed=frame + 40, random_position_stength=scene_size / 400.0,
            random_rotation_seed=frame + 41, random_rotation_stength=0.3
        )
        status = self._be.add_volume(
            volume=defensins_volume, representation=protein_representation,
            position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0),
            random_seed=102)

    def _set_materials(self):
        '''Default materials'''
        self._be.apply_default_color_scheme(
            shading_mode=BioExplorer.SHADING_MODE_DIFFUSE, specular_exponent=50.0)

    def _set_rendering_settings(self):
        '''Renderer'''
        status = self._core.set_renderer(
            background_color=[96 / 255, 125 / 255, 139 / 255],
            current='bio_explorer', head_light=True,
            samples_per_pixel=1, subsampling=1, max_accum_frames=self._image_samples_per_pixels)
        params = self._core.BioExplorerRendererParams()
        params.exposure = 1.0
        params.gi_samples = 1
        params.gi_weight = 0.3
        params.gi_distance = 5000
        params.shadows = 1.0
        params.soft_shadows = 0.02
        params.fog_start = 1000
        params.fog_thickness = 300
        params.max_bounces = 1
        params.use_hardware_randomizer = True
        status = self._core.set_renderer_params(params)

        '''Lights'''
        status = self._core.clear_lights()
        status = self._core.add_light_directional(
            angularDiameter=0.5, color=[1, 1, 1], direction=[-0.7, -0.4, -1],
            intensity=1.0, is_visible=False
        )

        '''Camera'''
        status = self._core.set_camera(current='bio_explorer_perspective')

    def _build_frame(self, frame):
        self._log(2, '- Resetting scene...')
        self._be.reset()

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

        self._log(2, '- Setting materials...')
        self._set_materials()

        self._log(2, '- Showing models...')
        status = self._be.set_models_visibility(True)
        status = self._core.set_renderer()

    def _make_export_folder(self):
        import os
        command_line = 'mkdir -p ' + self._image_output_folder
        os.system(command_line)
        command_line = 'ls ' + self._image_output_folder
        if os.system(command_line) != 0:
            self._log(3, 'ERROR: Failed to create output folder')

    def _prepare_movie(self, projection, image_k):
        if projection == 'perspective':
            aperture_ratio = 1.0
            self._image_size = [image_k*960, image_k*540]
            self._core.set_camera(current='bio_explorer_perspective')
        elif projection == 'fisheye':
            self._image_size = [int(image_k*1024), int(image_k*1024)]
            self._core.set_camera(current='fisheye')
        elif projection == 'panoramic':
            self._image_size = [int(image_k*1024), int(image_k*1024)]
            self._core.set_camera(current='panoramic')
        elif projection == 'opendeck':
            self._image_size = [7*2160, 3840]
            self._core.set_camera(current='cylindric')

        self._image_output_folder = self._image_output_folder + '/' + \
            projection + '/' + str(self._image_size[0]) + 'x' + str(self._image_size[1])
        self._make_export_folder()

    def _set_clipping_planes(self):
        '''Clipping planes'''
        clip_planes = [
            [1.0, 0.0, 0.0, scene_size * 1.5 + 5],
            [-1.0, 0.0, 0.0, scene_size * 1.5 + 5],
            [0.0, 0.0, 1.0, scene_size + 5],
            [0.0, 0.0, -1.0, scene_size + 5]
        ]
        cps = self._core.get_clip_planes()
        ids = list()
        if cps:
            for cp in cps:
                ids.append(cp['id'])
        self._core.remove_clip_planes(ids)
        for plane in clip_planes:
            self._core.add_clip_plane(plane)

    def render_movie(self, start_frame=0, end_frame=0, frame_step=1, frame_list=list()):
        '''Accelerate loading by not showing models as they are loaded'''
        status = self._be.set_general_settings(model_visibility_on_creation=False)

        aperture_ratio = 0.0
        cameras_key_frames = [
            {  # 1. Cell view (frame 0)
                'apertureRadius': aperture_ratio * 0.0,
                'direction': [0.0, 0.0, -1.0],
                'focusDistance': 1.0,
                'origin': [150.0, -170.0, 400.0],
                'up': [0.0, 1.0, 0.0]
            },
            {  # 2. Virus view (frame 500)
                'apertureRadius': aperture_ratio * 0.0,
                'direction': [0.0, 0.0, -1.0],
                'focusDistance': 449.50,
                'origin': [-67.501, -17.451, 254.786],
                'up': [0.0, 1.0, 0.0]
            },
            {  # 3. Surfactant Head (frame 1000)
                'apertureRadius': aperture_ratio * 0.02,
                'direction': [0.276, -0.049, -0.959],
                'focusDistance': 25.54,
                'origin': [38.749, 35.228, 5.536],
                'up': [0.0, 1.0, 0.0]
            },
            {  # 4. Virus overview (frame 1500)
                'apertureRadius': aperture_ratio * 0.0,
                'direction': [0.009, 0.055, -0.998],
                'focusDistance': 109.59,
                'origin': [-0.832, 72.134, 387.389],
                'up': [0.017, 0.998, 0.055]
            },
            {  # 5. ACE2 receptor (frame 2000)
                'apertureRadius': aperture_ratio * 0.02,
                'direction': [-0.436, 0.035, -0.898],
                'focusDistance': 62.17,
                'origin': [-33.619, -164.994, 276.296],
                'up': [0.011, 0.999, 0.033]
            },
            {  # 6. Membrane overview (frame 2500)
                'apertureRadius': aperture_ratio * 0.0,
                'direction': [0.009, 0.055, -0.998],
                'focusDistance': 1.0,
                'origin': [0.293, 19.604, 1000],
                'up': [0.017, 0.998, 0.055]
            },
            {  # 7. (frame 3000)
                'apertureRadius': aperture_ratio * 0.0,
                'focusDistance': 1.0,
                'direction': [0.009, 0.055, -0.998],
                'origin': [0.293, 19.604, 1000],
                'up': [0.017, 0.998, 0.055]
            },
            {  # 8. (frame 3500)
                'apertureRadius': aperture_ratio * 0.0,
                'focusDistance': 60,
                'direction': [0.009, 0.055, -0.998],
                'origin': [0.293, 19.604, 1000],
                'up': [0.017, 0.998, 0.055]
            }
        ]

        '''Double the frames to make it smoother'''
        key_frames = list()
        for cameras_key_frame in cameras_key_frames:
            key_frames.append(cameras_key_frame)
            key_frames.append(cameras_key_frame)

        mm = MovieMaker(self._be)
        mm.build_camera_path(key_frames, 250, 150)
        self._log(1, '- Total number of frames: %d' % mm.get_nb_frames())

        self._core.set_application_parameters(viewport=self._image_size)
        self._core.set_application_parameters(image_stream_fps=0)

        frames_to_render = list()
        if len(frame_list) != 0:
            frames_to_render = frame_list
        else:
            if end_frame == 0:
                end_frame = mm.get_nb_frames()
            for i in range(start_frame, end_frame + 1, frame_step):
                frames_to_render.append(i)

        cumulated_rendering_time = 0
        nb_frames = len(frames_to_render)
        frame_count = 1

        '''Rendering settings'''
        self._set_rendering_settings()

        '''Clipping planes'''
        self._set_clipping_planes()

        '''Frames'''
        for frame in frames_to_render:
            try:
                start = time.time()
                self._log(1, '- Rendering frame %i (%i/%i)' % (frame, frame_count, nb_frames))
                self._log(1, '------------------------------')
                self._build_frame(frame)
                mm.set_current_frame(
                    frame=frame, camera_params=self._core.BioExplorerPerspectiveCameraParams())
                mm.create_snapshot(
                    size=self._image_size,
                    path=self._image_output_folder, base_name='%05d' % frame,
                    samples_per_pixel=self._image_samples_per_pixels)
                end = time.time()

                rendering_time = end - start
                cumulated_rendering_time += rendering_time
                average_rendering_time = cumulated_rendering_time / frame_count
                remaining_rendering_time = (nb_frames - frame_count) * average_rendering_time
                self._log(1, '------------------------------')
                self._log(1, 'Frame %i successfully rendered in %i seconds' %
                          (frame, rendering_time))

                hours = math.floor(remaining_rendering_time / 3600)
                minutes = math.floor((remaining_rendering_time - hours * 3600) / 60)
                seconds = math.floor(remaining_rendering_time - hours * 3600 - minutes * 60)

                expected_end_time = datetime.now() + timedelta(seconds=remaining_rendering_time)
                self._log(1, 'Estimated remaining time: %i hours, %i minutes, %i seconds' %
                          (hours, minutes, seconds))
                self._log(1, 'Expected end time       : %s' % expected_end_time)
                self._log(
                    1, '--------------------------------------------------------------------------------')
                frame_count += 1
            except Exception as e:
                self._log(1, 'ERROR: Failed to render frame %i' % frame)
                self._log(1, str(e))
                time.sleep(10)
                self._be = BioExplorer(self._url)
                self._core = self._be.core_api()
                mm = MovieMaker(self._be)

        self._core.set_application_parameters(image_stream_fps=20)
        self._log(1, 'Movie rendered, live long and prosper \V/')


def main(argv):
    parser = argparse.ArgumentParser(description='Missing frames')
    parser.add_argument('-e', '--export_folder', help='Export folder', type=str, default='/tmp')
    parser.add_argument('-n', '--hostname',
                        help='BioExplorer server hostname', type=str, default='localhost')
    parser.add_argument('-p', '--port',
                        help='BioExplorer server port', type=int, default=5000)
    parser.add_argument('-j', '--projection', help='Camera projection',
                        type=str, default='perspective',
                        choices=['perspective', 'fisheye', 'panoramic', 'opendeck'])
    parser.add_argument('-k', '--image_resolution_k',
                        help='Image resolution in K', type=int, default=4)
    parser.add_argument('-s', '--image_samples_per_pixel',
                        help='Image samples per pixel', type=int, default=64)
    parser.add_argument('-f', '--from_frame', type=int, help='Start frame', default=0)
    parser.add_argument('-t', '--to_frame', type=int, help='End frame', default=0)
    parser.add_argument('-m', '--frame_step', type=int, help='Frame step', default=1)
    parser.add_argument('-g', '--log-level', type=int, help='Frame step', default=1)
    parser.add_argument('-l', '--frame-list', type=int, nargs='*',
                        help='List of frames to render', default=list())
    args = parser.parse_args(argv)

    scenario = HighGlucoseScenario(
        hostname=args.hostname,
        port=args.port,
        projection=args.projection,
        output_folder=args.export_folder,
        image_k=args.image_resolution_k,
        image_samples_per_pixels=args.image_samples_per_pixel,
        log_level=args.log_level)

    scenario.render_movie(
        start_frame=args.from_frame,
        end_frame=args.to_frame,
        frame_step=args.frame_step,
        frame_list=args.frame_list)


if __name__ == "__main__":
    main(sys.argv[1:])
