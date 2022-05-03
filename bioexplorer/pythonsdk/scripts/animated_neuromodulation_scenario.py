#!/usr/bin/env python
"""Animated neuromodulation scenario"""

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

from bioexplorer import BioExplorer, Protein, AnimationParams, Volume, Vector3, MovieScenario
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm.notebook import tqdm
import seaborn as sns
import os
import sys

resource_folder = './tests/test_files/'
pdb_folder = os.path.join(resource_folder + 'pdb')
acetylcholin_path = os.path.join(pdb_folder, 'neuromodulation', 'acetylcholin.pdb')

''' Protein representation '''
representation = BioExplorer.REPRESENTATION_ATOMS
atom_radius_multiplier = 2.0

'''Scene information'''
scale = Vector3(1000, 1000, 1000)

class NeuromodulationScenario(MovieScenario):

    def __init__(self, hostname, port, projection, output_folder, image_k=4,
                 image_samples_per_pixel=64, log_level=1, shaders=list(['bio_explorer']),
                 nb_frames_between_keys=200, draft=False, gi_length=1e6):
        super().__init__(hostname, port, projection, output_folder,
                         image_k, image_samples_per_pixel, log_level, False, shaders,
                         draft, gi_length)

        db_host = os.environ['DB_HOST']
        db_name = os.environ['DB_NAME']
        db_user = os.environ['DB_USER']
        db_password = os.environ['DB_PASSWORD']
        self._neuron_population_name = 'o1'
        self._vasculature_population_name = 'vasculature'
        self._astrocytes_population_name = 'astrocytes'
        self._varicosity = Vector3()
        self._synapse = Vector3()

        self._nb_frames_between_keys = nb_frames_between_keys
        self._diffusion_1_nb_molecules = 1500000
        self._diffusion_1_start_frame = self._nb_frames_between_keys * 2
        self._diffusion_1_nb_frames = int(self._nb_frames_between_keys * 2.2)

        self._diffusion_2_nb_molecules = 500000
        self._diffusion_2_start_frame = int(self._nb_frames_between_keys * 6.1)
        self._diffusion_2_nb_frames = int(self._nb_frames_between_keys * 2.5)


        db_connection_string = 'postgresql+psycopg2://%s:%s@%s:5432/%s' % (db_user, db_password, db_host, db_name)
        print('Connection string: ' + db_connection_string + ', schema: ' + self._neuron_population_name)

        self._engine = create_engine(db_connection_string)
        self._db_connection = self._engine.connect()

    def _add_presynaptic_neuron(self):
        presynaptic_neuron_id = 47211

        presynaptic_assembly_name = 'PreSynaptic Neuron'
        self._check(self._be.remove_assembly(presynaptic_assembly_name))
        self._check(self._be.add_assembly(presynaptic_assembly_name))
        self._check(self._be.add_neurons(
            assembly_name=presynaptic_assembly_name,
            population_name=self._neuron_population_name,
            geometry_quality=self._be.GEOMETRY_QUALITY_MEDIUM,
            morphology_color_scheme=self._be.MORPHOLOGY_COLOR_SCHEME_SECTION,
            use_sdf=True, load_synapses=True, generate_varicosities=True,
            load_somas=True, load_axon=True, show_membrane=True,
            load_basal_dendrites=True, load_apical_dendrites=True,
            generate_internals=False, generate_externals=False,
            sql_node_filter='guid=%d' % presynaptic_neuron_id,
            scale=scale
        ))

        varicosities = self._be.get_neuron_varicosities(
            assembly_name=presynaptic_assembly_name,
            neuron_guid=presynaptic_neuron_id)

        self._varicosity = Vector3(
            varicosities[0][0],
            varicosities[0][1],
            varicosities[0][2]
        )
        self._synapse = Vector3(
            varicosities[4][0] + 0.25,
            varicosities[4][1],
            varicosities[4][2]
        )        

    def _add_postsynaptic_neuron_1(self):
        postsynaptic_1_neuron_id = 49
        postsynaptic_1_assembly_name = 'PostSynaptic Neuron 1'
        self._check(self._be.remove_assembly(postsynaptic_1_assembly_name))
        self._check(self._be.add_assembly(postsynaptic_1_assembly_name))

        with Session(self._engine) as session:
            data = session.execute('SELECT x,y,z FROM %s.node WHERE guid=%d' % (
                self._neuron_population_name, postsynaptic_1_neuron_id))
            soma_position = data.all()[0]

        self._check(self._be.add_neurons(
            assembly_name=postsynaptic_1_assembly_name,
            population_name=self._neuron_population_name,
            geometry_quality=self._be.GEOMETRY_QUALITY_MEDIUM,
            morphology_color_scheme=self._be.MORPHOLOGY_COLOR_SCHEME_SECTION,
            use_sdf=True, load_synapses=True, generate_varicosities=False,
            load_somas=True, load_axon=False, show_membrane=True,
            load_basal_dendrites=True, load_apical_dendrites=True,
            generate_internals=False, generate_externals=False,
            sql_node_filter='guid=%d' % postsynaptic_1_neuron_id,
            scale=scale
        ))

        model_ids = self._be.get_model_ids()['ids']
        model_id = model_ids[len(model_ids)-1]
        tf = {
            'rotation': [0.0, 0.0, 0.0, 1.0], 
            'rotation_center': [0.0, 0.0, 0.0], 
            'scale': [1.0, 1.0, 1.0], 
            'translation': [
                scale.x * (self._varicosity.x - soma_position[0] + 1.2),
                scale.y * (self._varicosity.y - soma_position[1] - 0.9),
                scale.z * (self._varicosity.z - soma_position[2] + 1.093 - 7.5),
            ]
        }
        self._core.update_model(id=model_id, transformation=tf)

    def _add_postsynaptic_neuron_2(self):
        postsynaptic_2_neuron_id = 47211
        postsynaptic_2_assembly_name = 'PostSynaptic Neuron 2'
        self._check(self._be.remove_assembly(postsynaptic_2_assembly_name))
        self._check(self._be.add_assembly(postsynaptic_2_assembly_name))
        self._check(self._be.add_neurons(
            assembly_name=postsynaptic_2_assembly_name,
            population_name=self._neuron_population_name,
            geometry_quality=self._be.GEOMETRY_QUALITY_MEDIUM,
            morphology_color_scheme=self._be.MORPHOLOGY_COLOR_SCHEME_SECTION,
            use_sdf=True, load_synapses=True, generate_varicosities=True,
            load_somas=True, load_axon=False, show_membrane=True,
            load_basal_dendrites=True, load_apical_dendrites=True,
            generate_internals=False, generate_externals=False,
            sql_node_filter='guid=%d' % postsynaptic_2_neuron_id,
            scale=scale
        ))

        model_ids = self._be.get_model_ids()['ids']
        model_id = model_ids[len(model_ids)-1]
        tf = {
            'rotation': [0.0, 0.0, 0.0, 1.0], 
            'rotation_center': [0.0, 0.0, 0.0], 
            'scale': [1.0, 1.0, 1.0], 
            'translation': [scale.x * 37.5, scale.y * 47, scale.z * -13.5],
        }
        self._core.update_model(id=model_id, transformation=tf)

    def _add_vasculature(self):
        vasculature_assembly_name = 'Vasculature'
        self._check(self._be.remove_assembly(vasculature_assembly_name))
        self._check(self._be.add_assembly(vasculature_assembly_name))
        self._check(self._be.add_vasculature(
            assembly_name=vasculature_assembly_name,
            population_name=self._vasculature_population_name,
            quality=self._be.VASCULATURE_QUALITY_MEDIUM,
            use_sdf=True,
            sql_filter='sqrt(pow(x - %f, 2) + pow(y - %f, 2) + pow(z - %f, 2)) < 200' % (
                self._varicosity.x, self._varicosity.y, self._varicosity.z),
            scale=scale
        ))

    def _add_astrocytes(self):
        astrocytes_assembly_name = 'Astrocytes'
        self._check(self._be.remove_assembly(astrocytes_assembly_name))
        self._check(self._be.add_assembly(astrocytes_assembly_name))
        self._check(self._be.add_astrocytes(
            assembly_name=astrocytes_assembly_name,
            population_name=self._astrocytes_population_name,
            vasculature_population_name=self._vasculature_population_name,
            radius_multiplier=0.5,
            use_sdf=True, generate_internals=True,
            load_somas=True, load_dendrites=True,
            sql_filter='sqrt(pow(x - %f, 2) + pow(y - %f, 2) + pow(z - %f, 2)) < 200 AND guid%%3=1' % (
                self._varicosity.x, self._varicosity.y, self._varicosity.z),
            scale=scale))

    def _add_acetylcholin(self, position, radius, nb_molecules, frame=0):
        acetylcholin_assembly_name = 'Acetylcholin'
        acetylcholin_name = 'Acetylcholin'
        self._check(self._be.remove_assembly(acetylcholin_assembly_name))
        if radius == 0:
            return

        scaled_position = Vector3(position.x * scale.x, position.y * scale.y, position.z * scale.z)

        acetylcholin = Protein(
            name=acetylcholin_name, 
            source=acetylcholin_path,
            load_non_polymer_chemicals=True, load_bonds=True, load_hydrogen=True,
            occurences=nb_molecules,
            animation_params=AnimationParams(3, (frame + 1) * 2, 0.5, (frame + 2) * 2, 1.0)
        )
        volume = Volume(
            name=acetylcholin_assembly_name,
            shape=self._be.ASSEMBLY_SHAPE_FILLED_SPHERE,
            shape_params=Vector3(radius * scale.x, 0.0, 0.0),
            protein=acetylcholin)
        self._check(self._be.add_volume(
            volume=volume, 
            representation=representation,
            position=scaled_position,
            atom_radius_multiplier=atom_radius_multiplier))

        model_ids = self._be.get_model_ids()['ids']
        model_id = model_ids[len(model_ids)-1]
        material_ids = self._be.get_material_ids(model_id)['ids']
        self._set_morphology_materials(
            model_id, 'Greys', len(material_ids),
            self._be.SHADING_MODE_NONE, 1.0, 1.0)

    def _set_morphology_materials(self, model_id, palette_name, palette_size, shading_mode, glossiness=1.0, emission=0.0, user_param=1.0):
        colors = list()
        opacities = list()
        refraction_indices = list()
        specular_exponents = list()
        shading_modes = list()
        user_params = list()
        glossinesses = list()
        emissions = list()
        
        material_ids = self._be.get_material_ids(model_id)['ids']
        palette_size = len(material_ids)
        palette = sns.color_palette(palette_name, palette_size)

        for material_id in material_ids:
            mid = material_id % palette_size
            if mid in [self._be.NEURON_MATERIAL_AFFERENT_SYNPASE, self._be.NEURON_MATERIAL_EFFERENT_SYNPASE]:
                colors.append(palette[1])
                opacities.append(1.0)
                shading_modes.append(shading_mode)
                glossinesses.append(glossiness)
                specular_exponents.append(5.0)
            elif mid in [self._be.NEURON_MATERIAL_VARICOSITY]:
                colors.append(palette[min(2, palette_size-1)])
                opacities.append(1.0)
                shading_modes.append(shading_mode)
                glossinesses.append(glossiness)
                specular_exponents.append(5.0)
            elif mid == self._be.NEURON_MATERIAL_MITOCHONDRION:
                colors.append([0.5, 0.1, 0.6])
                opacities.append(1.0)
                shading_modes.append(shading_mode)
                glossinesses.append(glossiness)
                specular_exponents.append(6.0)
            elif mid == self._be.NEURON_MATERIAL_NUCLEUS:
                colors.append([1.0, 1.0, 1.0])
                opacities.append(1.0)
                shading_modes.append(shading_mode)
                glossinesses.append(glossiness)
                specular_exponents.append(30.0)
            elif mid == self._be.NEURON_MATERIAL_SOMA:
                colors.append(palette[0])
                opacities.append(1.0)
                shading_modes.append(shading_mode)
                glossinesses.append(glossiness)
                specular_exponents.append(5.0)
            elif mid == self._be.NEURON_MATERIAL_MYELIN_STEATH:
                colors.append([0.4, 0.3, 0.5])
                opacities.append(1.0)
                shading_modes.append(shading_mode)
                glossinesses.append(glossiness)
                specular_exponents.append(50.0)
            else:
                # Membrane
                colors.append(palette[0])
                opacities.append(1.0)
                shading_modes.append(shading_mode)
                glossinesses.append(glossiness)
                specular_exponents.append(5.0)
                
            refraction_indices.append(1.0)
            emissions.append(emission)
            user_params.append(user_param)
            
        self._check(self._be.set_materials(
            model_ids=[model_id], material_ids=material_ids,
            diffuse_colors=colors, specular_colors=colors,
            opacities=opacities, refraction_indices=refraction_indices,
            shading_modes=shading_modes, specular_exponents=specular_exponents,
            user_parameters=user_params, glossinesses=glossinesses,
            emissions=emissions
        ))

    def _set_materials(self):
        palettes = ['GnBu_r', 'PuRd', 'GnBu', 'Reds', 'Wistia', 'Greys']
        model_ids = self._be.get_model_ids()['ids']
        i = 0
        user_param = 0.1 / scale.x
        for model_id in model_ids:
            emission = 0.0
            if i==5:
                continue
            if i==3:
                user_param = 0.05 / scale.x
            if self._draft:
                self._set_morphology_materials(
                    model_id, palettes[i], self._be.NB_MATERIALS_PER_MORPHOLOGY,
                    self._be.SHADING_MODE_CARTOON, 1.0, emission, 3.0)
            else:
                self._set_morphology_materials(
                    model_id, palettes[i], self._be.NB_MATERIALS_PER_MORPHOLOGY,
                    self._be.SHADING_MODE_PERLIN, 0.1, emission, user_param)
            i += 1

    def set_rendering_settings(self, renderer):
        if renderer == 'bio_explorer':
            self._core.set_renderer(
                background_color=[0.18, 0.43, 0.41],
                current=renderer,subsampling=4, max_accum_frames=64)
            params = self._core.BioExplorerRendererParams()
            params.fog_start = scale.x
            params.fog_thickness = 300.0 * scale.x
            params.gi_samples = 0
            params.gi_weight = 0.2
            params.gi_distance = 0.25 * scale.x
            params.shadows = 0.0
            params.soft_shadows = 1.0
            params.epsilon_factor = 10.0
            params.max_bounces = 3
            params.show_background = False
            params.use_hardware_randomizer=False
            self._core.set_renderer_params(params)

    def setup_scene(self):
        self._check(self._be.set_general_settings(model_visibility_on_creation=False))
        self._check(self._be.reset_scene())
        self._log(1, 'Loading presynaptic_neuron...')
        self._add_presynaptic_neuron()
        self._log(1, 'Loading postsynaptic_neuron 1...')
        self._add_postsynaptic_neuron_1()
        self._log(1, 'Loading postsynaptic_neuron 2...')
        self._add_postsynaptic_neuron_2()
        self._log(1, 'Loading vasculature...')
        self._add_vasculature()
        self._log(1, 'Loading astrocytes...')
        self._add_astrocytes()
        self._log(1, 'Applying materials...')
        self._set_materials()
        self._log(1, 'Building geometry...')
        self._check(self._be.set_models_visibility(True))
        self._log(1, 'Diffusion 1: %d -> %d' % (
            self._diffusion_1_start_frame,
            self._diffusion_1_start_frame + self._diffusion_1_nb_frames))
        self._log(1, 'Diffusion 2: %d -> %d' % (
            self._diffusion_2_start_frame,
            self._diffusion_2_start_frame + self._diffusion_2_nb_frames))
        self._log(1, 'Done')

    def build_frame(self, frame):
        if frame >= self._diffusion_1_start_frame and \
            frame<self._diffusion_1_start_frame + self._diffusion_1_nb_frames:
            radius = 0.55 + 1.5 * float(frame - self._diffusion_1_start_frame) / float(self._diffusion_1_nb_frames)
            self._log(1, 'Diffusion 1 : %f' % radius)
            self._add_acetylcholin(
                self._varicosity, radius,
                self._diffusion_1_nb_molecules, 
                frame)
            self._check(self._be.set_models_visibility(True))
        elif frame >= self._diffusion_1_start_frame + self._diffusion_1_nb_frames and \
            frame < self._diffusion_2_start_frame:
            self._add_acetylcholin(self._synapse, 0.0, 0, 0)
        elif frame >= self._diffusion_2_start_frame and \
            frame < self._diffusion_2_start_frame + self._diffusion_2_nb_frames:
            radius = 0.1 + 1.0 * float(frame - self._diffusion_2_start_frame) / float(self._diffusion_2_nb_frames)
            self._log(1, 'Diffusion 2 : %f' % radius)
            self._add_acetylcholin(
                self._synapse, radius,
                self._diffusion_2_nb_molecules,
                frame)
            self._check(self._be.set_models_visibility(True))
        self._log(1, 'Done')

    def render_movie(self, start_frame=0, end_frame=0, frame_step=1, frame_list=list()):
        aperture = 0.0
        focus = 1000000.0
        cameras_key_frames = [
            {
                'apertureRadius': aperture,
                'direction': [-0.8567466019896534, -0.2935542128559732, -0.42404148864668023],
                'focusDistance': focus,
                'origin': [226503.11469778, 1142396.388033219, 429603.3430434604],
                'up': [-0.23865856208548414, 0.9545366945220539, -0.1786107207146317]
            },
            {
                'apertureRadius': aperture,
                'direction': [-0.8774240248499731, -0.28853039754422877, -0.3832457309730182],
                'focusDistance': focus,
                'origin': [81533.66171824484, 1092225.2105497972, 342254.1077413495],
                'up': [-0.24885532118265122, 0.9567656315113919, -0.15056744494639906]
            }, 
            {
                'apertureRadius': aperture,
                'direction': [-1.0, 0.0, -2.220446049250313e-16],
                'focusDistance': focus,
                'origin': [65388.56508492622, 1085758.185925619, 333209.7135301765],
                'up': [0.0, 1.0, 0.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [-1.0, 0.0, -2.220446049250313e-16],
                'focusDistance': focus,
                'origin': [65388.56508492622, 1085758.185925619, 333209.7135301765],
                'up': [0.0, 1.0, 0.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [-1.0, 0.0, -2.220446049250313e-16],
                'focusDistance': focus,
                'origin': [65388.56508492622, 1085758.185925619, 333209.7135301765],
                'up': [0.0, 1.0, 0.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [0.24494364376278907, 0.1756421722455308, -0.9534948551036231],
                'focusDistance': focus,
                'origin': [35140, 1063763, 354702],
                'up': [0.011431871571080332, 0.982861900325922, 0.1839885789936418]
            }, 
            {
                'apertureRadius': aperture,
                'direction': [0.2449436437721276, 0.17564217226494977, -0.953494855097647],
                'focusDistance': focus,
                'origin': [39734.57699805979, 1077057.9739209255, 336820.2550883301],
                'up': [0.011431871566109829, 0.9828619003223579, 0.18398857901299054]
            },
            {
                'apertureRadius': aperture,
                'direction': [0.2449436437721276, 0.17564217226494977, -0.953494855097647],
                'focusDistance': focus,
                'origin': [39734.57699805979, 1077057.9739209255, 336820.2550883301],
                'up': [0.011431871566109829, 0.9828619003223579, 0.18398857901299054]
            },
            {
                'apertureRadius': aperture,
                'direction': [0.2449436437721276, 0.17564217226494977, -0.953494855097647],
                'focusDistance': focus,
                'origin': [39734.57699805979, 1077057.9739209255, 336820.2550883301],
                'up': [0.011431871566109829, 0.9828619003223579, 0.18398857901299054]
            },
            {
                'apertureRadius': aperture,
                'direction': [0.2449436437483312, 0.17564217224481946, -0.9534948551074682],
                'focusDistance': focus,
                'origin': [4741.122617675595, 1051965.1556396896, 473039.665412839],
                'up': [0.011431871571465562, 0.9828619003261981, 0.18398857899214302]
            }
        ]
        super().render_movie(
            cameras_key_frames, self._nb_frames_between_keys, self._nb_frames_between_keys,
            start_frame, end_frame, frame_step, frame_list)

def main(argv):
    args = MovieScenario.parse_arguments(argv)

    scenario = NeuromodulationScenario(
        hostname=args.hostname,
        port=args.port,
        projection=args.projection,
        output_folder=args.export_folder,
        image_k=args.image_resolution_k,
        image_samples_per_pixel=args.image_samples_per_pixel,
        log_level=args.log_level,
        shaders=args.shaders,
        draft=args.draft,
        gi_length=10.0 * scale.x)

    scenario.setup_scene()
    scenario.set_rendering_settings('bio_explorer')
    scenario.render_movie(
        start_frame=args.from_frame,
        end_frame=args.to_frame,
        frame_step=args.frame_step,
        frame_list=args.frame_list)

if __name__ == "__main__":
    main(sys.argv[1:])
