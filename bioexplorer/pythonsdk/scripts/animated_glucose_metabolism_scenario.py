#!/usr/bin/env python
"""Animated glucose metabolism scenario"""

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

from bioexplorer import BioExplorer, Protein, Membrane, Cell, \
    AnimationParams, Volume, Vector2, Vector3, Quaternion, \
    MovieScenario
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import os
import sys
import math

resource_folder = './tests/test_files/'
pdb_folder = resource_folder + 'pdb/'
off_folder = resource_folder + 'off/'
membrane_folder = pdb_folder + 'membrane/'
lipids_folder = membrane_folder + 'lipids/'
ion_channels_folder = pdb_folder + 'ion_channels/'
metabolites_folder = pdb_folder + 'metabolites/'
transporters_folder = pdb_folder + 'transporters/'

''' Simulation identifier '''
simulation_guid = 4

''' Ratio to apply to concentrations for visualization purpose '''
concentration_visualization_ratio = 1.0

''' Protein representation '''
representation = BioExplorer.REPRESENTATION_ATOMS_AND_STICKS
atom_radius_multiplier = 1.0

''' Scene size '''
scene_size = Vector3(250.0, 500.0, 250.0)
membrane_size = Vector3(scene_size.x, scene_size.y / 50.0, scene_size.z)

fullNGVUnitVolumeInLiters = 2e-11 / 0.45
nanometersCubicToLiters = 1e-24
avogadro = 6.02e23
fullSceneVolumeInLiters = scene_size.x * scene_size.y * scene_size.z * nanometersCubicToLiters
scene_ratio = fullSceneVolumeInLiters / fullNGVUnitVolumeInLiters


''' Neuron trans-membrane proteins '''
pdb_glut3 = transporters_folder + '4zwc.pdb'
pdb_glut3_closed = transporters_folder + '5c65.pdb'
pdb_mct2_lac = transporters_folder + '7bp3.pdb'

''' Astrocyte trans-membrane proteins '''
pdb_glut1 = transporters_folder + '4pyp.pdb'
pdb_mct1_lac = transporters_folder + '6lz0.pdb'

''' Trans-membrane proteins '''
pdb_nka = transporters_folder + '4hqj.pdb'

''' Lipids '''
pdb_lipids = [
    lipids_folder + 'lipid_430.pdb',
    lipids_folder + 'lipid_426.pdb',
    lipids_folder + 'lipid_424.pdb',
    lipids_folder + 'lipid_410.pdb'
]
lipid_density = 3.0

''' Regions of interest '''
region_astrocyte_mitochondrion = 0
region_astrocyte = 1
region_extracellular_space = 2
region_neuron = 3
region_neuron_mitochondrion = 4

region_mapping = dict()
region_mapping[0] = region_astrocyte_mitochondrion
region_mapping[1] = region_astrocyte
region_mapping[2] = region_extracellular_space
region_mapping[3] = region_neuron
region_mapping[4] = region_neuron_mitochondrion

location_areas = dict()
location_areas[region_astrocyte_mitochondrion] = [0.0575 * 0.25,
                                                  Vector2(100, scene_size.x / 2.0)]  # Mitochondria of the astrocyte
location_areas[region_astrocyte] = [0.25, Vector2(20, 100)]  # Cytosol of the astrocyte
location_areas[region_extracellular_space] = [0.20, Vector2(-20, 20)]  # Extracellular space
location_areas[region_neuron] = [0.45, Vector2(-10, -100)]  # Cytosol of the neuron
location_areas[region_neuron_mitochondrion] = [0.0575 * 0.45,
                                               Vector2(-100, -scene_size.x / 2.0)]  # Mitochondria of the neuron

# Currently not used:
# -------------------
# region_capilarities = 5
# region_synaptic = 6
# region_vasculature = 7
# -------------------
# location_areas[region_capilarities] = [0.0055, Vector2(0, 0)] # Capillaries
# location_areas[region_synaptic] = [0.0, Vector2(0, 0)] # Synaptic part of the extracellular space
# location_areas[region_vasculature] = [0.0, Vector2(0, 0)] # Vasculature
# -------------------

indices = list()
indices.append(-scene_size.y / 2.0)
total = 0.0
for location_area in location_areas:
    total = total + location_areas[location_area][0] * scene_size.y
    indices.append(-scene_size.y / 2.0 + total)
indices.append(scene_size.y / 2.0)
# self._log(1, indices)

i = 0
for region in region_mapping:
    location_areas[region_mapping[region]][1] = Vector2(indices[i], indices[i+1])
    i += 1


class GlucoseMetabolismScenario(MovieScenario):

    def __init__(self, hostname, port, projection, output_folder, image_k=4,
                 image_samples_per_pixel=64, log_level=1, shaders=list(['bio_explorer'])):
        super().__init__(hostname, port, projection, output_folder,
                         image_k, image_samples_per_pixel, log_level, False, shaders)

        db_host = os.environ['DB_HOST']
        db_name = os.environ['DB_NAME']
        db_user = os.environ['DB_USER']
        db_password = os.environ['DB_PASSWORD']
        self._db_schema = os.environ['DB_SCHEMA']

        db_connection_string = 'postgresql+psycopg2://%s:%s@%s:5432/%s' % (db_user, db_password, db_host, db_name)
        self._engine = create_engine(db_connection_string)
        self._db_connection = self._engine.connect()

    def _get_simulations(self):
        simulations = dict()
        with Session(self._engine) as session:
            sql = "SELECT guid, description FROM %s.simulations ORDER BY guid" % self._db_schema
            data = session.execute(sql)
            for d in data.all():
                simulations[d[0]] = d[1]
        return simulations

    def _get_variables(self):
        variables = dict()
        with Session(self._engine) as session:
            sql = "SELECT guid, pdb, description FROM %s.variable WHERE pdb IS NOT NULL ORDER BY guid" % self._db_schema
            data = session.execute(sql)
            for d in data.all():
                pdb_guid = d[1]
                if pdb_guid:
                    variables[d[0]] = [pdb_guid, d[2]]
        return variables

    def _get_metabolites(self):
        metabolites = dict()
        with Session(self._engine) as session:
            sql = "SELECT guid, pdb, description FROM %s.variable WHERE type=1 AND pdb IS NOT NULL ORDER BY guid" % self._db_schema
            data = session.execute(sql)
            for d in data.all():
                pdb_guid = d[1]
                if pdb_guid:
                    metabolites[d[0]] = [pdb_guid, d[2]]
        return metabolites

    def _get_locations(self):
        locations = dict()
        with Session(self._engine) as session:
            sql = "SELECT guid, description FROM %s.location ORDER BY guid" % self._db_schema
            data = session.execute(sql)
            for d in data.all():
                locations[d[0]] = d[1]
        return locations

    def _get_concentration(self, variable_guid, simulation_guid, frame, location_guid):
        with Session(self._engine) as session:
            sql = 'SELECT v.guid AS guid, c.value AS concentration FROM %s.variable as v, %s.concentration AS c WHERE c.variable_guid=%d AND v.guid=c.variable_guid AND c.frame=%d AND c.simulation_guid=%d AND v.location_guid=%d ORDER BY v.guid' % (
                self._db_schema, self._db_schema, variable_guid, frame, simulation_guid, location_guid)
            data = session.execute(sql)
            for d in data.all():
                return(float(d[0]))
        return 0.0

    def _get_nb_molecules(self, concentration, location_guid):
        return int(concentration_visualization_ratio * scene_ratio * avogadro * (1e-3 * concentration) * fullNGVUnitVolumeInLiters * location_areas[location_guid][0])

    def _get_nb_proteins(self, concentration, location_guid):
        return int(math.exp(concentration) * 1e-6 * avogadro * fullNGVUnitVolumeInLiters * scene_ratio * location_areas[location_guid][0])

    def _add_metabolites(self, frame):
        locations = self._get_locations()
        variables = self._get_metabolites()

        random_seed = 1
        for location in locations:
            if location not in region_mapping.values():
                continue

            for variable in variables:
                pdb_guid = variables[variable][0]
                if not pdb_guid:
                    continue

                variable_guid = int(variable)
                variable_description = variables[variable][1]
                location_guid = int(location)
                file_name = metabolites_folder + pdb_guid + '.pdb'
                concentration = self._get_concentration(
                    variable_guid, simulation_guid, frame, location_guid)
                nb_molecules = max(1, self._get_nb_molecules(concentration, location_guid))
                self._log(2, 'Loading %d molecules for variable %s' % (nb_molecules, variable_description))
                location_name = locations[location_guid]
                self._log(1, '- [%s] [%d] %s: %s.pdb: %d' % (location_name,
                                                                variable_guid, variable_description, pdb_guid, nb_molecules))
                location_area = location_areas[location_guid][1]
                area_size = Vector3(
                    scene_size.x,
                    0.95 * (location_area.y - location_area.x),
                    scene_size.z)
                area_position = Vector3(0.0, (location_area.y + location_area.x) / 2.0, 0.0)

                name = location_name + '_' + variable_description

                metabolite = Protein(
                    name=name, source=file_name,
                    load_bonds=True, load_hydrogen=True,
                    load_non_polymer_chemicals=True,
                    occurences=nb_molecules,
                    animation_params=AnimationParams(
                        random_seed,
                        random_seed + frame + 1, 0.2,
                        random_seed + frame + 2, 1.0))

                volume = Volume(
                    name=name,
                    shape=self._be.ASSEMBLY_SHAPE_CUBE,
                    shape_params=area_size,
                    protein=metabolite
                )
                self._check(self._be.add_volume(
                    volume=volume, representation=representation, position=area_position))
                random_seed += 3

    def _add_neuron(self, frame):
        name = 'Neuron'
        transmembrane_proteins = list()

        # Transporter GLUT3 (https://opm.phar.umich.edu/proteins/442)
        nb_transporter_glut3 = self._get_nb_proteins(-0.250079574048427, region_neuron)
        if nb_transporter_glut3 > 0:
            self._log(1, '- Transpoter GLUT3: %d' % nb_transporter_glut3)
            transmembrane_proteins.append(Protein(
                name=name + '_GLUT3',
                chain_ids=[1],
                source=pdb_glut3,
                occurences=30,  # nb_transporter_glut3,
                rotation=Quaternion(0.707, -0.693, -0.139, 0.0),
                load_non_polymer_chemicals=True, load_hydrogen=True, load_bonds=True,
                animation_params=AnimationParams(1, frame + 1, 0.1, frame + 2, 0.25),
                transmembrane_params=Vector2(0.0, 2.0)
            ))

        # Transporter MCT2 (https://opm.phar.umich.edu/proteins/5233)
        nb_transporter_mct2 = self._get_nb_proteins(-2.00832096285463, region_neuron)
        if nb_transporter_mct2 > 0:
            self._log(1, '- Transporter MCT2: %d' % nb_transporter_mct2)
            transmembrane_proteins.append(Protein(
                name=name + '_MCT2',
                position=Vector3(0.0, 1.0, 0.0),
                source=pdb_mct2_lac,
                occurences=nb_transporter_mct2,
                load_non_polymer_chemicals=True, load_hydrogen=True, load_bonds=True,
                animation_params=AnimationParams(2, frame + 3, 0.1, frame + 4, 0.005),
                transmembrane_params=Vector2(0.0, 2.0)
            ))

        # Transporter NKA
        nb_transporter_nka = self._get_nb_proteins(0.325597445694269, region_neuron)
        if nb_transporter_nka > 0:
            self._log(1, '- Transporter NKA: %d' % nb_transporter_nka)
            transmembrane_proteins.append(Protein(
                name=name + '_NKA',
                source=pdb_nka,
                occurences=nb_transporter_nka,
                load_non_polymer_chemicals=True, load_hydrogen=True, load_bonds=True,
                animation_params=AnimationParams(3, frame + 5, 0.1, frame + 6, 0.25),
                transmembrane_params=Vector2(0.0, 2.0)
            ))

        # Membrane definition
        membrane = Membrane(
            lipid_sources=pdb_lipids,
            lipid_density=lipid_density,
            load_non_polymer_chemicals=True, load_bonds=True,
            animation_params=AnimationParams(0, frame + 7, 0.1, frame + 8, 0.25)
        )

        # Cell definition
        neuron = Cell(
            name=name,
            shape=self._be.ASSEMBLY_SHAPE_SINUSOID,
            shape_params=membrane_size,
            membrane=membrane,
            proteins=transmembrane_proteins)

        # Add cell to scene
        self._check(self._be.add_cell(
            cell=neuron, representation=representation,
            atom_radius_multiplier=atom_radius_multiplier,
            position=Vector3(0, location_areas[region_neuron][1].x, 0)))

    def _add_neuron_mitochondrion(self, frame):
        name = 'NeuronMitochondrion'

        # Transporter
        transporter = Protein(
            name=name + '_GLUT3',
            source=pdb_glut3,
            occurences=0,
            rotation=Quaternion(0.707, 0.707, 0.0, 0.0),
            load_non_polymer_chemicals=True, load_hydrogen=True, load_bonds=True,
            animation_params=AnimationParams(4, frame + 9, 0.1, frame + 10, 0.25),
            transmembrane_params=Vector2(0.0, 2.0)
        )

        # Membrane definition
        membrane = Membrane(
            lipid_sources=pdb_lipids,
            lipid_density=lipid_density,
            load_non_polymer_chemicals=True, load_bonds=True,
            animation_params=AnimationParams(0, frame + 11, 0.1, frame + 12, 0.25)
        )

        # Cell definition
        neuron_mitochodrion = Cell(
            name=name,
            shape=self._be.ASSEMBLY_SHAPE_SINUSOID,
            shape_params=membrane_size,
            membrane=membrane,
            proteins=[transporter])

        # Add cell to scene
        self._check(self._be.add_cell(
            cell=neuron_mitochodrion, representation=representation,
            rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            position=Vector3(0, location_areas[region_neuron_mitochondrion][1].x, 0)))

    def _add_astrocyte(self, frame):
        name = 'Astrocyte'
        transmembrane_proteins = list()

        # GLUT1 (https://opm.phar.umich.edu/proteins/2454)
        nb_transporter_glut1 = self._get_nb_proteins(-0.672647584446565, region_neuron)
        if nb_transporter_glut1 > 0:
            self._log(1, '- Transporter GLUT1: %d' % nb_transporter_glut1)
            transmembrane_proteins.append(Protein(
                name=name + '_GLUT1',
                source=pdb_glut1,
                occurences=nb_transporter_glut1,
                rotation=Quaternion(0.707, 0.707, 0.0, 0.0),
                load_non_polymer_chemicals=True, load_hydrogen=True, load_bonds=True,
                animation_params=AnimationParams(11, frame + 13, 0.1, frame + 14, 0.25),
                transmembrane_params=Vector2(0.0, 2.0)
            ))

        # Transporter MCT1 (https://opm.phar.umich.edu/proteins/6402)
        nb_transporter_mct1 = self._get_nb_proteins(-0.86948422680206, region_neuron)
        if nb_transporter_mct1 > 0:
            self._log(1, '- Transporter MCT1:  %d' % nb_transporter_mct1)
            transmembrane_proteins.append(Protein(
                name=name + '_MCT1',
                position=Vector3(0.0, 1.0, 0.0),
                source=pdb_mct1_lac,
                occurences=nb_transporter_mct1,
                load_non_polymer_chemicals=True, load_hydrogen=True, load_bonds=True,
                animation_params=AnimationParams(12, frame + 15, 0.1, frame + 16, 0.25),
                transmembrane_params=Vector2(0.0, 2.0)
            ))

        # Membrane definition
        membrane = Membrane(
            lipid_sources=pdb_lipids,
            lipid_density=lipid_density,
            load_non_polymer_chemicals=True, load_bonds=True,
            animation_params=AnimationParams(0, frame + 17, 0.1, frame + 18, 0.25))

        # Cell definition
        astrocyte = Cell(
            name=name,
            shape=self._be.ASSEMBLY_SHAPE_SINUSOID,
            shape_params=membrane_size,
            membrane=membrane,
            proteins=transmembrane_proteins)

        # Add cell to scene
        self._check(self._be.add_cell(
            cell=astrocyte, representation=representation,
            rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            position=Vector3(0, location_areas[region_astrocyte][1].y, 0)))

    def _add_astrocyte_mitochondrion(self, frame):
        name = 'AstrocyteMitochondrion'

        # Transporter
        transporter = Protein(
            name=name + '_GLUT1',
            source=pdb_glut1,
            occurences=0,
            rotation=Quaternion(0.707, 0.707, 0.0, 0.0),
            load_non_polymer_chemicals=True, load_hydrogen=True, load_bonds=True,
            animation_params=AnimationParams(15, frame + 19, 0.1, frame + 20, 0.25),
            transmembrane_params=Vector2(0.0, 2.0)
        )

        # Membrane definition
        membrane = Membrane(
            lipid_sources=pdb_lipids,
            lipid_density=lipid_density,
            load_non_polymer_chemicals=True, load_bonds=True,
            animation_params=AnimationParams(0, frame + 21, 0.1, frame + 22, 0.25)
        )

        # Cell definition
        astrocyte_mitochodrion = Cell(
            name=name,
            shape=self._be.ASSEMBLY_SHAPE_SINUSOID,
            shape_params=membrane_size,
            membrane=membrane,
            proteins=[transporter])

        # Add cell to scene
        self._check(self._be.add_cell(
            cell=astrocyte_mitochodrion, representation=representation,
            rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
            position=Vector3(0, location_areas[region_astrocyte_mitochondrion][1].y, 0)))

    def _set_materials_to_transmembrane_proteins(self):
        name = 'Neuron'
        self._check(self._be.set_protein_color_scheme(
            assembly_name=name, name=name+'_GLUT3',
            color_scheme=self._be.COLOR_SCHEME_AMINO_ACID_SEQUENCE,
            palette_name='Reds_r', palette_size=2))

        self._check(self._be.set_protein_color_scheme(
            assembly_name=name, name=name+'_MCT2',
            color_scheme=self._be.COLOR_SCHEME_AMINO_ACID_SEQUENCE,
            palette_name='Greens_r', palette_size=2))

        self._check(self._be.set_protein_color_scheme(
            assembly_name=name, name=name+'_NKA',
            color_scheme=self._be.COLOR_SCHEME_AMINO_ACID_SEQUENCE,
            palette_name='OrRd_r', palette_size=2))

        name = 'Astrocyte'
        self._check(self._be.set_protein_color_scheme(
            assembly_name=name, name=name+'_GLUT1',
            color_scheme=self._be.COLOR_SCHEME_AMINO_ACID_SEQUENCE,
            palette_name='Reds_r', palette_size=2))

        self._check(self._be.set_protein_color_scheme(
            assembly_name=name, name=name+'_MCT1',
            color_scheme=self._be.COLOR_SCHEME_AMINO_ACID_SEQUENCE,
            palette_name='Greens_r', palette_size=2))

    def _add_shapes(self):
        radius = 50000.0
        self._be.add_sphere(
            name='AstrocyteSphere',
            position=Vector3(0, location_areas[region_astrocyte][1].y - radius, 0),
            radius=radius,
            color=Vector3(0.4, 0.4, 0.25),
            opacity=0.5
        )
        self._be.add_sphere(
            name='AstrocyteMitochondrionSphere',
            position=Vector3(0, location_areas[region_astrocyte_mitochondrion][1].y - radius, 0),
            radius=radius,
            color=Vector3(0.2, 0.15, 0.3),
            opacity=0.5
        )
        self._be.add_sphere(
            name='NeuronSphere',
            position=Vector3(0, location_areas[region_neuron][1].x + radius, 0),
            radius=radius,
            color=Vector3(0.3, 0.37, 0.4),
            opacity=0.5
        )
        self._be.add_sphere(
            name='NeuronMytochondrionSphere',
            position=Vector3(0, location_areas[region_neuron_mitochondrion][1].x + radius, 0),
            radius=radius,
            color=Vector3(0.2, 0.15, 0.3),
            opacity=0.5
        )

    def _set_color_scheme(self, shading_mode, user_parameter=1.0, specular_exponent=5.0, glossiness=1.0):
        """
        Apply a default color scheme to all components in the scene

        :shading_mode: Shading mode (None, basic, diffuse, electron, etc)
        :user_parameter: User parameter specific to each shading mode
        :specular_exponent: Specular exponent for diffuse shading modes
        :glossiness: Glossiness
        """

        import seaborn as sns
        model_ids = self._be.get_model_ids()
        global_palette = sns.color_palette('Set3', len(model_ids["ids"]))

        index = 0
        for model_id in model_ids["ids"]:
            model_name = self._be.get_model_name(model_id)['name']
            material_ids = self._be.get_material_ids(model_id)["ids"]
            nb_materials = len(material_ids)

            if model_name.find('Neuron') != -1 and model_name.find('Mitochondrion') == -1:
                palette = sns.color_palette("Blues", nb_materials)
                self._be.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )
            elif model_name.find('Astrocyte') != -1 and model_name.find('Mitochondrion') == -1:
                palette = sns.color_palette("Wistia", nb_materials)
                self._be.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )
            elif model_name.find('Mitochondrion') != -1:
                palette = sns.color_palette("Purples", nb_materials)
                self._be.set_materials_from_palette(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    palette=palette,
                    shading_mode=shading_mode,
                    user_parameter=user_parameter,
                    glossiness=glossiness,
                    specular_exponent=specular_exponent,
                )
            elif model_name.find('AABB') != -1:
                continue
            else:
                colors = list()
                shading_modes = list()
                user_parameters = list()
                glossinesses = list()
                specular_exponents = list()

                for _ in material_ids:
                    colors.append(global_palette[index])
                    shading_modes.append(shading_mode)
                    user_parameters.append(user_parameter)
                    glossinesses.append(glossiness)
                    specular_exponents.append(specular_exponent)

                self._be.set_materials(
                    model_ids=[model_id],
                    material_ids=material_ids,
                    diffuse_colors=colors,
                    specular_colors=colors,
                    shading_modes=shading_modes,
                    user_parameters=user_parameters,
                    glossinesses=glossinesses,
                    specular_exponents=specular_exponents
                )
                index += 1

    def _add_aabb(self):
        return self._be.add_bounding_box(
            name='AABB',
            bottom_left_corner=Vector3(-scene_size.x / 2.0, -
                                       scene_size.y / 2.0, -scene_size.z / 2.0),
            top_right_corner=Vector3(scene_size.x / 2.0, scene_size.y / 2.0, scene_size.z / 2.0),
            radius=0.25)

    def _set_glucose_molecule_transformation(self, frame):
        ''' Set glucose molecule position '''
        p = self._core.get_camera()['position']
        t = self._core.get_camera()['target']
        r = self._core.get_camera()['orientation']

        pos = [0, 0, 0]
        for i in range(3):
            pos[i] = p[i] + (t[i] - p[i]) * 3.0

        target = Vector3(pos[0], pos[1], pos[2])
        glucose_model_name = 'Extracellular space_Glucose'

        roll = math.pi / 2.0 + 0.091 * math.pi * math.cos(frame * math.pi / 45.0)
        yaw = 0.125 * math.pi * math.cos(frame * math.pi / 180.0)
        pitch = 0

        self._check(self._be.set_protein_instance_transformation(
            assembly_name=glucose_model_name,
            name=glucose_model_name,
            instance_index=0,
            position=target,
            rotation=Quaternion(r[3], r[0], r[1], r[2]) * self._euler_to_quaternion(yaw, pitch, roll))
        )

    @staticmethod
    def _euler_to_quaternion(yaw, pitch, roll):
        import numpy as np
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
            np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
            np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
            np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
            np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return Quaternion(qx, qy, qz, qw)

    def set_rendering_settings(self, renderer):
        if renderer == 'bio_explorer':
            self._be.set_rendering_quality(self._be.RENDERING_QUALITY_HIGH)
            params = self._core.BioExplorerRendererParams()
            params.shadows = 1.0
            params.soft_shadows = 0.1
            params.use_hardware_randomizer = True
            params.fog_start = 1000.0
            params.fog_thickness = 500.0
            params.gi_distance = 50.0
            params.gi_weight = 0.2
            params.gi_samples = 1
            params = self._core.set_renderer_params(params)
            params = self._core.set_renderer(head_light=True)

    def build_frame(self, frame):
        self._check(self._be.reset_scene())
        self._check(self._be.set_general_settings(model_visibility_on_creation=False))
        self._log(1, 'Scene bounding box...')
        self._add_aabb()
        self._log(1, 'Loading metabolites...')
        self._add_metabolites(frame % 450)  # 450 should be the number of simulation frame!!!
        self._log(1, 'Loading astrocyte mitochondrion membrane...')
        self._add_astrocyte_mitochondrion(frame)
        self._log(1, 'Loading astrocyte membrane...')
        self._add_astrocyte(frame)
        self._log(1, 'Loading neuron membrane...')
        self._add_neuron(frame)
        self._log(1, 'Loading neuron mitochondrion membrane...')
        self._add_neuron_mitochondrion(frame)
        self._log(1, 'Setting glucose molecule transformation')
        self._set_glucose_molecule_transformation(frame)
        self._log(1, 'Applying materials...')
        self._set_color_scheme(shading_mode=self._be.SHADING_MODE_PERLIN,
                               user_parameter=0.001, specular_exponent=50.0)
        self._set_materials_to_transmembrane_proteins()
        self._log(1, 'Building geometry...')
        self._check(self._be.set_models_visibility(True))
        self._log(1, 'Done')

    def render_movie(self, start_frame=0, end_frame=0, frame_step=1, frame_list=list()):
        aperture = 0.0
        focus = 2.938
        cameras_key_frames = [
            {
                'apertureRadius': aperture,
                'direction': [-0.0, 0.0, -1.0],
                'focusDistance': focus,
                'origin': [-0.12309164325388476, -68.17223360869282, 528.224633426888],
                'up': [-1.0, -6.661338147750939e-16, 0.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [-0.0, 0.0, -1.0],
                'focusDistance': focus,
                'origin': [-0.12309164325388476, -68.17223360869282, 140.11376864318507],
                'up': [-1.0, -6.661338147750939e-16, 0.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [0.0, 1.0, -2.220446049250313e-16],
                'focusDistance': focus,
                'origin': [-28.5, -67.86393737792969, -2.0],
                'up': [-0.0, 2.220446049250313e-16, 1.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [-0.0, 1.0, -4.440892098500626e-16],
                'focusDistance': focus,
                'origin': [-29.22458300330654, -18.82142928630076, -2.386302379782111],
                'up': [0.0, 4.440892098500626e-16, 1.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [-0.0, 1.0, -2.220446049250313e-16],
                'focusDistance': focus,
                'origin': [-29.22458300330654, -0.82142928630076, -2.386302379782111],
                'up': [0.0, 2.220446049250313e-16, 1.0]
            },
            {
                'apertureRadius': aperture,
                'direction': [0.9999970255814865, 1.5866404717910239e-21, -0.0024390219719829798],
                'focusDistance': focus,
                'origin': [-117.00812364660472, -14.7012431105328, -16.27792336717559],
                'up': [0.0, 1.0, 6.505232384196665e-19]
            },
            {
                'apertureRadius': aperture,
                'direction': [0.9999970255814865, 1.5866404717910239e-21, -0.0024390219719829798],
                'focusDistance': focus,
                'origin': [-117.00812364660472, -14.7012431105328, -16.27792336717559],
                'up': [0.0, 1.0, 6.505232384196665e-19]
            }
        ]
        super().render_movie(cameras_key_frames, 120, 120, start_frame, end_frame, frame_step, frame_list)


def main(argv):
    args = MovieScenario.parse_arguments(argv)

    scenario = GlucoseMetabolismScenario(
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
