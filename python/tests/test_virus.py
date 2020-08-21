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

from bioexplorer import BioExplorer, Membrane, Protein, Sugars, Virus, RNASequence, Vector2, Vector3, Quaternion


def test_virus():
    resource_folder = 'test_files/'
    pdb_folder = resource_folder + 'pdb/'
    rna_folder = resource_folder + 'rna/'
    glycan_folder = pdb_folder + 'glycans/'

    be = BioExplorer('localhost:5000')
    be.reset()
    print('BioExplorer version ' + be.version())

    ''' Settings '''
    protein_radius_multiplier = 1.0
    protein_representation = BioExplorer.REPRESENTATION_ATOMS
    protein_load_hydrogen = False

    ''' Virus configuration '''
    nb_protein_s = 62
    nb_protein_e = 42
    nb_protein_m = 50

    ''' Suspend image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=0)
    be.set_camera(
        direction=[-0.882, 0.154, -0.444], origin=[65, -157, 304],
        up=[0.082, 0.980, 0.176]
    )

    ''' Virus parameters'''
    add_glycans = True
    add_rna = False

    ''' Glycan trees '''
    glycan_add_sticks = True

    ''' Protein S '''
    open_conformation_indices = [1]
    closed_conformation_indices = list()
    for i in range(nb_protein_s):
        if i not in open_conformation_indices:
            closed_conformation_indices.append(i)

    virus_protein_s = Protein(
        sources=[
            pdb_folder + '6vyb.pdb',  # Open conformation
            pdb_folder + 'sars-cov-2-v1.pdb'  # Closed conformation
        ],
        load_hydrogen=protein_load_hydrogen, number_of_instances=nb_protein_s,
        assembly_params=Vector2(11.5, 0.0), cutoff_angle=0.999,
        orientation=Quaternion(0.087, 0.0, 0.996, 0.0),
        instance_indices=[open_conformation_indices, closed_conformation_indices])

    ''' Protein M (QHD43419 ) '''
    virus_protein_m = Protein(
        sources=[pdb_folder + 'QHD43419a.pdb'], load_hydrogen=protein_load_hydrogen,
        number_of_instances=nb_protein_m, assembly_params=Vector2(2.0, 0.0),
        cutoff_angle=0.999, orientation=Quaternion(0.99, 0.0, 0.0, 0.135))

    ''' Protein E (QHD43418 P0DTC4) '''
    virus_protein_e = Protein(
        sources=[pdb_folder + 'QHD43418a.pdb'], load_hydrogen=protein_load_hydrogen,
        number_of_instances=nb_protein_e, assembly_params=Vector2(3.0, 0.0),
        cutoff_angle=0.9999, orientation=Quaternion(0.705, 0.705, -0.04, -0.04))

    ''' Virus membrane '''
    virus_membrane = Membrane(
        sources=[pdb_folder + 'membrane/popc.pdb'],
        number_of_instances=15000
    )

    ''' RNA Sequence '''
    rna_sequence = None
    if add_rna:
        import math

        rna_sequence = RNASequence(
            source=rna_folder + 'sars-cov-2.rna',
            assembly_params=Vector2(11.0, 0.5),
            t_range=Vector2(0, 30.5 * math.pi), shape=be.RNA_SHAPE_TREFOIL_KNOT,
            shape_params=Vector3(1.51, 1.12, 1.93))

    ''' Coronavirus '''
    name = 'Coronavirus'
    coronavirus = Virus(
        name=name,
        protein_s=virus_protein_s,
        protein_e=virus_protein_e,
        protein_m=virus_protein_m,
        membrane=virus_membrane,
        rna_sequence=rna_sequence,
        assembly_params=Vector2(45.0, 1.5))

    clip_planes = list()
    if add_rna:
        clip_planes.append([0, 0, -1, 15])
    be.add_virus(
        virus=coronavirus, position=Vector3(-70.0, -100.0, 230.0),
        representation=protein_representation,
        atom_radius_multiplier=protein_radius_multiplier,
        clipping_planes=clip_planes
    )


    def add_glycans_to_virus(virus_name):
        complex_paths = [
            glycan_folder + 'complex/5.pdb',
            glycan_folder + 'complex/15.pdb',
            glycan_folder + 'complex/25.pdb',
            glycan_folder + 'complex/35.pdb'
        ]
        high_mannose_paths = [
            glycan_folder + 'high-mannose/1.pdb',
            glycan_folder + 'high-mannose/2.pdb',
            glycan_folder + 'high-mannose/3.pdb',
            glycan_folder + 'high-mannose/4.pdb'
        ]
        hybrid_paths = [glycan_folder + 'hybrid/20.pdb']
        o_glycan_paths = [glycan_folder + 'o-glycan/12.pdb']

        ''' High-mannose '''
        indices = [61, 122, 234, 603, 709, 717, 801, 1074]
        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE,
            protein_name=be.NAME_PROTEIN_S_CLOSED, paths=high_mannose_paths,
            indices=indices, add_sticks=glycan_add_sticks,
            allowed_occurrences=closed_conformation_indices)
        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE,
            protein_name=be.NAME_PROTEIN_S_OPEN, paths=high_mannose_paths,
            indices=indices, index_offset=19, add_sticks=glycan_add_sticks,
            allowed_occurrences=open_conformation_indices)

        ''' Complex '''
        indices1 = [17, 74, 149, 165, 282, 331, 343, 616, 1098, 1134, 1158, 1173, 1194]
        indices2 = [17, 74, 149, 165, 282, 331, 343, 1098, 1134, 1158, 1173, 1194]

        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_COMPLEX,
            protein_name=be.NAME_PROTEIN_S_CLOSED, paths=complex_paths,
            indices=indices1, add_sticks=glycan_add_sticks,
            allowed_occurrences=closed_conformation_indices)
        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_COMPLEX,
            protein_name=be.NAME_PROTEIN_S_OPEN, paths=complex_paths,
            indices=indices2, index_offset=19, add_sticks=glycan_add_sticks,
            allowed_occurrences=open_conformation_indices)

        ''' Hybrid '''
        indices = [657]

        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_HYBRID,
            protein_name=be.NAME_PROTEIN_S_CLOSED, paths=hybrid_paths,
            indices=indices, add_sticks=glycan_add_sticks, allowed_occurrences=closed_conformation_indices)
        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_HYBRID,
            protein_name=be.NAME_PROTEIN_S_OPEN, paths=hybrid_paths,
            indices=indices, index_offset=19, add_sticks=glycan_add_sticks,
            allowed_occurrences=open_conformation_indices)

        ''' O-Glycans '''
        indices = [[323, [0.0, 0.0, 0.0, 1.0]], [325, [0.0, 0.0, 0.0, 1.0]]]

        for index in indices:
            o_glycan_name = virus_name + '_' + be.NAME_GLYCAN_O_GLYCAN + '_' + str(index[0])
            o_glycan = Sugars(
                assembly_name=virus_name, name=o_glycan_name,
                contents=''.join(open(o_glycan_paths[0]).readlines()),
                protein_name=virus_name + '_' + be.NAME_PROTEIN_S_CLOSED,
                add_sticks=glycan_add_sticks, site_indices=[index[0]],
                orientation=index[1]
            )
            be.add_glucoses(o_glycan)

        ''' High-mannose glycans on Protein M '''
        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE,
            protein_name=be.NAME_PROTEIN_M, paths=high_mannose_paths,
            add_sticks=glycan_add_sticks)

        ''' Complex glycans on Protein E '''
        be.add_multiple_glycans(
            assembly_name=virus_name, glycan_type=be.NAME_GLYCAN_COMPLEX,
            protein_name=be.NAME_PROTEIN_E, paths=complex_paths,
            add_sticks=glycan_add_sticks)

    if add_glycans:
        add_glycans_to_virus(name)

    ''' Restore image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
