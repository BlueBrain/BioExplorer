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

from bioexplorer import BioExplorer, Membrane, Protein, Sugars, Virus, RNASequence, Vector2, Vector3, Quaternion


def test_virus():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/'
    rna_folder = resource_folder + 'rna/'
    glycan_folder = pdb_folder + 'glycans/'

    be = BioExplorer('localhost:5000')
    be.reset()

    # Settings
    virus_radius = 45.0
    add_glycans = True
    protein_radius_multiplier = 1.0
    protein_representation = BioExplorer.REPRESENTATION_ATOMS
    protein_load_hydrogen = False

    # Virus configuration
    nb_protein_s = 62
    nb_protein_s_indices = [1, 27, 43]
    nb_protein_e = 42
    nb_protein_m = 50
    show_rna = False

    # Virus parameters
    show_functional_regions = False
    show_glycosylation_sites = False

    ''' Suspend image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=0)

    # Protein S
    open_conformation_indices = nb_protein_s_indices
    closed_conformation_indices = list()
    for i in range(nb_protein_s):
        if i not in open_conformation_indices:
            closed_conformation_indices.append(i)

    virus_protein_s = Protein(sources=[pdb_folder + '6vyb.pdb', pdb_folder + 'sars-cov-2-v1.pdb'],
                              load_hydrogen=protein_load_hydrogen, occurences=nb_protein_s,
                              assembly_params=Vector2(11.5, 0.0), cutoff_angle=0.999,
                              orientation=Quaternion(0.087, 0.0, 0.996, 0.0),
                              instance_indices=[open_conformation_indices, closed_conformation_indices])

    # Protein M (QHD43419)
    virus_protein_m = Protein(sources=[pdb_folder + 'QHD43419a.pdb'], load_hydrogen=protein_load_hydrogen,
                              occurences=nb_protein_m, assembly_params=Vector2(2.0, 0.0), cutoff_angle=0.999,
                              orientation=Quaternion(0.99, 0.0, 0.0, 0.135))
    # Protein E (QHD43418 P0DTC4)
    virus_protein_e = Protein(sources=[pdb_folder + 'QHD43418a.pdb'], load_hydrogen=protein_load_hydrogen,
                              occurences=nb_protein_e, assembly_params=Vector2(3.0, 0.0), cutoff_angle=0.9999,
                              orientation=Quaternion(0.705, 0.705, -0.04, -0.04))

    # Virus membrane
    virus_membrane = Membrane(sources=[pdb_folder + 'membrane/popc.pdb'], occurences=15000)

    # RNA Sequence
    clip_planes = list()
    rna_sequence = None
    if show_rna:
        clip_planes.append([0.0, 0.0, -1.0, 15.0])
        import math
        rna_sequence = RNASequence(source=rna_folder + 'sars-cov-2.rna', assembly_params=Vector2(11.0, 0.5),
                                   t_range=Vector2(0, 30.5 * math.pi), shape=be.RNA_SHAPE_TREFOIL_KNOT,
                                   shape_params=Vector3(1.51, 1.12, 1.93))

    # Coronavirus
    name = 'Coronavirus'
    coronavirus = Virus(name=name, protein_s=virus_protein_s, protein_e=virus_protein_e, protein_m=virus_protein_m,
                        membrane=virus_membrane, rna_sequence=rna_sequence, assembly_params=Vector2(virus_radius, 1.5))

    be.add_virus(
        virus=coronavirus, position=Vector3(-70.0, -100.0, 230.0), representation=protein_representation,
        atom_radius_multiplier=protein_radius_multiplier, clipping_planes=clip_planes)

    # Glycans
    if add_glycans:
        complex_paths = [glycan_folder + 'complex/5.pdb', glycan_folder + 'complex/15.pdb',
                         glycan_folder + 'complex/25.pdb', glycan_folder + 'complex/35.pdb']
        high_mannose_paths = [glycan_folder + 'high-mannose/1.pdb', glycan_folder + 'high-mannose/2.pdb',
                              glycan_folder + 'high-mannose/3.pdb', glycan_folder + 'high-mannose/4.pdb']
        o_glycan_paths = [glycan_folder + 'o-glycan/12.pdb']

        # High-mannose
        indices = [61, 122, 234, 603, 709, 717, 801, 1074]
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE, protein_name=be.NAME_PROTEIN_S_CLOSED,
            paths=high_mannose_paths, indices=indices, representation=protein_representation,
            allowed_occurrences=closed_conformation_indices, atom_radius_multiplier=protein_radius_multiplier)
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE, protein_name=be.NAME_PROTEIN_S_OPEN,
            paths=high_mannose_paths, indices=indices, index_offset=19, representation=protein_representation,
            allowed_occurrences=open_conformation_indices, atom_radius_multiplier=protein_radius_multiplier)

        # Complex
        indices1 = [17, 74, 149, 165, 282, 331, 343, 616, 657, 1098, 1134, 1158, 1173, 1194]
        indices2 = [17, 74, 149, 165, 282, 331, 343, 1098, 657, 1134, 1158, 1173, 1194]
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_COMPLEX, protein_name=be.NAME_PROTEIN_S_CLOSED,
            paths=complex_paths, indices=indices1, representation=protein_representation,
            allowed_occurrences=closed_conformation_indices, atom_radius_multiplier=protein_radius_multiplier)
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_COMPLEX, protein_name=be.NAME_PROTEIN_S_OPEN,
            paths=complex_paths, indices=indices2, index_offset=19, representation=protein_representation,
            allowed_occurrences=open_conformation_indices, atom_radius_multiplier=protein_radius_multiplier)

        # O-Glycans
        for index in [323, 325]:
            o_glycan_name = name + '_' + be.NAME_GLYCAN_O_GLYCAN + '_' + str(index)
            o_glycan = Sugars(
                assembly_name=name, name=o_glycan_name, source=o_glycan_paths[0],
                protein_name=name + '_' + be.NAME_PROTEIN_S_CLOSED, representation=protein_representation,
                site_indices=[index], atom_radius_multiplier=protein_radius_multiplier)
            be.add_sugars(o_glycan)

        # High-mannose glycans on Protein M
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_HIGH_MANNOSE, protein_name=be.NAME_PROTEIN_M,
            paths=high_mannose_paths, representation=protein_representation, atom_radius_multiplier=protein_radius_multiplier)

        # Complex glycans on Protein E
        be.add_multiple_glycans(
            assembly_name=name, glycan_type=be.NAME_GLYCAN_COMPLEX, protein_name=be.NAME_PROTEIN_E, paths=complex_paths,
            representation=protein_representation, atom_radius_multiplier=protein_radius_multiplier)

    # Apply default materials
    be.apply_default_color_scheme(shading_mode=be.SHADING_MODE_BASIC)

    # Functional regions on open spike
    if show_functional_regions:
        indices = [1, 16, 306, 330, 438, 507, 522, 816, 835, 908, 986, 1076, 1274, 2000]
        region_colors = [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.4, 0.1, 0.1],
                         [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [
                             1.0, 1.0, 1.0], [1.0, 1.0, 0.0],
                         [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        palette = list()
        for index in range(len(indices) - 1):
            for i in range(0, indices[index + 1] - indices[index]):
                c = list()
                for j in range(3):
                    c.append(region_colors[index][j] * 1)
                palette.append(c)
        be.set_protein_color_scheme(assembly_name=name, name=name + '_' + be.NAME_PROTEIN_S_OPEN,
                                    color_scheme=be.COLOR_SCHEME_REGION, palette=palette)

    # Display glycosylation sites
    if show_glycosylation_sites:
        import seaborn as sns
        palette = sns.color_palette('Set2', 2)
        be.set_protein_color_scheme(assembly_name=name, name=name + '_' + be.NAME_PROTEIN_S_OPEN,
                                    color_scheme=be.COLOR_SCHEME_GLYCOSYLATION_SITE, palette=palette)
        be.set_protein_color_scheme(assembly_name=name, name=name + '_' + be.NAME_PROTEIN_S_CLOSED,
                                    color_scheme=be.COLOR_SCHEME_GLYCOSYLATION_SITE, palette=palette)

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
