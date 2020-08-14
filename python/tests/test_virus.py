from bioexplorer import BioExplorer, Membrane, Sugars, Virus, VirusProtein, \
                        RNASequence, Vector2, Vector3, Quaternion

resource_folder = 'test_files/'
pdb_folder = resource_folder + 'pdb/'
rna_folder = resource_folder + 'rna/'
glycan_folder = pdb_folder + 'glycans/'

be = BioExplorer('localhost:5000')
be.reset()
print('BioExplorer version ' + be.version())

''' Suspend image streaming '''
be.get_client().set_application_parameters(image_stream_fps=0)
be.get_client().set_camera(
    orientation=[0.0, 0.0, 0.0, 1.0], position=[0, 0, 200], target=[0, 0, 0])

''' Virus parameters'''
add_glycans = True
add_rna = True
nb_protein_s = 62
nb_protein_e = 42
nb_protein_m = 50

''' Proteins '''
protein_radius_multiplier = 1.0
protein_representation = be.REPRESENTATION_ATOMS
protein_load_hydrogen = False

''' Glycan trees '''
glycan_radius_multiplier = 1.0
glycan_add_sticks = True


open_conformation_indices = [1]
closed_conformation_indices = list()
for i in range(nb_protein_s):
    if i not in open_conformation_indices:
        closed_conformation_indices.append(i)

''' Protein S '''
virus_protein_s = VirusProtein(
    sources=[
        pdb_folder + '6vyb.pdb',          # Open conformation
        pdb_folder + 'sars-cov-2-v1.pdb'  # Closed conformation
    ],
    load_hydrogen=protein_load_hydrogen, number_of_instances=nb_protein_s,
    assembly_params=Vector2(11.5, 0.0), cutoff_angle=0.999,
    orientation=Quaternion(0.087, 0.0, 0.996, 0.0),
    instance_indices=[open_conformation_indices, closed_conformation_indices])

''' Protein M (QHD43419 ) '''
virus_protein_m = VirusProtein(
    sources=[pdb_folder + 'QHD43419a.pdb'], load_hydrogen=protein_load_hydrogen,
    number_of_instances=nb_protein_m, assembly_params=Vector2(2.0, 0.0),
    cutoff_angle=0.999, orientation=Quaternion(0.99, 0.0, 0.0, 0.135))

''' Protein E (QHD43418 P0DTC4) '''
virus_protein_e = VirusProtein(
    sources=[pdb_folder + 'QHD43418a.pdb'], load_hydrogen=protein_load_hydrogen,
    number_of_instances=nb_protein_e, assembly_params=Vector2(3.0, 0.0),
    cutoff_angle=0.9999, orientation=Quaternion(0.705, 0.705, -0.04, -0.04))

''' Virus membrane '''
virus_membrane = Membrane(
    sources=[pdb_folder + 'membrane/popc.pdb'],
    number_of_instances=15000
)

''' Coronavirus '''
name = 'Coronavirus'
coronavirus = Virus(
    name=name,
    protein_s=virus_protein_s,
    protein_e=virus_protein_e,
    protein_m=virus_protein_m,
    membrane=virus_membrane,
    assembly_params=Vector2(45.0, 1.5))

clip_planes = list()
if add_rna:
    clip_planes.append([0, 0, -1, 15])
be.add_virus(
    virus=coronavirus,
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
        indices=indices2, index_offset=19,  add_sticks=glycan_add_sticks,
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


def add_rna_to_virus(virus_name):
    import math
    path = rna_folder + 'sars-cov-2.rna'

    rna_sequence = RNASequence(
        assembly_name=virus_name, source=path,
        name=virus_name + '_' + be.NAME_RNA_SEQUENCE, assembly_params=Vector2(11.0, 0.5),
        t_range=Vector2(0, 30.5 * math.pi), shape=be.RNA_SHAPE_TREFOIL_KNOT,
        shape_params=Vector3(1.51, 1.12, 1.93))
    be.add_rna_sequence(rna_sequence)


if add_glycans:
    add_glycans_to_virus(name)

if add_rna:
    add_rna_to_virus(name)

''' Materials '''
be.apply_default_color_scheme(shading_mode=be.SHADING_MODE_BASIC)

''' Restore image streaming '''
be.get_client().set_application_parameters(image_stream_fps=20)
