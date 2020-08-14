from bioexplorer import BioExplorer, Cell, Membrane, SurfaceReceptor, \
                        Vector2, Vector3

resource_folder = 'test_files/'

be = BioExplorer('localhost:5000')
be.reset()
print('BioExplorer version ' + be.version())

''' Suspend image streaming '''
be.get_client().set_application_parameters(image_stream_fps=0)
be.get_client().set_camera(
    orientation=[0.0, 0.0, 0.0, 1.0], position=[0, 0, 200], target=[0, 0, 0])

''' Proteins '''
protein_radius_multiplier = 1.0
protein_representation = be.REPRESENTATION_ATOMS
protein_load_hydrogen = False

''' Glycan trees '''
glycan_radius_multiplier = 1.0
glycan_add_sticks = True

''' Membrane parameters '''
membrane_size = 800
membrane_height = 80

membrane = Membrane(
    sources=[resource_folder + 'pdb/membrane/popc.pdb'],
    number_of_instances=400000)

ace2_receptor = SurfaceReceptor(
    source=resource_folder + 'pdb/6m1d.pdb',
    number_of_instances=20,
    position=Vector3(0.0, 6.0, 0.0))

cell = Cell(
    name='Cell',
    size=Vector2(membrane_size, membrane_height),
    shape=be.ASSEMBLY_SHAPE_SINUSOIDAL,
    membrane=membrane, receptor=ace2_receptor)

be.add_cell(
    cell=cell,
    representation=protein_representation,
    atom_radius_multiplier=protein_radius_multiplier)

''' Materials '''
be.apply_default_color_scheme(shading_mode=be.SHADING_MODE_BASIC)

''' Restore image streaming '''
be.get_client().set_application_parameters(image_stream_fps=20)
