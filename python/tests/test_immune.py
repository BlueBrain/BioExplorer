from bioexplorer import BioExplorer, Volume, Protein, Vector2, Vector3

def test_immune():
    resource_folder = 'test_files/'
    pdb_folder = resource_folder + 'pdb/immune/'

    be = BioExplorer('localhost:5000')
    be.reset()
    print('BioExplorer version ' + be.version())

    ''' Suspend image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=0)

    ''' Proteins '''
    glycan_radius_multiplier = 1.0
    glycan_add_sticks = True

    lactoferrin_path = pdb_folder + '1b0l.pdb'
    defensin_path = pdb_folder + '1ijv.pdb'

    ''' Scene parameters '''
    scene_size = 800

    ''' Lactoferrins'''
    lactoferrins = Protein(
        sources=[lactoferrin_path],
        load_non_polymer_chemicals=True,
        number_of_instances=150
    )

    lactoferrins_volume = Volume(
        name=be.NAME_LACTOFERRIN,
        size=Vector2(scene_size, scene_size),
        protein=lactoferrins
    )

    be.add_volume(
        volume=lactoferrins_volume,
        representation=be.REPRESENTATION_ATOMS,
        position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0)
    )

    ''' Defensins '''
    defensins = Protein(
        sources=[defensin_path],
        load_non_polymer_chemicals=True,
        number_of_instances=300
    )

    defensins_volume = Volume(
        name=be.NAME_DEFENSIN,
        size=Vector2(scene_size, scene_size),
        protein=defensins
    )

    be.add_volume(
        volume=defensins_volume,
        representation=be.REPRESENTATION_ATOMS,
        position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0)
    )

    ''' Restore image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=20)

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)