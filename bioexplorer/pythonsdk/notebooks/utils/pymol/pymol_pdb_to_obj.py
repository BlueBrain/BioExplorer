# Install the following packages
#
# sudo apt-get install pymol python3-tk python3-pip
# sudo pip3 install pmw

# Run that script from PyMol

import glob
import os
from pymol import cmd

resource_folder = os.path.abspath('../../../tests/test_files')
output_folder = os.path.join(resource_folder, 'obj')
pdb_folder = os.path.join(resource_folder, 'pdb', 'glycans', 'high-mannose')
pdb_files = glob.glob(os.path.join(pdb_folder, '*.pdb'))

for pdb_file in pdb_files:
    cmd.reinitialize()
    cmd.load(pdb_file)
    cmd.show('surface', 'all')
    obj_file = os.path.join(output_folder, os.path.basename(
        pdb_file).replace('.pdb', '.obj').lower())
    cmd.save(obj_file)
