#!/usr/bin/env python
"""Build the list of missing frames"""

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

import glob
import sys
import os
import argparse


def find_missing(lst):
    return [x for x in range(lst[0], lst[-1]+1)
            if x not in lst]


def get_missing_ids(folder):
    present_files = glob.glob(folder + '/*.png')
    present_ids = list()
    for present_file in present_files:
        basename = os.path.basename(present_file)
        name = os.path.splitext(basename)[0]
        present_ids.append(int(name))
    present_ids.sort()
    return find_missing(present_ids)


def main(argv):
    parser = argparse.ArgumentParser(description='Missing frames')
    parser.add_argument('-i', '--input_folder', required=True, help='Input folder')
    args = parser.parse_args(argv)
    input_folder = args.input_folder

    missing_ids = get_missing_ids(input_folder)
    s = ''
    for missing_id in missing_ids:
        s += str(missing_id) + ' '
    print('Number of missing frames: %i' % len(missing_ids))
    print('Missing frames          : ' + s)


if __name__ == "__main__":
    main(sys.argv[1:])
