# Copyright 2020 - 2023 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
