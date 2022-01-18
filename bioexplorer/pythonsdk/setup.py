#!/usr/bin/env python
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

"""setup.py"""
import os
import pathlib
import pkg_resources
from setuptools import find_packages, setup

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def parse_reqs(reqs_file):
    ''' parse the requirements '''
    install_reqs = list()
    with pathlib.Path(reqs_file).open() as requirements_txt:
        install_reqs = [str(requirement)
                        for requirement
                        in pkg_resources.parse_requirements(requirements_txt)]
    return install_reqs


REQS = parse_reqs(os.path.join(BASEDIR, "requirements.txt"))

# read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    packages=find_packages(),
    install_requires=REQS,
    name='bioexplorer',
    description='Python API for the Blue Brain BioExplorer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BlueBrain/BioExplorer.git',
    author='Blue Brain Project, EPFL',
    license='LGPLv3',
    project_urls={
            "Documentation": "https://bluebrain.github.io/BioExplorer/",
            "Source": "https://github.com/BlueBrain/BioExplorer",
    }
)
