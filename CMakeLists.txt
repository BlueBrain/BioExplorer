# Copyright (c) 2020-2021, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/BlueBrain/BioExplorer

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_subdirectory(bioexplorer/core)

option(BIOEXPLORER_MEDIA_MAKER_ENABLED "Enable media maker plugin" ON)
if(BIOEXPLORER_MEDIA_MAKER_ENABLED)
    add_subdirectory(extras/MediaMaker)
endif()
