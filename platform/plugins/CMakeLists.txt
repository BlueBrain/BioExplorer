# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Core <https://github.com/BlueBrain/Core>

option(PLATFORM_DEFLECT_ENABLED "Activate streaming to display wall" OFF)
if(PLATFORM_DEFLECT_ENABLED)
  add_subdirectory(deflect)
endif()

if(PLATFORM_NETWORKING_ENABLED)
  add_subdirectory(rockets)
endif()

option(PLATFORM_MULTIVIEW_ENABLED "Activate Multiview camera plugin" OFF)
if(PLATFORM_MULTIVIEW_ENABLED)
  add_subdirectory(multiview)
endif()

option(PLATFORM_OPENDECK_ENABLED "Activate OpenDeck plugin" OFF)
if(PLATFORM_OPENDECK_ENABLED)
  add_subdirectory(openDeck)
endif()

option(PLATFORM_VRPN_ENABLED "Activate vrpn plugin" OFF)
if(PLATFORM_VRPN_ENABLED)
  add_subdirectory(vrpn)
endif()
