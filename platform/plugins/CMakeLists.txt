# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

option(CORE_DEFLECT_ENABLED "Activate streaming to display wall" OFF)
if(CORE_DEFLECT_ENABLED)
  add_subdirectory(deflect)
endif()

if(CORE_NETWORKING_ENABLED)
  add_subdirectory(rockets)
endif()

option(CORE_MULTIVIEW_ENABLED "Activate Multiview camera plugin" OFF)
if(CORE_MULTIVIEW_ENABLED)
  add_subdirectory(multiview)
endif()

option(CORE_OPENDECK_ENABLED "Activate OpenDeck plugin" OFF)
if(CORE_OPENDECK_ENABLED)
  add_subdirectory(openDeck)
endif()

option(CORE_VRPN_ENABLED "Activate vrpn plugin" OFF)
if(CORE_VRPN_ENABLED)
  add_subdirectory(vrpn)
endif()
