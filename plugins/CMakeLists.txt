# Copyright (c) 2015-2023, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of Brayns <https://github.com/BlueBrain/Brayns>

option(BRAYNS_DEFLECT_ENABLED "Activate streaming to display wall" OFF)
if(BRAYNS_DEFLECT_ENABLED)
  add_subdirectory(Deflect)
endif()

if(BRAYNS_NETWORKING_ENABLED)
  add_subdirectory(Rockets)
endif()

option(BRAYNS_MULTIVIEW_ENABLED "Activate Multiview camera plugin" OFF)
if(BRAYNS_MULTIVIEW_ENABLED)
  add_subdirectory(Multiview)
endif()

option(BRAYNS_OPENDECK_ENABLED "Activate OpenDeck plugin" OFF)
if(BRAYNS_OPENDECK_ENABLED)
  add_subdirectory(OpenDeck)
endif()

option(BRAYNS_VRPN_ENABLED "Activate vrpn plugin" OFF)
if(BRAYNS_VRPN_ENABLED)
  add_subdirectory(VRPN)
endif()

option(BRAYNS_MEG_ENABLED "Activate MagnetoEncephaloGram plugin" OFF)
if(BRAYNS_MEG_ENABLED)
  add_subdirectory(MEG)
endif()
