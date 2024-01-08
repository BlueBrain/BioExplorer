# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2015-2024 Blue BrainProject / EPFL
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
#

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
