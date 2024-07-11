# Copyright 2020 - 2024 Blue Brain Project / EPFL
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

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

set(TESTS_SRC
    MeshBasedMembrane.cpp
    Protein.cpp
    Glycans.cpp
    RNASequence.cpp
)

add_definitions(-DBOOST_TEST_DYN_LINK)

foreach(FILE ${TESTS_SRC})
    get_filename_component(NAME ${FILE} NAME_WE)
    add_executable(${NAME} ${FILE})

    target_link_libraries(${NAME}
        PRIVATE
        Core CoreParameters CoreCommon CoreEngine CoreIO OSPRayEngine CorePluginRockets stdc++fs
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${LIBRARY_NAME}
    )

    add_test(NAME ${NAME} COMMAND ${NAME})
endforeach()
