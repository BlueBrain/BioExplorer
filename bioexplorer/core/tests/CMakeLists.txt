# Copyright (c) 2020-2022, Cyrille Favreau
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of https://github.com/BlueBrain/BioExplorer

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(Brayns REQUIRED)

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
        brayns braynsParameters braynsCommon braynsEngine braynsIO braynsOSPRayEngine braynsRockets
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${LIBRARY_NAME}
    )

    add_test(NAME ${NAME} COMMAND ${NAME})
endforeach()
