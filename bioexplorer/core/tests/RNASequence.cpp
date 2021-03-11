/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <plugin/bioexplorer/RNASequence.h>

#include <brayns/Brayns.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Scene.h>

#define BOOST_TEST_MODULE rnasequence
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace bioexplorer;

std::string getFileContents(const std::string& filename)
{
    std::ifstream file(filename);
    std::string str;
    if (file)
    {
        std::ostringstream ss;
        ss << file.rdbuf();
        str = ss.str();
    }
    else
        throw std::runtime_error("Failed to open " + filename);
    return str;
}

RNASequenceDescriptor getDescriptor()
{
    RNASequenceDescriptor descriptor;

    descriptor.assemblyName = "test";
    descriptor.name = "test";
    descriptor.contents = getFileContents(
        "./bioexplorer/pythonsdk/tests/test_files/rna/sars-cov-2.rna");
    descriptor.shape = RNAShape::trefoilKnot;
    descriptor.assemblyParams = {11.f, 0.5f};
    descriptor.range = {0.f, 30.5f * M_PI};
    descriptor.params = {1.51f, 1.12f, 1.93f};
    descriptor.position = {0.f, 0.f, 0.f};
    return descriptor;
}

BOOST_AUTO_TEST_CASE(rna_sequence)
{
    std::vector<const char*> argv{"brayns", "--http-server", "localhost:0",
                                  "--plugin", "BioExplorer"};
    brayns::Brayns brayns(argv.size(), argv.data());
    auto& scene = brayns.getEngine().getScene();
    RNASequence rnaSequence(scene, getDescriptor());

    BOOST_CHECK(rnaSequence.getRNASequences().size() == 0);
}
