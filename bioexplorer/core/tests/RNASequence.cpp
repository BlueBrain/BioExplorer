/* Copyright (c) 2020-2022, EPFL/Blue Brain Project
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

#include <plugin/common/Assembly.h>
#include <plugin/common/Logs.h>
#include <plugin/molecularsystems/RNASequence.h>

#include <brayns/Brayns.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Scene.h>

#define BOOST_TEST_MODULE rnasequence
#include <boost/test/unit_test.hpp>

#include <fstream>

namespace tests
{
using namespace bioexplorer;
using namespace molecularsystems;

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
        PLUGIN_THROW("Failed to open " + filename);
    return str;
}

AssemblyDetails getAssemblyDescriptor()
{
    AssemblyDetails descriptor;
    descriptor.name = "assembly";
    descriptor.shape = AssemblyShape::point;
    return descriptor;
}

RNASequenceDetails getRNASequenceDescriptor()
{
    RNASequenceDetails descriptor;

    descriptor.assemblyName = "test";
    descriptor.name = "test";
    descriptor.contents = getFileContents(
        "./bioexplorer/pythonsdk/tests/test_files/rna/sars-cov-2.rna");
    descriptor.shape = RNAShapeType::trefoilKnot;
    descriptor.shapeParams = {11.0, 0.5};
    descriptor.valuesRange = {0.0, 30.5 * static_cast<double>(M_PI)};
    descriptor.curveParams = {1.51, 1.12, 1.93};
    descriptor.position = {0.0, 0.0, 0.0};
    descriptor.rotation = {1.0, 0.0, 0.0, 0.0};
    return descriptor;
}

BOOST_AUTO_TEST_CASE(rna_sequence)
{
    std::vector<const char*> argv{"brayns", "--http-server", "localhost:0",
                                  "--plugin", "BioExplorer"};
    brayns::Brayns brayns(argv.size(), argv.data());
    auto& scene = brayns.getEngine().getScene();
    Assembly assembly(scene, getAssemblyDescriptor());
    assembly.addRNASequence(getRNASequenceDescriptor());

    BOOST_CHECK(assembly.getRNASequence()->getRNASequences().size() == 0);
}
} // namespace tests
