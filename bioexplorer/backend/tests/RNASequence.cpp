/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <science/common/Assembly.h>
#include <science/common/Logs.h>
#include <science/molecularsystems/RNASequence.h>

#include <platform/core/Core.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Scene.h>

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
    descriptor.contents = getFileContents("./bioexplorer/pythonsdk/tests/test_files/rna/sars-cov-2.rna");
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
    std::vector<const char*> argv{"service", "--http-server", "localhost:0", "--plugin",
                                  "BioExplorer --db-name=bioexplorer --db-user=core "
                                  "--db-password=core --db-host=localhost --db-port=5432"};
    core::Core core(argv.size(), argv.data());
    auto& scene = core.getEngine().getScene();
    Assembly assembly(scene, getAssemblyDescriptor());
    assembly.addRNASequence(getRNASequenceDescriptor());

    BOOST_CHECK(assembly.getRNASequence()->getRNASequences().size() == 0);
}
} // namespace tests