/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
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
using namespace details;
using namespace common;

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
