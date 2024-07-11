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

#include <science/common/Logs.h>
#include <science/common/Node.h>
#include <science/molecularsystems/Glycans.h>

#include <platform/core/Core.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Scene.h>

#define BOOST_TEST_MODULE glycans
#include <boost/test/unit_test.hpp>

#include <fstream>

namespace tests
{
using namespace bioexplorer;
using namespace details;
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

SugarDetails getDescriptor()
{
    SugarDetails descriptor;
    descriptor.assemblyName = "test";
    descriptor.name = "test";
    descriptor.pdbId = "1";
    descriptor.contents = getFileContents("./bioexplorer/pythonsdk/tests/test_files/pdb/glycans/complex/1.pdb");
    descriptor.loadBonds = true;
    descriptor.representation = ProteinRepresentation::atoms;
    descriptor.recenter = true;
    descriptor.chainIds = {};
    descriptor.siteIndices = {};
    descriptor.rotation = {0.0, 0.0, 0.0, 1.0};
    return descriptor;
}

BOOST_AUTO_TEST_CASE(glycans)
{
    std::vector<const char*> argv{"service", "--http-server", "localhost:0", "--plugin",
                                  "BioExplorer --db-name=bioexplorer --db-user=core "
                                  "--db-password=core --db-host=localhost --db-port=5432"};
    core::Core core(argv.size(), argv.data());
    auto& scene = core.getEngine().getScene();
    Glycans glycans(scene, getDescriptor());

    BOOST_CHECK(glycans.getAtoms().size() == 291);
    BOOST_CHECK(glycans.getResidues().size() == 6);
    BOOST_CHECK(glycans.getSequencesAsString().size() == 0);
}
} // namespace tests