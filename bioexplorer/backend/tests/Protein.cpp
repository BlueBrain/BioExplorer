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
#include <science/molecularsystems/Protein.h>

#include <platform/core/Core.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Scene.h>

#define BOOST_TEST_MODULE protein
#include <boost/test/unit_test.hpp>

#include <fstream>

using namespace core;

namespace tests
{
using namespace bioexplorer;
using namespace molecularsystems;
using namespace details;

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

ProteinDetails getProteinDescriptor()
{
    ProteinDetails descriptor;
    descriptor.assemblyName = "test";
    descriptor.name = "test";
    descriptor.contents = getFileContents("./bioexplorer/pythonsdk/tests/test_files/pdb/6m1d.pdb");
    descriptor.atomRadiusMultiplier = 1.0;
    descriptor.loadBonds = true;
    descriptor.loadNonPolymerChemicals = true;
    descriptor.loadHydrogen = true;
    descriptor.representation = ProteinRepresentation::atoms;
    descriptor.chainIds = {};
    descriptor.recenter = true;
    descriptor.occurrences = 1;
    descriptor.allowedOccurrences = {};
    descriptor.animationParams = {};
    descriptor.position = {0.0, 0.0, 0.0};
    descriptor.rotation = {0.0, 0.0, 0.0, 1.0};
    return descriptor;
}

BOOST_AUTO_TEST_CASE(protein)
{
    std::vector<const char*> argv{"service", "--http-server", "localhost:0", "--plugin",
                                  "BioExplorer --db-name=bioexplorer --db-user=core "
                                  "--db-password=core --db-host=localhost --db-port=5432"};
    core::Core core(argv.size(), argv.data());
    auto& scene = core.getEngine().getScene();
    Protein protein(scene, getProteinDescriptor());

    BOOST_CHECK(protein.getAtoms().size() == 21776);
    BOOST_CHECK(protein.getResidues().size() == 20);
    BOOST_CHECK(protein.getResidueSequences().size() == 4);

    std::vector<Vector3d> positions;
    std::vector<Quaterniond> rotations;
    const std::vector<size_t> siteIndices = {};
    protein.getGlycosylationSites(positions, rotations, siteIndices);
    BOOST_CHECK(positions.size() == 24);
    BOOST_CHECK(rotations.size() == positions.size());

    positions.clear();
    rotations.clear();
    const size_ts chainIds = {};
    protein.getSugarBindingSites(positions, rotations, siteIndices, chainIds);
    BOOST_CHECK(positions.size() == 0);
    BOOST_CHECK(rotations.size() == 0);

    const auto sites = protein.getGlycosylationSites(siteIndices);
    BOOST_CHECK(sites.size() == 4);

    const std::vector<size_t> expectedSizes{5, 7, 5, 7};
    size_t count = 0;
    for (const auto& site : sites)
    {
        BOOST_CHECK(site.second.size() == expectedSizes[count]);
        ++count;
    }
}
} // namespace tests
