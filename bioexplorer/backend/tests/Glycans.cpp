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