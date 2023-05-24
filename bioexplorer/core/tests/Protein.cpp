/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <plugin/common/Logs.h>
#include <plugin/molecularsystems/Protein.h>

#include <brayns/Brayns.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Scene.h>

#define BOOST_TEST_MODULE protein
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
    std::vector<const char*> argv{"brayns", "--http-server", "localhost:0", "--plugin",
                                  "BioExplorer --db-name=bioexplorer --db-user=brayns "
                                  "--db-password=brayns --db-host=localhost --db-port=5432"};
    brayns::Brayns brayns(argv.size(), argv.data());
    auto& scene = brayns.getEngine().getScene();
    Protein protein(scene, getProteinDescriptor());

    BOOST_CHECK(protein.getAtoms().size() == 21776);
    BOOST_CHECK(protein.getResidues().size() == 20);
    BOOST_CHECK(protein.getResidueSequences().size() == 4);

    std::vector<Vector3d> positions;
    std::vector<Quaterniond> rotations;
    const std::vector<size_t> siteIndices = {};
    protein.getGlycosilationSites(positions, rotations, siteIndices);
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
