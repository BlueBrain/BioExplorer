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

#include <plugin/bioexplorer/Protein.h>

#include <brayns/Brayns.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Scene.h>

#define BOOST_TEST_MODULE protein
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

ProteinDescriptor getProteinDescriptor()
{
    ProteinDescriptor descriptor;
    descriptor.assemblyName = "test";
    descriptor.name = "test";
    descriptor.contents =
        getFileContents("./python/tests/test_files/pdb/6m1d.pdb");
    descriptor.shape = AssemblyShape::spherical;
    descriptor.assemblyParams = {0.f, 0.f};
    descriptor.atomRadiusMultiplier = 1.f;
    descriptor.loadBonds = true;
    descriptor.loadNonPolymerChemicals = true;
    descriptor.loadHydrogen = true;
    descriptor.representation = ProteinRepresentation::atoms;
    descriptor.chainIds = {};
    descriptor.recenter = true;
    descriptor.occurrences = 1;
    descriptor.allowedOccurrences = {};
    descriptor.randomSeed = 0;
    descriptor.locationCutoffAngle = 0.f;
    descriptor.positionRandomizationType = PositionRandomizationType::circular;
    descriptor.position = {0.f, 0.f, 0.f};
    descriptor.orientation = {0.f, 0.f, 0.f, 1.f};
    return descriptor;
}

BOOST_AUTO_TEST_CASE(protein)
{
    std::vector<const char*> argv{"brayns", "--http-server", "localhost:0",
                                  "--plugin", "BioExplorer"};
    brayns::Brayns brayns(argv.size(), argv.data());
    auto& scene = brayns.getEngine().getScene();
    Protein protein(scene, getProteinDescriptor());

    BOOST_CHECK(protein.getAtoms().size() == 21776);
    BOOST_CHECK(protein.getResidues().size() == 20);
    BOOST_CHECK(protein.getSequences().size() == 4);

    std::vector<Vector3f> positions;
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
