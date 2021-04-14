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

#include <plugin/biology/Glycans.h>
#include <plugin/common/Logs.h>

#include <brayns/Brayns.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Scene.h>

#define BOOST_TEST_MODULE glycans
#include <boost/test/unit_test.hpp>

#include <fstream>

namespace tests
{
using namespace bioexplorer;
using namespace biology;

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

SugarsDetails getDescriptor()
{
    SugarsDetails descriptor;
    descriptor.assemblyName = "test";
    descriptor.name = "test";
    descriptor.contents = getFileContents(
        "./bioexplorer/pythonsdk/tests/test_files/pdb/glycans/complex/1.pdb");
    descriptor.proteinName;
    descriptor.atomRadiusMultiplier;
    descriptor.loadBonds = true;
    descriptor.representation = ProteinRepresentation::atoms;
    descriptor.recenter = true;
    descriptor.chainIds = {};
    descriptor.siteIndices = {};
    descriptor.rotation = {0.f, 0.f, 0.f, 1.f};
    return descriptor;
}

BOOST_AUTO_TEST_CASE(glycans)
{
    std::vector<const char*> argv{"brayns", "--http-server", "localhost:0",
                                  "--plugin", "BioExplorer"};
    brayns::Brayns brayns(argv.size(), argv.data());
    auto& scene = brayns.getEngine().getScene();
    Glycans glycans(scene, getDescriptor());

    BOOST_CHECK(glycans.getAtoms().size() == 291);
    BOOST_CHECK(glycans.getResidues().size() == 6);
    BOOST_CHECK(glycans.getSequencesAsString().size() == 0);
}
} // namespace tests