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
#include <plugin/molecularsystems/Membrane.h>
#include <plugin/molecularsystems/Protein.h>

#include <brayns/Brayns.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Scene.h>

#define BOOST_TEST_MODULE mesh
#include <boost/test/unit_test.hpp>

#include <fstream>

namespace tests
{
using namespace bioexplorer;
using namespace molecularsystems;

const std::string folder = "./bioexplorer/pythonsdk/tests/test_files/";

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
    descriptor.shape = AssemblyShape::mesh;
    descriptor.shapeMeshContents = getFileContents(folder + "obj/suzanne.obj");
    return descriptor;
}

MembraneDetails getMembraneDescriptor()
{
    MembraneDetails descriptor;

    descriptor.assemblyName = "test";
    descriptor.name = "test";
    descriptor.lipidContents =
        getFileContents(folder + "pdb/membrane/popc.pdb");
    descriptor.representation = ProteinRepresentation::atoms;
    descriptor.animationParams = {};
    return descriptor;
}

BOOST_AUTO_TEST_CASE(meshBasedMembrane)
{
    std::vector<const char*> argv{
        "brayns", "--http-server", "localhost:0", "--plugin",
        "BioExplorer --db-name=bioexplorer --db-user=brayns "
        "--db-password=brayns --db-host=localhost --db-port=5432"};
    brayns::Brayns brayns(argv.size(), argv.data());
    auto& scene = brayns.getEngine().getScene();

    Assembly assembly(scene, getAssemblyDescriptor());
    assembly.addMembrane(getMembraneDescriptor());

    BOOST_CHECK(assembly.getMembrane()
                    ->getLipids()
                    .begin()
                    ->second->getAtoms()
                    .size() == 426);

    BOOST_CHECK(assembly.isInside(Vector3d(0.0, 0.0, 0.0)));
}
} // namespace tests
