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

/**
 * @brief This is a C++ function that takes a filename as input, reads the
 * entire content of the file and returns it as a string. It first initializes
 * an input file stream object named file with the provided filename. Then it
 * checks whether the file was successfully opened by checking the boolean value
 * of file. If the file was opened successfully, it uses the std::ostringstream
 * class to create a string stream and copies all the data from the file into
 * this stream by using the rdbuf() function. Finally, it returns the string
 * content of the file. If the file failed to open, it throws an exception using
 * the PLUGIN_THROW macro along with an error message explaining the reason for
 * the failure.
 *
 * @param filename
 * @return std::string
 */
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

/**
 * @brief The code defines a function called getAssemblyDescriptor() that
returns an object of type AssemblyDetails. Within the function, a new
AssemblyDetails object is declared and assigned to a variable called descriptor.
Then, the name field of the descriptor object is set to the string "assembly".
The shape field is set to the value AssemblyShape::mesh, which is an enum value
defined elsewhere in the codebase. Finally, the shapeMeshContents field is set
to the result of calling another function called getFileContents() with an
argument that concatenates the value of a variable called folder with the string
"obj/suzanne.obj".


This code is likely part of a larger program that describes the properties of a
3D object, possibly for rendering or visualization. The getAssemblyDescriptor()
function is responsible for constructing an AssemblyDetails object with specific
values for its properties. The getFileContents() function is likely used to read
in the contents of a 3D mesh file in OBJ format and assign it to the
shapeMeshContents field.
 *
 * @return AssemblyDetails
 */
AssemblyDetails getAssemblyDescriptor()
{
    AssemblyDetails descriptor;
    descriptor.name = "assembly";
    descriptor.shape = AssemblyShape::mesh;
    descriptor.shapeMeshContents = getFileContents(folder + "obj/suzanne.obj");
    return descriptor;
}

/**
 * @brief The given code defines a function named getMembraneDescriptor() which
returns an object of type MembraneDetails.


The function creates an instance of MembraneDetails using the default
constructor. It then sets some of its member variables: assemblyName and name
are set to the string "test", lipidContents is set by calling getFileContents()
which likely returns the contents of a file whose path is specified by the
folder variable concatenated with the string "pdb/membrane/popc.pdb".


The representation member is set to an enumerated type value
ProteinRepresentation::atoms and animationParams member is initialized with an
empty initializer list. Finally, the function returns the created
MembraneDetails object descriptor.
 *
 * @return MembraneDetails
 */
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
