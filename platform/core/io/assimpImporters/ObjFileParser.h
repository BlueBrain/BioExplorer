/*
    Copyright 2006 - 2017 Blue Brain Project / EPFL

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
#ifndef OBJ_FILEPARSER_H_INC
#define OBJ_FILEPARSER_H_INC

#include "IOStreamBuffer.h"
#include <assimp/mesh.h>
#include <assimp/vector2.h>
#include <assimp/vector3.h>
#include <map>
#include <string>
#include <vector>

namespace Assimp
{
namespace ObjFile
{
struct Model;
struct Object;
struct Material;
struct Point3;
struct Point2;
} // namespace ObjFile

class ObjFileImporter;
class IOSystem;
class ProgressHandler;

/// \class  ObjFileParser
/// \brief  Parser for a obj waveform file
class ASSIMP_API ObjFileParser
{
public:
    static const size_t Buffersize = 4096;
    typedef std::vector<char> DataArray;
    typedef std::vector<char>::iterator DataArrayIt;
    typedef std::vector<char>::const_iterator ConstDataArrayIt;

public:
    /// @brief  The default constructor.
    ObjFileParser();
    /// @brief  Constructor with data array.
    ObjFileParser(IOStreamBuffer<char> &streamBuffer, const std::string &modelName, IOSystem *io,
                  ProgressHandler *progress, const std::string &originalObjFileName);
    /// @brief  Destructor
    ~ObjFileParser();
    /// @brief  If you want to load in-core data.
    void setBuffer(std::vector<char> &buffer);
    /// @brief  Model getter.
    ObjFile::Model *GetModel() const;

protected:
    /// Parse the loaded file
    void parseFile(IOStreamBuffer<char> &streamBuffer);
    /// Method to copy the new delimited word in the current line.
    void copyNextWord(char *pBuffer, size_t length);
    /// Method to copy the new line.
    //    void copyNextLine(char *pBuffer, size_t length);
    /// Get the number of components in a line.
    size_t getNumComponentsInDataDefinition();
    /// Stores the vector
    void getVector(std::vector<aiVector3D> &point3d_array);
    /// Stores the following 3d vector.
    void getVector3(std::vector<aiVector3D> &point3d_array);
    /// Stores the following homogeneous vector as a 3D vector
    void getHomogeneousVector3(std::vector<aiVector3D> &point3d_array);
    /// Stores the following two 3d vectors on the line.
    void getTwoVectors3(std::vector<aiVector3D> &point3d_array_a, std::vector<aiVector3D> &point3d_array_b);
    /// Stores the following 3d vector.
    void getVector2(std::vector<aiVector2D> &point2d_array);
    /// Stores the following face.
    void getFace(aiPrimitiveType type);
    /// Reads the material description.
    void getMaterialDesc();
    /// Gets a comment.
    void getComment();
    /// Gets a a material library.
    void getMaterialLib();
    /// Creates a new material.
    void getNewMaterial();
    /// Gets the group name from file.
    void getGroupName();
    /// Gets the group number from file.
    void getGroupNumber();
    /// Gets the group number and resolution from file.
    void getGroupNumberAndResolution();
    /// Returns the index of the material. Is -1 if not material was found.
    int getMaterialIndex(const std::string &strMaterialName);
    /// Parse object name
    void getObjectName();
    /// Creates a new object.
    void createObject(const std::string &strObjectName);
    /// Creates a new mesh.
    void createMesh(const std::string &meshName);
    /// Returns true, if a new mesh instance must be created.
    bool needsNewMesh(const std::string &rMaterialName);
    /// Error report in token
    void reportErrorTokenInFace();

private:
    // Copy and assignment constructor should be private
    // because the class contains pointer to allocated memory
    ObjFileParser(const ObjFileParser &rhs);
    ObjFileParser &operator=(const ObjFileParser &rhs);

    /// Default material name
    static const std::string DEFAULT_MATERIAL;
    //! Iterator to current position in buffer
    DataArrayIt m_DataIt;
    //! Iterator to end position of buffer
    DataArrayIt m_DataItEnd;
    //! Pointer to model instance
    ObjFile::Model *m_pModel;
    //! Current line (for debugging)
    unsigned int m_uiLine;
    //! Helper buffer
    char m_buffer[Buffersize];
    /// Pointer to IO system instance.
    IOSystem *m_pIO;
    //! Pointer to progress handler
    ProgressHandler *m_progress;
    /// Path to the current model, name of the obj file where the buffer comes
    /// from
    const std::string m_originalObjFileName;
};

} // Namespace Assimp

#endif
