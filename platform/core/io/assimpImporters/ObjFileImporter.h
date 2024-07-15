/*
    Copyright 2006 - 2024 Blue Brain Project / EPFL

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
#ifndef OBJ_FILE_IMPORTER_H_INC
#define OBJ_FILE_IMPORTER_H_INC

#include "BaseImporter.h"
#include <assimp/material.h>
#include <vector>

struct aiMesh;
struct aiNode;

namespace Assimp
{
namespace ObjFile
{
struct Object;
struct Model;
} // namespace ObjFile

// ------------------------------------------------------------------------------------------------
/// \class  ObjFileImporter
/// \brief  Imports a waveform obj file
// ------------------------------------------------------------------------------------------------
class ObjFileImporter : public BaseImporter
{
public:
    /// \brief  Default constructor
    ObjFileImporter();

    /// \brief  Destructor
    ~ObjFileImporter();

public:
    /// \brief  Returns whether the class can handle the format of the given
    /// file.
    /// \remark See BaseImporter::CanRead() for details.
    bool CanRead(const std::string& pFile, IOSystem* pIOHandler, bool checkSig) const;

private:
    //! \brief  Appends the supported extension.
    const aiImporterDesc* GetInfo() const;

    //! \brief  File import implementation.
    void InternReadFile(const std::string& pFile, aiScene* pScene, IOSystem* pIOHandler);

    //! \brief  Create the data from imported content.
    void CreateDataFromImport(const ObjFile::Model* pModel, aiScene* pScene);

    //! \brief  Creates all nodes stored in imported content.
    aiNode* createNodes(const ObjFile::Model* pModel, const ObjFile::Object* pData, aiNode* pParent, aiScene* pScene,
                        std::vector<aiMesh*>& MeshArray);

    //! \brief  Creates topology data like faces and meshes for the geometry.
    aiMesh* createTopology(const ObjFile::Model* pModel, const ObjFile::Object* pData, unsigned int uiMeshIndex);

    //! \brief  Creates vertices from model.
    void createVertexArray(const ObjFile::Model* pModel, const ObjFile::Object* pCurrentObject,
                           unsigned int uiMeshIndex, aiMesh* pMesh, unsigned int numIndices);

    //! \brief  Object counter helper method.
    void countObjects(const std::vector<ObjFile::Object*>& rObjects, int& iNumMeshes);

    //! \brief  Material creation.
    void createMaterials(const ObjFile::Model* pModel, aiScene* pScene);

    /// @brief  Adds special property for the used texture mapping mode of the
    /// model.
    void addTextureMappingModeProperty(aiMaterial* mat, aiTextureType type, int clampMode = 1, int index = 0);

    //! \brief  Appends a child node to a parent node and updates the data
    //! structures.
    void appendChildToParentNode(aiNode* pParent, aiNode* pChild);

private:
    //! Data buffer
    std::vector<char> m_Buffer;
    //! Pointer to root object instance
    ObjFile::Object* m_pRootObject;
};

// ------------------------------------------------------------------------------------------------

} // Namespace Assimp

#endif
