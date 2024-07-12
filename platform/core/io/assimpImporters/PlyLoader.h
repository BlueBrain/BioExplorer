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

/** @file  PLYLoader.h
 *  @brief Declaration of the .ply importer class.
 */
#ifndef AI_PLYLOADER_H_INCLUDED
#define AI_PLYLOADER_H_INCLUDED

#include "BaseImporter.h"
#include "PlyParser.h"
#include <assimp/types.h>
#include <vector>

struct aiNode;
struct aiMaterial;
struct aiMesh;

namespace Assimp
{
using namespace PLY;

// ---------------------------------------------------------------------------
/** Importer class to load the stanford PLY file format
 */
class PLYImporter : public BaseImporter
{
public:
    PLYImporter();
    ~PLYImporter();

public:
    // -------------------------------------------------------------------
    /** Returns whether the class can handle the format of the given file.
     * See BaseImporter::CanRead() for details.
     */
    bool CanRead(const std::string& pFile, IOSystem* pIOHandler, bool checkSig) const;

protected:
    // -------------------------------------------------------------------
    /** Return importer meta information.
     * See #BaseImporter::GetInfo for the details
     */
    const aiImporterDesc* GetInfo() const;

    // -------------------------------------------------------------------
    /** Imports the given file into the given scene structure.
     * See BaseImporter::InternReadFile() for details
     */
    void InternReadFile(const std::string& pFile, aiScene* pScene, IOSystem* pIOHandler);

protected:
    // -------------------------------------------------------------------
    /** Extract vertices from the DOM
     */
    void LoadVertices(std::vector<aiVector3D>* pvOut, bool p_bNormals = false);

    // -------------------------------------------------------------------
    /** Extract vertex color channels from the DOM
     */
    void LoadVertexColor(std::vector<aiColor4D>* pvOut);

    // -------------------------------------------------------------------
    /** Extract texture coordinate channels from the DOM
     */
    void LoadTextureCoordinates(std::vector<aiVector2D>* pvOut);

    // -------------------------------------------------------------------
    /** Extract a face list from the DOM
     */
    void LoadFaces(std::vector<PLY::Face>* pvOut);

    // -------------------------------------------------------------------
    /** Extract a material list from the DOM
     */
    void LoadMaterial(std::vector<aiMaterial*>* pvOut);

    // -------------------------------------------------------------------
    /** Validate material indices, replace default material identifiers
     */
    void ReplaceDefaultMaterial(std::vector<PLY::Face>* avFaces, std::vector<aiMaterial*>* avMaterials);

    // -------------------------------------------------------------------
    /** Convert all meshes into our ourer representation
     */
    void ConvertMeshes(std::vector<PLY::Face>* avFaces, const std::vector<aiVector3D>* avPositions,
                       const std::vector<aiVector3D>* avNormals, const std::vector<aiColor4D>* avColors,
                       const std::vector<aiVector2D>* avTexCoords, const std::vector<aiMaterial*>* avMaterials,
                       std::vector<aiMesh*>* avOut);

    // -------------------------------------------------------------------
    /** Static helper to parse a color from four single channels in
     */
    static void GetMaterialColor(const std::vector<PLY::PropertyInstance>& avList, unsigned int aiPositions[4],
                                 PLY::EDataType aiTypes[4], aiColor4D* clrOut);

    // -------------------------------------------------------------------
    /** Static helper to parse a color channel value. The input value
     *  is normalized to 0-1.
     */
    static ai_real NormalizeColorValue(PLY::PropertyInstance::ValueUnion val, PLY::EDataType eType);

    /** Buffer to hold the loaded file */
    unsigned char* mBuffer;

    /** Document object model representation extracted from the file */
    PLY::DOM* pcDOM;
};

} // end of namespace Assimp

#endif // AI_3DSIMPORTER_H_INC
