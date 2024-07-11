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
#ifndef OBJFILEMTLIMPORTER_H_INC
#define OBJFILEMTLIMPORTER_H_INC

#include <assimp/defs.h>
#include <string>
#include <vector>

struct aiColor3D;
struct aiString;

namespace Assimp
{
namespace ObjFile
{
struct Model;
struct Material;
} // namespace ObjFile
typedef float ai_real;

/**
 *  @class  ObjFileMtlImporter
 *  @brief  Loads the material description from a mtl file.
 */
class ObjFileMtlImporter
{
public:
    static const size_t BUFFERSIZE = 2048;
    typedef std::vector<char> DataArray;
    typedef std::vector<char>::iterator DataArrayIt;
    typedef std::vector<char>::const_iterator ConstDataArrayIt;

public:
    //! \brief  Default constructor
    ObjFileMtlImporter(std::vector<char> &buffer, const std::string &strAbsPath, ObjFile::Model *pModel);

    //! \brief  DEstructor
    ~ObjFileMtlImporter();

private:
    /// Copy constructor, empty.
    ObjFileMtlImporter(const ObjFileMtlImporter &rOther);
    /// \brief  Assignment operator, returns only a reference of this instance.
    ObjFileMtlImporter &operator=(const ObjFileMtlImporter &rOther);
    /// Load the whole material description
    void load();
    /// Get color data.
    void getColorRGBA(aiColor3D *pColor);
    /// Get illumination model from loaded data
    void getIlluminationModel(int &illum_model);
    /// Gets a float value from data.
    void getFloatValue(ai_real &value);
    /// Creates a new material from loaded data.
    void createMaterial();
    /// Get texture name from loaded data.
    void getTexture();
    void getTextureOption(bool &clamp, int &clampIndex, aiString *&out);

private:
    //! Absolute pathname
    std::string m_strAbsPath;
    //! Data iterator showing to the current position in data buffer
    DataArrayIt m_DataIt;
    //! Data iterator to end of buffer
    DataArrayIt m_DataItEnd;
    //! USed model instance
    ObjFile::Model *m_pModel;
    //! Current line in file
    unsigned int m_uiLine;
    //! Helper buffer
    char m_buffer[BUFFERSIZE];
};

// ------------------------------------------------------------------------------------------------

} // Namespace Assimp

#endif // OBJFILEMTLIMPORTER_H_INC
