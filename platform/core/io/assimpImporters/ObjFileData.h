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

#pragma once
#ifndef OBJ_FILEDATA_H_INC
#define OBJ_FILEDATA_H_INC

#include <assimp/mesh.h>
#include <assimp/types.h>
#include <map>
#include <vector>

namespace Assimp
{
namespace ObjFile
{
struct Object;
struct Face;
struct Material;
typedef float ai_real;

// ------------------------------------------------------------------------------------------------
//! \struct Face
//! \brief  Data structure for a simple obj-face, describes discredit,l.ation
//! and materials
// ------------------------------------------------------------------------------------------------
struct Face
{
    typedef std::vector<unsigned int> IndexArray;

    //! Primitive type
    aiPrimitiveType m_PrimitiveType;
    //! Vertex indices
    IndexArray m_vertices;
    //! Normal indices
    IndexArray m_normals;
    //! Texture coordinates indices
    IndexArray m_texturCoords;
    //! Pointer to assigned material
    Material *m_pMaterial;

    //! \brief  Default constructor
    Face(aiPrimitiveType pt = aiPrimitiveType_POLYGON)
        : m_PrimitiveType(pt)
        , m_vertices()
        , m_normals()
        , m_texturCoords()
        , m_pMaterial(0L)
    {
        // empty
    }

    //! \brief  Destructor
    ~Face()
    {
        // empty
    }
};

// ------------------------------------------------------------------------------------------------
//! \struct Object
//! \brief  Stores all objects of an obj-file object definition
// ------------------------------------------------------------------------------------------------
struct Object
{
    enum ObjectType
    {
        ObjType,
        GroupType
    };

    //! Object name
    std::string m_strObjName;
    //! Transformation matrix, stored in OpenGL format
    aiMatrix4x4 m_Transformation;
    //! All sub-objects referenced by this object
    std::vector<Object *> m_SubObjects;
    /// Assigned meshes
    std::vector<unsigned int> m_Meshes;

    //! \brief  Default constructor
    Object()
        : m_strObjName("")
    {
        // empty
    }

    //! \brief  Destructor
    ~Object()
    {
        for (std::vector<Object *>::iterator it = m_SubObjects.begin(); it != m_SubObjects.end(); ++it)
        {
            delete *it;
        }
    }
};

// ------------------------------------------------------------------------------------------------
//! \struct Material
//! \brief  Data structure to store all material specific data
// ------------------------------------------------------------------------------------------------
struct Material
{
    //! Name of material description
    aiString MaterialName;

    //! Texture names
    aiString texture;
    aiString textureSpecular;
    aiString textureAmbient;
    aiString textureEmissive;
    aiString textureBump;
    aiString textureNormal;
    aiString textureReflection[6];
    aiString textureSpecularity;
    aiString textureOpacity;
    aiString textureDisp;

    enum TextureType
    {
        TextureDiffuseType = 0,
        TextureSpecularType,
        TextureAmbientType,
        TextureEmissiveType,
        TextureBumpType,
        TextureNormalType,
        TextureReflectionSphereType,
        TextureReflectionCubeTopType,
        TextureReflectionCubeBottomType,
        TextureReflectionCubeFrontType,
        TextureReflectionCubeBackType,
        TextureReflectionCubeLeftType,
        TextureReflectionCubeRightType,
        TextureSpecularityType,
        TextureOpacityType,
        TextureDispType,
        TextureTypeCount
    };
    bool clamp[TextureTypeCount];

    //! Ambient color
    aiColor3D ambient;
    //! Diffuse color
    aiColor3D diffuse;
    //! Specular color
    aiColor3D specular;
    //! Emissive color
    aiColor3D emissive;
    //! Alpha value
    ai_real alpha;
    //! Shineness factor
    ai_real shineness;
    //! Illumination model
    int illumination_model;
    //! Index of refraction
    ai_real ior;
    //! Transparency color
    aiColor3D transparent;

    //! Constructor
    Material()
        : diffuse(ai_real(0.6), ai_real(0.6), ai_real(0.6))
        , alpha(ai_real(1.0))
        , shineness(ai_real(0.0))
        , illumination_model(1)
        , ior(ai_real(1.0))
        , transparent(ai_real(1.0), ai_real(1.0), ai_real(1.0))
    {
        // empty
        for (size_t i = 0; i < TextureTypeCount; ++i)
        {
            clamp[i] = false;
        }
    }

    // Destructor
    ~Material()
    {
        // empty
    }
};

// ------------------------------------------------------------------------------------------------
//! \struct Mesh
//! \brief  Data structure to store a mesh
// ------------------------------------------------------------------------------------------------
struct Mesh
{
    static const unsigned int NoMaterial = ~0u;
    /// The name for the mesh
    std::string m_name;
    /// Array with pointer to all stored faces
    std::vector<Face *> m_Faces;
    /// Assigned material
    Material *m_pMaterial;
    /// Number of stored indices.
    unsigned int m_uiNumIndices;
    /// Number of UV
    unsigned int m_uiUVCoordinates[AI_MAX_NUMBER_OF_TEXTURECOORDS];
    /// Material index.
    unsigned int m_uiMaterialIndex;
    /// True, if normals are stored.
    bool m_hasNormals;
    /// True, if vertex colors are stored.
    bool m_hasVertexColors;

    /// Constructor
    explicit Mesh(const std::string &name)
        : m_name(name)
        , m_pMaterial(NULL)
        , m_uiNumIndices(0)
        , m_uiMaterialIndex(NoMaterial)
        , m_hasNormals(false)
    {
        memset(m_uiUVCoordinates, 0, sizeof(unsigned int) * AI_MAX_NUMBER_OF_TEXTURECOORDS);
    }

    /// Destructor
    ~Mesh()
    {
        for (std::vector<Face *>::iterator it = m_Faces.begin(); it != m_Faces.end(); ++it)
        {
            delete *it;
        }
    }
};

// ------------------------------------------------------------------------------------------------
//! \struct Model
//! \brief  Data structure to store all obj-specific model datas
// ------------------------------------------------------------------------------------------------
struct Model
{
    typedef std::map<std::string, std::vector<unsigned int> *> GroupMap;
    typedef std::map<std::string, std::vector<unsigned int> *>::iterator GroupMapIt;
    typedef std::map<std::string, std::vector<unsigned int> *>::const_iterator ConstGroupMapIt;

    //! Model name
    std::string m_ModelName;
    //! List ob assigned objects
    std::vector<Object *> m_Objects;
    //! Pointer to current object
    ObjFile::Object *m_pCurrent;
    //! Pointer to current material
    ObjFile::Material *m_pCurrentMaterial;
    //! Pointer to default material
    ObjFile::Material *m_pDefaultMaterial;
    //! Vector with all generated materials
    std::vector<std::string> m_MaterialLib;
    //! Vector with all generated vertices
    std::vector<aiVector3D> m_Vertices;
    //! vector with all generated normals
    std::vector<aiVector3D> m_Normals;
    //! vector with all vertex colors
    std::vector<aiVector3D> m_VertexColors;
    //! Group map
    GroupMap m_Groups;
    //! Group to face id assignment
    std::vector<unsigned int> *m_pGroupFaceIDs;
    //! Active group
    std::string m_strActiveGroup;
    //! Vector with generated texture coordinates
    std::vector<aiVector3D> m_TextureCoord;
    //! Current mesh instance
    Mesh *m_pCurrentMesh;
    //! Vector with stored meshes
    std::vector<Mesh *> m_Meshes;
    //! Material map
    std::map<std::string, Material *> m_MaterialMap;

    //! \brief  The default class constructor
    Model()
        : m_ModelName("")
        , m_pCurrent(NULL)
        , m_pCurrentMaterial(NULL)
        , m_pDefaultMaterial(NULL)
        , m_pGroupFaceIDs(NULL)
        , m_strActiveGroup("")
        , m_pCurrentMesh(NULL)
    {
        // empty
    }

    //! \brief  The class destructor
    ~Model()
    {
        // Clear all stored object instances
        for (std::vector<Object *>::iterator it = m_Objects.begin(); it != m_Objects.end(); ++it)
        {
            delete *it;
        }
        m_Objects.clear();

        // Clear all stored mesh instances
        for (std::vector<Mesh *>::iterator it = m_Meshes.begin(); it != m_Meshes.end(); ++it)
        {
            delete *it;
        }
        m_Meshes.clear();

        for (GroupMapIt it = m_Groups.begin(); it != m_Groups.end(); ++it)
        {
            delete it->second;
        }
        m_Groups.clear();

        for (std::map<std::string, Material *>::iterator it = m_MaterialMap.begin(); it != m_MaterialMap.end(); ++it)
        {
            delete it->second;
        }
    }
};

// ------------------------------------------------------------------------------------------------

} // Namespace ObjFile
} // Namespace Assimp

#endif // OBJ_FILEDATA_H_INC
