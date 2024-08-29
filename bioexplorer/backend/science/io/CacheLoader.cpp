/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include <science/common/Logs.h>

#include "CacheLoader.h"

#include <science/api/Params.h>
#include <science/common/Assembly.h>
#include <science/common/GeneralSettings.h>
#include <science/common/Utils.h>
#include <science/molecularsystems/Protein.h>

#include <platform/core/common/scene/ClipPlane.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>

#ifdef USE_LASLIB
#include <laswriter.hpp>
#endif

#include <fstream>

using namespace core;

namespace
{
bool inBounds(const Vector3d& point, const Boxd& bounds)
{
    const auto mi = bounds.getMin();
    const auto ma = bounds.getMax();
    return point.x >= mi.x && point.x < ma.x && point.y >= mi.y && point.y < ma.y && point.z >= mi.z && point.z < ma.z;
}
} // namespace

namespace bioexplorer
{
namespace io
{
using namespace common;
using namespace db;

const std::string LOADER_NAME = "BioExplorer cache loader";
const std::string SUPPORTED_EXTENTION_BIOEXPLORER = "bioexplorer";

const size_t CACHE_VERSION_1 = 1;

CacheLoader::CacheLoader(Scene& scene, PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
}

std::string CacheLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> CacheLoader::getSupportedStorage() const
{
    return {SUPPORTED_EXTENTION_BIOEXPLORER};
}

bool CacheLoader::isSupported(const std::string& /*filename*/, const std::string& extension) const
{
    const std::set<std::string> types = {SUPPORTED_EXTENTION_BIOEXPLORER};
    return types.find(extension) != types.end();
}

ModelDescriptorPtr CacheLoader::importFromBlob(Blob&& /*blob*/, const LoaderProgress& /*callback*/,
                                               const PropertyMap& /*properties*/) const
{
    PLUGIN_THROW("Loading cache from blob is not supported");
}

ModelDescriptorPtr CacheLoader::_importModel(std::stringstream& buffer, const int32_t brickId) const
{
    auto model = _scene.createModel();

    // Geometry
    size_t nbSpheres = 0;
    size_t nbCylinders = 0;
    size_t nbCones = 0;
    size_t nbMeshes = 0;
    size_t nbVertices = 0;
    size_t nbIndices = 0;
    size_t nbNormals = 0;
    size_t nbTexCoords = 0;

    // Name
    const auto name = _readString(buffer);

    // Path
    const auto path = _readString(buffer);

    // Metadata
    size_t nbElements;
    ModelMetadata metadata;
    buffer.read((char*)&nbElements, sizeof(size_t));
    for (size_t i = 0; i < nbElements; ++i)
        metadata[_readString(buffer)] = _readString(buffer);

    if (brickId != UNDEFINED_BOX_ID)
    {
        metadata.clear();
        metadata[METADATA_BRICK_ID] = std::to_string(brickId);
    }

    // Instances
    std::vector<Transformation> transformations;
    buffer.read((char*)&nbElements, sizeof(size_t));
    for (size_t i = 0; i < nbElements; ++i)
    {
        Transformation tf;
        Vector3d t;
        Vector3d rc;
        Vector3d s;
        Quaterniond q;

        buffer.read((char*)&t, sizeof(Vector3d));
        tf.setTranslation(t);
        buffer.read((char*)&rc, sizeof(Vector3d));
        tf.setRotationCenter(rc);
        buffer.read((char*)&q, sizeof(Quaterniond));
        tf.setRotation(q);
        buffer.read((char*)&s, sizeof(Vector3d));
        tf.setScale(s);
        transformations.push_back(tf);
    }

    size_t nbMaterials;
    buffer.read((char*)&nbMaterials, sizeof(size_t));

    // Materials
    size_t materialId;
    for (size_t i = 0; i < nbMaterials; ++i)
    {
        buffer.read((char*)&materialId, sizeof(size_t));

        auto name = _readString(buffer);
        auto material = model->createMaterial(materialId, name);

        Vector3d value3f;
        buffer.read((char*)&value3f, sizeof(Vector3d));
        material->setDiffuseColor(value3f);
        buffer.read((char*)&value3f, sizeof(Vector3d));
        material->setSpecularColor(value3f);
        double value;
        buffer.read((char*)&value, sizeof(double));
        material->setSpecularExponent(value);
        buffer.read((char*)&value, sizeof(double));
        material->setReflectionIndex(value);
        buffer.read((char*)&value, sizeof(double));
        material->setOpacity(value);
        buffer.read((char*)&value, sizeof(double));
        material->setRefractionIndex(value);
        buffer.read((char*)&value, sizeof(double));
        material->setEmission(value);
        buffer.read((char*)&value, sizeof(double));
        material->setGlossiness(value);

        core::PropertyMap props;
        double userParameter;
        buffer.read((char*)&userParameter, sizeof(double));
        material->setUserParameter(value);

        int32_t shadingMode;
        buffer.read((char*)&shadingMode, sizeof(int32_t));
        material->setShadingMode(static_cast<MaterialShadingMode>(shadingMode));

        int32_t chameleonMode;
        buffer.read((char*)&chameleonMode, sizeof(int32_t));
        material->setChameleonMode(static_cast<MaterialChameleonMode>(chameleonMode));

        int32_t nodeId;
        buffer.read((char*)&nodeId, sizeof(int32_t));
        material->setNodeId(nodeId);
    }

    uint64_t bufferSize{0};

    // Spheres
    buffer.read((char*)&nbSpheres, sizeof(size_t));
    for (size_t i = 0; i < nbSpheres; ++i)
    {
        buffer.read((char*)&materialId, sizeof(size_t));
        buffer.read((char*)&nbElements, sizeof(size_t));
        auto& spheres = model->getSpheres()[materialId];
        spheres.resize(nbElements);
        bufferSize = nbElements * sizeof(Sphere);
        buffer.read((char*)spheres.data(), bufferSize);
    }

    // Cylinders
    buffer.read((char*)&nbCylinders, sizeof(size_t));
    for (size_t i = 0; i < nbCylinders; ++i)
    {
        buffer.read((char*)&materialId, sizeof(size_t));
        buffer.read((char*)&nbElements, sizeof(size_t));
        auto& cylinders = model->getCylinders()[materialId];
        cylinders.resize(nbElements);
        bufferSize = nbElements * sizeof(Cylinder);
        buffer.read((char*)cylinders.data(), bufferSize);
    }

    // Cones
    buffer.read((char*)&nbCones, sizeof(size_t));
    for (size_t i = 0; i < nbCones; ++i)
    {
        buffer.read((char*)&materialId, sizeof(size_t));
        buffer.read((char*)&nbElements, sizeof(size_t));
        auto& cones = model->getCones()[materialId];
        cones.resize(nbElements);
        bufferSize = nbElements * sizeof(Cone);
        buffer.read((char*)cones.data(), bufferSize);
    }

    // Meshes
    buffer.read((char*)&nbMeshes, sizeof(size_t));
    for (size_t i = 0; i < nbMeshes; ++i)
    {
        buffer.read((char*)&materialId, sizeof(size_t));
        auto& meshes = model->getTriangleMeshes()[materialId];
        // Vertices
        buffer.read((char*)&nbVertices, sizeof(size_t));
        if (nbVertices != 0)
        {
            bufferSize = nbVertices * sizeof(Vector3d);
            meshes.vertices.resize(nbVertices);
            buffer.read((char*)meshes.vertices.data(), bufferSize);
        }

        // Indices
        buffer.read((char*)&nbIndices, sizeof(size_t));
        if (nbIndices != 0)
        {
            bufferSize = nbIndices * sizeof(Vector3ui);
            meshes.indices.resize(nbIndices);
            buffer.read((char*)meshes.indices.data(), bufferSize);
        }

        // Normals
        buffer.read((char*)&nbNormals, sizeof(size_t));
        if (nbNormals != 0)
        {
            bufferSize = nbNormals * sizeof(Vector3d);
            meshes.normals.resize(nbNormals);
            buffer.read((char*)meshes.normals.data(), bufferSize);
        }

        // Texture coordinates
        buffer.read((char*)&nbTexCoords, sizeof(size_t));
        if (nbTexCoords != 0)
        {
            bufferSize = nbTexCoords * sizeof(Vector2f);
            meshes.textureCoordinates.resize(nbTexCoords);
            buffer.read((char*)meshes.textureCoordinates.data(), bufferSize);
        }
    }

    // Streamlines
    size_t nbStreamlines;
    auto& streamlines = model->getStreamlines();
    buffer.read((char*)&nbStreamlines, sizeof(size_t));
    for (size_t i = 0; i < nbStreamlines; ++i)
    {
        StreamlinesData streamlineData;
        // Id
        size_t id;
        buffer.read((char*)&id, sizeof(size_t));

        // Vertex
        buffer.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(Vector4f);
        streamlineData.vertex.resize(nbElements);
        buffer.read((char*)streamlineData.vertex.data(), bufferSize);

        // Vertex Color
        buffer.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(Vector4f);
        streamlineData.vertexColor.resize(nbElements);
        buffer.read((char*)streamlineData.vertexColor.data(), bufferSize);

        // Indices
        buffer.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(int32_t);
        streamlineData.indices.resize(nbElements);
        buffer.read((char*)streamlineData.indices.data(), bufferSize);

        streamlines[id] = streamlineData;
    }

    // SDF geometry
    auto& sdfData = model->getSDFGeometryData();
    buffer.read((char*)&nbElements, sizeof(size_t));

    if (nbElements > 0)
    {
        // Geometries
        sdfData.geometries.resize(nbElements);
        bufferSize = nbElements * sizeof(SDFGeometry);
        buffer.read((char*)sdfData.geometries.data(), bufferSize);

        // SDF Indices
        buffer.read((char*)&nbElements, sizeof(size_t));
        for (size_t i = 0; i < nbElements; ++i)
        {
            buffer.read((char*)&materialId, sizeof(size_t));
            size_t size;
            buffer.read((char*)&size, sizeof(size_t));
            bufferSize = size * sizeof(uint64_t);
            sdfData.geometryIndices[materialId].resize(size);
            buffer.read((char*)sdfData.geometryIndices[materialId].data(), bufferSize);
        }

        // Neighbours
        buffer.read((char*)&nbElements, sizeof(size_t));
        sdfData.neighbours.resize(nbElements);

        for (size_t i = 0; i < nbElements; ++i)
        {
            size_t size;
            buffer.read((char*)&size, sizeof(size_t));
            bufferSize = size * sizeof(uint64_t);
            sdfData.neighbours[i].resize(size);
            buffer.read((char*)sdfData.neighbours[i].data(), bufferSize);
        }

        // Neighbours flat
        buffer.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(uint64_t);
        sdfData.neighboursFlat.resize(nbElements);
        buffer.read((char*)sdfData.neighboursFlat.data(), bufferSize);
    }

    if (!transformations.empty())
    {
        auto modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), name, path, metadata);

        bool first{true};
        for (const auto& tf : transformations)
        {
            if (first)
            {
                modelDescriptor->setTransformation(tf);
                first = false;
            }

            const ModelInstance instance(true, false, tf);
            modelDescriptor->addInstance(instance);
        }

        const auto visible = GeneralSettings::getInstance()->getModelVisibilityOnCreation();
        modelDescriptor->setVisible(visible);
        return modelDescriptor;
    }
    return nullptr;
}

std::vector<ModelDescriptorPtr> CacheLoader::importModelsFromFile(const std::string& filename, const int32_t brickId,
                                                                  const LoaderProgress& callback,
                                                                  const PropertyMap& properties) const
{
    std::vector<ModelDescriptorPtr> modelDescriptors;
    PropertyMap props = _defaults;
    props.merge(properties);

    callback.updateProgress("Loading BioExplorer scene...", 0);
    PLUGIN_DEBUG("Loading models from cache file: " << filename);
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.good())
        PLUGIN_THROW("Could not open cache file " + filename);

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    // File version
    size_t version;
    buffer.read((char*)&version, sizeof(size_t));
    PLUGIN_DEBUG("Version: " << version);

    // Models
    size_t nbModels;
    buffer.read((char*)&nbModels, sizeof(size_t));
    PLUGIN_DEBUG("Models : " << nbModels);
    for (size_t i = 0; i < nbModels; ++i)
    {
        auto modelDescriptor = _importModel(buffer, brickId);
        if (modelDescriptor)
            modelDescriptors.push_back(modelDescriptor);

        callback.updateProgress("Loading models", double(i) / double(nbModels));
    }

    return modelDescriptors;
}

ModelDescriptorPtr CacheLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                  const PropertyMap& properties) const
{
    const auto modelDescriptors = importModelsFromFile(storage, UNDEFINED_BOX_ID, callback, properties);
    for (const auto modelDescriptor : modelDescriptors)
        _scene.addModel(modelDescriptor);

    return (!modelDescriptors.empty() ? modelDescriptors[0] : nullptr);
}

std::string CacheLoader::_readString(std::stringstream& buffer) const
{
    size_t size;
    buffer.read((char*)&size, sizeof(size_t));
    std::vector<char> str;
    str.resize(size + 1, 0);
    buffer.read(&str[0], size);
    return str.data();
}

bool CacheLoader::_exportModel(const ModelDescriptorPtr modelDescriptor, std::stringstream& buffer,
                               const Boxd& bounds) const
{
    uint64_t bufferSize{0};
    const auto& model = modelDescriptor->getModel();

    // Instances
    std::vector<Transformation> transformations;
    const auto& instances = modelDescriptor->getInstances();
    bool first{true};
    for (const auto& instance : instances)
    {
        if (first)
        {
            first = false;
            const auto& tf = modelDescriptor->getTransformation();
            if (!inBounds(tf.getTranslation(), bounds))
                continue;
            transformations.push_back(tf);
        }
        else
        {
            const auto& tf = instance.getTransformation();
            if (!inBounds(tf.getTranslation(), bounds))
                continue;
            transformations.push_back(tf);
        }
    }

    if (transformations.empty())
        return false;

    // Name
    const auto& name = modelDescriptor->getName();
    size_t size = name.length();
    buffer.write((char*)&size, sizeof(size_t));
    buffer.write((char*)name.c_str(), size);

    // Path
    const auto& path = modelDescriptor->getPath();
    size = path.length();
    buffer.write((char*)&size, sizeof(size_t));
    buffer.write((char*)path.c_str(), size);

    // Metadata
    auto metadata = modelDescriptor->getMetadata();
    size_t nbElements = metadata.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& data : metadata)
    {
        size = data.first.length();
        buffer.write((char*)&size, sizeof(size_t));
        buffer.write((char*)data.first.c_str(), size);
        size = data.second.length();
        buffer.write((char*)&size, sizeof(size_t));
        buffer.write((char*)data.second.c_str(), size);
    }

    // Instances
    nbElements = transformations.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& tf : transformations)
    {
        const auto& t = tf.getTranslation();
        buffer.write((char*)&t, sizeof(core::Vector3d));
        const auto& rc = tf.getRotationCenter();
        buffer.write((char*)&rc, sizeof(core::Vector3d));
        const auto& q = tf.getRotation();
        buffer.write((char*)&q, sizeof(core::Quaterniond));
        const auto& s = tf.getScale();
        buffer.write((char*)&s, sizeof(core::Vector3d));
    }

    // Materials
    const auto& materials = model.getMaterials();
    nbElements = materials.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& material : materials)
    {
        buffer.write((char*)&material.first, sizeof(size_t));

        auto name = material.second->getName();
        size_t size = name.length();
        buffer.write((char*)&size, sizeof(size_t));
        buffer.write((char*)name.c_str(), size);

        core::Vector3d value3f;
        value3f = material.second->getDiffuseColor();
        buffer.write((char*)&value3f, sizeof(core::Vector3d));
        value3f = material.second->getSpecularColor();
        buffer.write((char*)&value3f, sizeof(core::Vector3d));
        double value = material.second->getSpecularExponent();
        buffer.write((char*)&value, sizeof(double));
        value = material.second->getReflectionIndex();
        buffer.write((char*)&value, sizeof(double));
        value = material.second->getOpacity();
        buffer.write((char*)&value, sizeof(double));
        value = material.second->getRefractionIndex();
        buffer.write((char*)&value, sizeof(double));
        value = material.second->getEmission();
        buffer.write((char*)&value, sizeof(double));
        value = material.second->getGlossiness();
        buffer.write((char*)&value, sizeof(double));
        value = material.second->getUserParameter();
        buffer.write((char*)&value, sizeof(double));
        int32_t shadingMode = material.second->getShadingMode();
        buffer.write((char*)&shadingMode, sizeof(int32_t));

        int32_t chameleonMode = MaterialChameleonMode::undefined_chameleon_mode;
        try
        {
            shadingMode = material.second->getChameleonMode();
        }
        catch (const std::runtime_error&)
        {
        }
        buffer.write((char*)&chameleonMode, sizeof(int32_t));

        int32_t nodeId = 0;
        try
        {
            shadingMode = material.second->getNodeId();
        }
        catch (const std::runtime_error&)
        {
        }
        buffer.write((char*)&nodeId, sizeof(int32_t));
    }

    // Spheres
    const auto& spheresMap = model.getSpheres();
    nbElements = spheresMap.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& spheres : spheresMap)
    {
        const auto materialId = spheres.first;
        buffer.write((char*)&materialId, sizeof(size_t));

        const auto& data = spheres.second;
        nbElements = data.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Sphere);
        buffer.write((char*)data.data(), bufferSize);
    }

    // Cylinders
    const auto& cylindersMap = model.getCylinders();
    nbElements = cylindersMap.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& cylinders : cylindersMap)
    {
        const auto materialId = cylinders.first;
        buffer.write((char*)&materialId, sizeof(size_t));

        const auto& data = cylinders.second;
        nbElements = data.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Cylinder);
        buffer.write((char*)data.data(), bufferSize);
    }

    // Cones
    const auto& conesMap = model.getCones();
    nbElements = conesMap.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& cones : conesMap)
    {
        const auto materialId = cones.first;
        buffer.write((char*)&materialId, sizeof(size_t));

        const auto& data = cones.second;
        nbElements = data.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Cone);
        buffer.write((char*)data.data(), bufferSize);
    }

    // Meshes
    const auto& trianglesMap = model.getTriangleMeshes();
    nbElements = trianglesMap.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& meshes : trianglesMap)
    {
        const auto materialId = meshes.first;
        buffer.write((char*)&materialId, sizeof(size_t));

        const auto& data = meshes.second;

        // Vertices
        nbElements = data.vertices.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Vector3d);
        buffer.write((char*)data.vertices.data(), bufferSize);

        // Indices
        nbElements = data.indices.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Vector3ui);
        buffer.write((char*)data.indices.data(), bufferSize);

        // Normals
        nbElements = data.normals.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Vector3d);
        buffer.write((char*)data.normals.data(), bufferSize);

        // Texture coordinates
        nbElements = data.textureCoordinates.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Vector2f);
        buffer.write((char*)data.textureCoordinates.data(), bufferSize);
    }

    // Streamlines
    const auto& streamlines = model.getStreamlines();
    nbElements = streamlines.size();
    buffer.write((char*)&nbElements, sizeof(size_t));
    for (const auto& streamline : streamlines)
    {
        const auto& streamlineData = streamline.second;
        // Id
        size_t id = streamline.first;
        buffer.write((char*)&id, sizeof(size_t));

        // Vertex
        nbElements = streamlineData.vertex.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Vector4f);
        buffer.write((char*)streamlineData.vertex.data(), bufferSize);

        // Vertex Color
        nbElements = streamlineData.vertexColor.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(core::Vector4f);
        buffer.write((char*)streamlineData.vertexColor.data(), bufferSize);

        // Indices
        nbElements = streamlineData.indices.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(int32_t);
        buffer.write((char*)streamlineData.indices.data(), bufferSize);
    }

    // SDF geometry
    const auto& sdfData = model.getSDFGeometryData();
    nbElements = sdfData.geometries.size();
    buffer.write((char*)&nbElements, sizeof(size_t));

    if (nbElements > 0)
    {
        // Geometries
        bufferSize = nbElements * sizeof(core::SDFGeometry);
        buffer.write((char*)sdfData.geometries.data(), bufferSize);

        // SDF indices
        nbElements = sdfData.geometryIndices.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        for (const auto& geometryIndex : sdfData.geometryIndices)
        {
            size_t materialId = geometryIndex.first;
            buffer.write((char*)&materialId, sizeof(size_t));
            nbElements = geometryIndex.second.size();
            buffer.write((char*)&nbElements, sizeof(size_t));
            bufferSize = nbElements * sizeof(uint64_t);
            buffer.write((char*)geometryIndex.second.data(), bufferSize);
        }

        // Neighbours
        nbElements = sdfData.neighbours.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        for (const auto& neighbour : sdfData.neighbours)
        {
            nbElements = neighbour.size();
            buffer.write((char*)&nbElements, sizeof(size_t));
            bufferSize = nbElements * sizeof(size_t);
            buffer.write((char*)neighbour.data(), bufferSize);
        }

        // Neighbours flat
        nbElements = sdfData.neighboursFlat.size();
        buffer.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(uint64_t);
        buffer.write((char*)sdfData.neighboursFlat.data(), bufferSize);
    }
    return true;
}

void CacheLoader::exportToFile(const std::string& filename, const Boxd& bounds) const
{
    PLUGIN_DEBUG("Saving scene to BioExplorer file: " << filename);

    std::stringstream buffer;
    const size_t version = CACHE_VERSION_1;
    buffer.write((char*)&version, sizeof(size_t));

    const auto& modelDescriptors = _scene.getModelDescriptors();
    size_t nbModels = modelDescriptors.size();
    buffer.write((char*)&nbModels, sizeof(size_t));

    nbModels = 0;
    for (const auto& modelDescriptor : modelDescriptors)
        nbModels += (_exportModel(modelDescriptor, buffer, bounds) ? 1 : 0);

    buffer.seekp(sizeof(size_t), std::ios_base::beg);
    buffer.write((char*)&nbModels, sizeof(size_t));

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.good())
        PLUGIN_THROW("Could not create BioExplorer file " + filename);

    file.write((char*)buffer.str().c_str(), buffer.str().size());
    file.close();
}

std::vector<ModelDescriptorPtr> CacheLoader::importBrickFromDB(const int32_t brickId) const
{
    std::vector<ModelDescriptorPtr> modelDescriptors;
    uint32_t nbModels = 0;

    auto& connector = DBConnector::getInstance();
    auto buffer = connector.getBrick(brickId, CACHE_VERSION_1, nbModels);

    for (size_t i = 0; i < nbModels; ++i)
    {
        auto modelDescriptor = _importModel(buffer, brickId);
        if (modelDescriptor)
            modelDescriptors.push_back(modelDescriptor);
    }

    return modelDescriptors;
}

void CacheLoader::exportBrickToDB(const int32_t brickId, const Boxd& bounds) const
{
    std::stringstream buffer;
    uint32_t nbModels = 0;
    const auto& modelDescriptors = _scene.getModelDescriptors();
    for (const auto& modelDescriptor : modelDescriptors)
        nbModels += (_exportModel(modelDescriptor, buffer, bounds) ? 1 : 0);

    if (nbModels > 0)
    {
        PLUGIN_INFO(3, "Saving brick " << brickId << " ( " << nbModels << " models) to database");
        auto& connector = DBConnector::getInstance();
        connector.insertBrick(brickId, CACHE_VERSION_1, nbModels, buffer);
    }
}

void CacheLoader::exportToXYZ(const std::string& filename, const XYZFileFormat fileFormat) const
{
    PLUGIN_INFO(3, "Saving scene to XYZ file: " << filename);
    std::ios_base::openmode flags = std::ios::out;
    if (fileFormat == XYZFileFormat::xyz_binary || fileFormat == XYZFileFormat::xyzr_binary)
        flags |= std::ios::binary;

    std::ofstream file(filename, flags);
    if (!file.good())
        PLUGIN_THROW("Could not create XYZ file " + filename);

    if (fileFormat == XYZFileFormat::xyz_ascii || fileFormat == XYZFileFormat::xyzr_ascii ||
        fileFormat == XYZFileFormat::xyzrv_ascii || fileFormat == XYZFileFormat::xyzr_rgba_ascii)
    {
        // ASCII file header
        file << "x" << ASCII_FILE_SEPARATOR << "y" << ASCII_FILE_SEPARATOR << "z";
        if (fileFormat == XYZFileFormat::xyzr_ascii || fileFormat == XYZFileFormat::xyzrv_ascii ||
            fileFormat == XYZFileFormat::xyzr_rgba_ascii)
        {
            file << ASCII_FILE_SEPARATOR << "radius";
        }
        if (fileFormat == XYZFileFormat::xyzrv_ascii)
            file << ASCII_FILE_SEPARATOR << "value";
        if (fileFormat == XYZFileFormat::xyzr_rgba_ascii)
            file << ASCII_FILE_SEPARATOR << "r" << ASCII_FILE_SEPARATOR << "g" << ASCII_FILE_SEPARATOR << "b"
                 << ASCII_FILE_SEPARATOR << "a";
        file << std::endl;
    }

    const auto clipPlanes = getClippingPlanes(_scene);

    const auto& modelDescriptors = _scene.getModelDescriptors();
    for (const auto modelDescriptor : modelDescriptors)
    {
        const auto& instances = modelDescriptor->getInstances();
        for (const auto& instance : instances)
        {
            const auto& tf = instance.getTransformation();
            const auto& model = modelDescriptor->getModel();
            const auto& spheresMap = model.getSpheres();
            for (const auto& spheres : spheresMap)
            {
                if (spheres.first == BOUNDINGBOX_MATERIAL_ID || spheres.first == SECONDARY_MODEL_MATERIAL_ID)
                    continue;
                const auto material = model.getMaterial(spheres.first);

                for (const auto& sphere : spheres.second)
                {
                    const Vector3d center =
                        tf.getTranslation() + tf.getRotation() * (Vector3d(sphere.center) - tf.getRotationCenter());

                    const Vector3d c = center;
                    if (isClipped(c, clipPlanes))
                        continue;

                    switch (fileFormat)
                    {
                    case XYZFileFormat::xyz_binary:
                    case XYZFileFormat::xyzr_binary:
                    case XYZFileFormat::xyzrv_binary:
                        file.write((char*)&c.x, sizeof(double));
                        file.write((char*)&c.y, sizeof(double));
                        file.write((char*)&c.z, sizeof(double));
                        if (fileFormat == XYZFileFormat::xyzr_binary || fileFormat == XYZFileFormat::xyzrv_binary)
                        {
                            file.write((char*)&sphere.radius, sizeof(double));
                            if (fileFormat == XYZFileFormat::xyzrv_binary)
                                file.write((char*)&sphere.radius, sizeof(double));
                        }
                        break;
                    case XYZFileFormat::xyz_ascii:
                    case XYZFileFormat::xyzr_ascii:
                    case XYZFileFormat::xyzrv_ascii:
                    case XYZFileFormat::xyzr_rgba_ascii:
                        file << c.x << ASCII_FILE_SEPARATOR << c.y << ASCII_FILE_SEPARATOR << c.z;
                        if (fileFormat == XYZFileFormat::xyzr_ascii || fileFormat == XYZFileFormat::xyzrv_ascii ||
                            fileFormat == XYZFileFormat::xyzr_rgba_ascii)
                        {
                            file << ASCII_FILE_SEPARATOR << sphere.radius;
                            if (fileFormat == XYZFileFormat::xyzrv_ascii)
                                file << ASCII_FILE_SEPARATOR << sphere.radius;
                        }
                        if (material && fileFormat == XYZFileFormat::xyzr_rgba_ascii)
                        {
                            const auto& color = material->getDiffuseColor();
                            const auto opacity = material->getOpacity();

                            file << ASCII_FILE_SEPARATOR << static_cast<size_t>(255.f * color.x) << ASCII_FILE_SEPARATOR
                                 << static_cast<size_t>(255.f * color.y) << ASCII_FILE_SEPARATOR
                                 << static_cast<size_t>(255.f * color.z) << ASCII_FILE_SEPARATOR
                                 << static_cast<size_t>(255.f * opacity);
                        }
                        file << std::endl;
                        break;
                    case XYZFileFormat::unspecified:
                        PLUGIN_THROW("A file format must be specified");
                        break;
                    }
                }
            }
        }
    }
    file.close();
}

#ifdef USE_LASLIB
void CacheLoader::exportToLas(const std::string& filename, const bool exportColors) const
{
    const double DEFAULT_SCALE_FACTOR = 0.01;
    PLUGIN_INFO(3, "Saving scene to LAS file: " << filename);

    LASheader header;
    header.x_scale_factor = DEFAULT_SCALE_FACTOR;
    header.y_scale_factor = DEFAULT_SCALE_FACTOR;
    header.z_scale_factor = DEFAULT_SCALE_FACTOR;
    header.x_offset = 0.0;
    header.y_offset = 0.0;
    header.z_offset = 0.0;
    header.point_data_format = 3;         // Format LAS point format 3
    header.point_data_record_length = 34; // Format 3 Point size

    Boxd bounds;

    LASwriteOpener laswriteopener;
    laswriteopener.set_file_name(filename.c_str());

    LASwriter* writer = laswriteopener.open(&header);
    if (!writer)
        PLUGIN_THROW("Failed to create " + filename);

    LASpoint point;
    point.init(&header, header.point_data_format, header.point_data_record_length);

    const auto clipPlanes = getClippingPlanes(_scene);

    const auto& modelDescriptors = _scene.getModelDescriptors();
    for (const auto modelDescriptor : modelDescriptors)
    {
        const auto& instances = modelDescriptor->getInstances();
        for (const auto& instance : instances)
        {
            const auto& tf = instance.getTransformation();
            const auto& model = modelDescriptor->getModel();
            const auto& spheresMap = model.getSpheres();
            for (const auto& spheres : spheresMap)
            {
                if (spheres.first == BOUNDINGBOX_MATERIAL_ID || spheres.first == SECONDARY_MODEL_MATERIAL_ID)
                    continue;
                const auto material = model.getMaterial(spheres.first);

                for (const auto& sphere : spheres.second)
                {
                    const Vector3d center =
                        tf.getTranslation() + tf.getRotation() * (Vector3d(sphere.center) - tf.getRotationCenter());

                    if (isClipped(center, clipPlanes))
                        continue;

                    point.set_x(center.x);
                    point.set_y(center.y);
                    point.set_z(center.z);
                    point.set_intensity(sphere.radius / DEFAULT_SCALE_FACTOR);

                    const double radius = static_cast<double>(sphere.radius);
                    bounds.merge(center - radius);
                    bounds.merge(center + radius);

                    if (exportColors)
                    {
                        const auto& color = material->getDiffuseColor();
                        point.set_R(static_cast<size_t>(255.f * color.x));
                        point.set_G(static_cast<size_t>(255.f * color.y));
                        point.set_B(static_cast<size_t>(255.f * color.z));
                    }
                    writer->write_point(&point);
                    writer->update_inventory(&point);
                }
            }
        }
    }
    header.min_x = bounds.getMin().x;
    header.min_y = bounds.getMin().y;
    header.min_z = bounds.getMin().z;
    header.max_x = bounds.getMax().x;
    header.max_y = bounds.getMax().y;
    header.max_z = bounds.getMax().z;
    writer->update_header(&header);
    writer->close();
    delete writer;
}
#endif

PropertyMap CacheLoader::getProperties() const
{
    return _defaults;
}

PropertyMap CacheLoader::getCLIProperties()
{
    PropertyMap pm("BioExplorerLoader");
    return pm;
}
} // namespace io
} // namespace bioexplorer
