/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <plugin/common/Logs.h>

#include "CacheLoader.h"

#include <plugin/api/Params.h>
#include <plugin/bioexplorer/Assembly.h>
#include <plugin/bioexplorer/Protein.h>
#include <plugin/common/CommonTypes.h>
#include <plugin/common/Utils.h>

#include <brayns/common/scene/ClipPlane.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>

#include <fstream>

namespace bioexplorer
{
const std::string LOADER_NAME = "Bio Explorer cache loader";
const std::string SUPPORTED_EXTENTION_BIOEXPLORER = "bioexplorer";

const size_t CACHE_VERSION_1 = 1;

CacheLoader::CacheLoader(Scene& scene, PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
    PLUGIN_INFO << "Registering " << LOADER_NAME << std::endl;
}

std::string CacheLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> CacheLoader::getSupportedExtensions() const
{
    return {SUPPORTED_EXTENTION_BIOEXPLORER};
}

bool CacheLoader::isSupported(const std::string& /*filename*/,
                              const std::string& extension) const
{
    const std::set<std::string> types = {SUPPORTED_EXTENTION_BIOEXPLORER};
    return types.find(extension) != types.end();
}

ModelDescriptorPtr CacheLoader::importFromBlob(
    Blob&& /*blob*/, const LoaderProgress& /*callback*/,
    const PropertyMap& /*properties*/) const
{
    throw std::runtime_error(
        "Loading molecular systems from blob is not supported");
}

void CacheLoader::_importModel(std::ifstream& file) const
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
    const auto name = _readString(file);

    // Path
    const auto path = _readString(file);

    // Metadata
    size_t nbElements;
    ModelMetadata metadata;
    file.read((char*)&nbElements, sizeof(size_t));
    for (size_t i = 0; i < nbElements; ++i)
        metadata[_readString(file)] = _readString(file);

    // Instances
    std::vector<Transformation> transformations;
    file.read((char*)&nbElements, sizeof(size_t));
    for (size_t i = 0; i < nbElements; ++i)
    {
        Transformation tf;
        Vector3d t;
        Vector3d rc;
        Vector3d s;
        Quaterniond q;

        file.read((char*)&t, sizeof(Vector3d));
        tf.setTranslation(t);
        file.read((char*)&rc, sizeof(Vector3d));
        tf.setRotationCenter(rc);
        file.read((char*)&q, sizeof(Quaterniond));
        tf.setRotation(q);
        file.read((char*)&s, sizeof(Vector3d));
        tf.setScale(s);
        transformations.push_back(tf);
    }

    size_t nbMaterials;
    file.read((char*)&nbMaterials, sizeof(size_t));

    // Materials
    size_t materialId;
    for (size_t i = 0; i < nbMaterials; ++i)
    {
        file.read((char*)&materialId, sizeof(size_t));

        auto name = _readString(file);
        auto material = model->createMaterial(materialId, name);

        Vector3f value3f;
        file.read((char*)&value3f, sizeof(Vector3f));
        material->setDiffuseColor(value3f);
        file.read((char*)&value3f, sizeof(Vector3f));
        material->setSpecularColor(value3f);
        float value;
        file.read((char*)&value, sizeof(float));
        material->setSpecularExponent(value);
        file.read((char*)&value, sizeof(float));
        material->setReflectionIndex(value);
        file.read((char*)&value, sizeof(float));
        material->setOpacity(value);
        file.read((char*)&value, sizeof(float));
        material->setRefractionIndex(value);
        file.read((char*)&value, sizeof(float));
        material->setEmission(value);
        file.read((char*)&value, sizeof(float));
        material->setGlossiness(value);

        double userParameter;
        file.read((char*)&userParameter, sizeof(double));
        material->updateProperty(MATERIAL_PROPERTY_USER_PARAMETER,
                                 userParameter);

        int32_t shadingMode;
        file.read((char*)&shadingMode, sizeof(int32_t));
        material->updateProperty(MATERIAL_PROPERTY_SHADING_MODE, shadingMode);
    }

    uint64_t bufferSize{0};

    // Spheres
    file.read((char*)&nbSpheres, sizeof(size_t));
    for (size_t i = 0; i < nbSpheres; ++i)
    {
        file.read((char*)&materialId, sizeof(size_t));
        file.read((char*)&nbElements, sizeof(size_t));
        auto& spheres = model->getSpheres()[materialId];
        spheres.resize(nbElements);
        bufferSize = nbElements * sizeof(Sphere);
        file.read((char*)spheres.data(), bufferSize);
    }

    // Cylinders
    file.read((char*)&nbCylinders, sizeof(size_t));
    for (size_t i = 0; i < nbCylinders; ++i)
    {
        file.read((char*)&materialId, sizeof(size_t));
        file.read((char*)&nbElements, sizeof(size_t));
        auto& cylinders = model->getCylinders()[materialId];
        cylinders.resize(nbElements);
        bufferSize = nbElements * sizeof(Cylinder);
        file.read((char*)cylinders.data(), bufferSize);
    }

    // Cones
    file.read((char*)&nbCones, sizeof(size_t));
    for (size_t i = 0; i < nbCones; ++i)
    {
        file.read((char*)&materialId, sizeof(size_t));
        file.read((char*)&nbElements, sizeof(size_t));
        auto& cones = model->getCones()[materialId];
        cones.resize(nbElements);
        bufferSize = nbElements * sizeof(Cone);
        file.read((char*)cones.data(), bufferSize);
    }

    // Meshes
    file.read((char*)&nbMeshes, sizeof(size_t));
    for (size_t i = 0; i < nbMeshes; ++i)
    {
        file.read((char*)&materialId, sizeof(size_t));
        auto& meshes = model->getTriangleMeshes()[materialId];
        // Vertices
        file.read((char*)&nbVertices, sizeof(size_t));
        if (nbVertices != 0)
        {
            bufferSize = nbVertices * sizeof(Vector3f);
            meshes.vertices.resize(nbVertices);
            file.read((char*)meshes.vertices.data(), bufferSize);
        }

        // Indices
        file.read((char*)&nbIndices, sizeof(size_t));
        if (nbIndices != 0)
        {
            bufferSize = nbIndices * sizeof(Vector3ui);
            meshes.indices.resize(nbIndices);
            file.read((char*)meshes.indices.data(), bufferSize);
        }

        // Normals
        file.read((char*)&nbNormals, sizeof(size_t));
        if (nbNormals != 0)
        {
            bufferSize = nbNormals * sizeof(Vector3f);
            meshes.normals.resize(nbNormals);
            file.read((char*)meshes.normals.data(), bufferSize);
        }

        // Texture coordinates
        file.read((char*)&nbTexCoords, sizeof(size_t));
        if (nbTexCoords != 0)
        {
            bufferSize = nbTexCoords * sizeof(Vector2f);
            meshes.textureCoordinates.resize(nbTexCoords);
            file.read((char*)meshes.textureCoordinates.data(), bufferSize);
        }
    }

    // Streamlines
    size_t nbStreamlines;
    auto& streamlines = model->getStreamlines();
    file.read((char*)&nbStreamlines, sizeof(size_t));
    for (size_t i = 0; i < nbStreamlines; ++i)
    {
        StreamlinesData streamlineData;
        // Id
        size_t id;
        file.read((char*)&id, sizeof(size_t));

        // Vertex
        file.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(Vector4f);
        streamlineData.vertex.resize(nbElements);
        file.read((char*)streamlineData.vertex.data(), bufferSize);

        // Vertex Color
        file.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(Vector4f);
        streamlineData.vertexColor.resize(nbElements);
        file.read((char*)streamlineData.vertexColor.data(), bufferSize);

        // Indices
        file.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(int32_t);
        streamlineData.indices.resize(nbElements);
        file.read((char*)streamlineData.indices.data(), bufferSize);

        streamlines[id] = streamlineData;
    }

    // SDF geometry
    auto& sdfData = model->getSDFGeometryData();
    file.read((char*)&nbElements, sizeof(size_t));

    if (nbElements > 0)
    {
        // Geometries
        sdfData.geometries.resize(nbElements);
        bufferSize = nbElements * sizeof(SDFGeometry);
        file.read((char*)sdfData.geometries.data(), bufferSize);

        // SDF Indices
        file.read((char*)&nbElements, sizeof(size_t));
        for (size_t i = 0; i < nbElements; ++i)
        {
            file.read((char*)&materialId, sizeof(size_t));
            size_t size;
            file.read((char*)&size, sizeof(size_t));
            bufferSize = size * sizeof(uint64_t);
            sdfData.geometryIndices[materialId].resize(size);
            file.read((char*)sdfData.geometryIndices[materialId].data(),
                      bufferSize);
        }

        // Neighbours
        file.read((char*)&nbElements, sizeof(size_t));
        sdfData.neighbours.resize(nbElements);

        for (size_t i = 0; i < nbElements; ++i)
        {
            size_t size;
            file.read((char*)&size, sizeof(size_t));
            bufferSize = size * sizeof(uint64_t);
            sdfData.neighbours[i].resize(size);
            file.read((char*)sdfData.neighbours[i].data(), bufferSize);
        }

        // Neighbours flat
        file.read((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(uint64_t);
        sdfData.neighboursFlat.resize(nbElements);
        file.read((char*)sdfData.neighboursFlat.data(), bufferSize);
    }

    auto modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), name, path,
                                          metadata);

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

    _scene.addModel(modelDescriptor);
}

ModelDescriptorPtr CacheLoader::importFromFile(
    const std::string& filename, const LoaderProgress& callback,
    const PropertyMap& properties) const
{
    auto model = _scene.createModel();

    PropertyMap props = _defaults;
    props.merge(properties);

    callback.updateProgress("Loading BioExplorer scene...", 0);
    PLUGIN_INFO << "Loading models from cache file: " << filename << std::endl;
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.good())
    {
        const std::string msg = "Could not open cache file " + filename;
        PLUGIN_THROW(std::runtime_error(msg));
    }

    // File version
    size_t version;
    file.read((char*)&version, sizeof(size_t));
    PLUGIN_INFO << "Version: " << version << std::endl;

    // Models
    size_t nbModels;
    file.read((char*)&nbModels, sizeof(size_t));
    for (size_t i = 0; i < nbModels; ++i)
    {
        _importModel(file);
        callback.updateProgress("Loading models", float(i) / float(nbModels));
    }

    file.close();
    return _scene.getModelDescriptors()[0];
}

std::string CacheLoader::_readString(std::ifstream& f) const
{
    size_t size;
    f.read((char*)&size, sizeof(size_t));
    char* str = new char[size + 1];
    f.read(str, size);
    str[size] = 0;
    std::string s{str};
    delete[] str;
    return s;
}

void CacheLoader::_exportModel(const ModelDescriptorPtr modelDescriptor,
                               std::ofstream& file) const
{
    uint64_t bufferSize{0};
    const auto& model = modelDescriptor->getModel();

    // Name
    const auto& name = modelDescriptor->getName();
    size_t size = name.length();
    file.write((char*)&size, sizeof(size_t));
    file.write((char*)name.c_str(), size);

    // Path
    const auto& path = modelDescriptor->getPath();
    size = path.length();
    file.write((char*)&size, sizeof(size_t));
    file.write((char*)path.c_str(), size);

    // Metadata
    auto metadata = modelDescriptor->getMetadata();
    size_t nbElements = metadata.size();
    file.write((char*)&nbElements, sizeof(size_t));
    for (const auto& data : metadata)
    {
        size = data.first.length();
        file.write((char*)&size, sizeof(size_t));
        file.write((char*)data.first.c_str(), size);
        size = data.second.length();
        file.write((char*)&size, sizeof(size_t));
        file.write((char*)data.second.c_str(), size);
    }

    // Instances
    const auto& instances = modelDescriptor->getInstances();
    nbElements = instances.size();
    file.write((char*)&nbElements, sizeof(size_t));
    bool first{true};
    for (const auto& instance : instances)
    {
        brayns::Vector3d t;
        brayns::Vector3d rc;
        brayns::Vector3d s;
        brayns::Quaterniond q;
        if (first)
        {
            const auto& tf = modelDescriptor->getTransformation();
            t = tf.getTranslation();
            rc = tf.getRotationCenter();
            q = tf.getRotation();
            s = tf.getScale();
            first = false;
        }
        else
        {
            const auto& tf = instance.getTransformation();
            t = tf.getTranslation();
            rc = tf.getRotationCenter();
            q = tf.getRotation();
            s = tf.getScale();
        }
        file.write((char*)&t, sizeof(brayns::Vector3d));
        file.write((char*)&rc, sizeof(brayns::Vector3d));
        file.write((char*)&q, sizeof(brayns::Quaterniond));
        file.write((char*)&s, sizeof(brayns::Vector3d));
    }

    // Materials
    const auto& materials = model.getMaterials();
    nbElements = materials.size();
    file.write((char*)&nbElements, sizeof(size_t));
    for (const auto& material : materials)
    {
        file.write((char*)&material.first, sizeof(size_t));

        auto name = material.second->getName();
        size_t size = name.length();
        file.write((char*)&size, sizeof(size_t));
        file.write((char*)name.c_str(), size);

        brayns::Vector3f value3f;
        value3f = material.second->getDiffuseColor();
        file.write((char*)&value3f, sizeof(brayns::Vector3f));
        value3f = material.second->getSpecularColor();
        file.write((char*)&value3f, sizeof(brayns::Vector3f));
        float value = material.second->getSpecularExponent();
        file.write((char*)&value, sizeof(float));
        value = material.second->getReflectionIndex();
        file.write((char*)&value, sizeof(float));
        value = material.second->getOpacity();
        file.write((char*)&value, sizeof(float));
        value = material.second->getRefractionIndex();
        file.write((char*)&value, sizeof(float));
        value = material.second->getEmission();
        file.write((char*)&value, sizeof(float));
        value = material.second->getGlossiness();
        file.write((char*)&value, sizeof(float));
        double v = 1.0;
        try
        {
            v = material.second->getProperty<double>(
                MATERIAL_PROPERTY_USER_PARAMETER);
        }
        catch (const std::runtime_error&)
        {
        }
        file.write((char*)&v, sizeof(double));

        int32_t shadingMode = MaterialShadingMode::undefined_shading_mode;
        try
        {
            shadingMode = material.second->getProperty<int32_t>(
                MATERIAL_PROPERTY_SHADING_MODE);
        }
        catch (const std::runtime_error&)
        {
        }
        file.write((char*)&shadingMode, sizeof(int32_t));
    }

    // Spheres
    const auto& spheresMap = model.getSpheres();
    nbElements = spheresMap.size();
    file.write((char*)&nbElements, sizeof(size_t));
    for (const auto& spheres : spheresMap)
    {
        const auto materialId = spheres.first;
        file.write((char*)&materialId, sizeof(size_t));

        const auto& data = spheres.second;
        nbElements = data.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Sphere);
        file.write((char*)data.data(), bufferSize);
    }

    // Cylinders
    const auto& cylindersMap = model.getCylinders();
    nbElements = cylindersMap.size();
    file.write((char*)&nbElements, sizeof(size_t));
    for (const auto& cylinders : cylindersMap)
    {
        const auto materialId = cylinders.first;
        file.write((char*)&materialId, sizeof(size_t));

        const auto& data = cylinders.second;
        nbElements = data.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Cylinder);
        file.write((char*)data.data(), bufferSize);
    }

    // Cones
    const auto& conesMap = model.getCones();
    nbElements = conesMap.size();
    file.write((char*)&nbElements, sizeof(size_t));
    for (const auto& cones : conesMap)
    {
        const auto materialId = cones.first;
        file.write((char*)&materialId, sizeof(size_t));

        const auto& data = cones.second;
        nbElements = data.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Cone);
        file.write((char*)data.data(), bufferSize);
    }

    // Meshes
    const auto& trianglesMap = model.getTriangleMeshes();
    nbElements = trianglesMap.size();
    file.write((char*)&nbElements, sizeof(size_t));
    for (const auto& meshes : trianglesMap)
    {
        const auto materialId = meshes.first;
        file.write((char*)&materialId, sizeof(size_t));

        const auto& data = meshes.second;

        // Vertices
        nbElements = data.vertices.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Vector3f);
        file.write((char*)data.vertices.data(), bufferSize);

        // Indices
        nbElements = data.indices.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Vector3ui);
        file.write((char*)data.indices.data(), bufferSize);

        // Normals
        nbElements = data.normals.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Vector3f);
        file.write((char*)data.normals.data(), bufferSize);

        // Texture coordinates
        nbElements = data.textureCoordinates.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Vector2f);
        file.write((char*)data.textureCoordinates.data(), bufferSize);
    }

    // Streamlines
    const auto& streamlines = model.getStreamlines();
    nbElements = streamlines.size();
    file.write((char*)&nbElements, sizeof(size_t));
    for (const auto& streamline : streamlines)
    {
        const auto& streamlineData = streamline.second;
        // Id
        size_t id = streamline.first;
        file.write((char*)&id, sizeof(size_t));

        // Vertex
        nbElements = streamlineData.vertex.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Vector4f);
        file.write((char*)streamlineData.vertex.data(), bufferSize);

        // Vertex Color
        nbElements = streamlineData.vertexColor.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(brayns::Vector4f);
        file.write((char*)streamlineData.vertexColor.data(), bufferSize);

        // Indices
        nbElements = streamlineData.indices.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(int32_t);
        file.write((char*)streamlineData.indices.data(), bufferSize);
    }

    // SDF geometry
    const auto& sdfData = model.getSDFGeometryData();
    nbElements = sdfData.geometries.size();
    file.write((char*)&nbElements, sizeof(size_t));

    if (nbElements > 0)
    {
        // Geometries
        bufferSize = nbElements * sizeof(brayns::SDFGeometry);
        file.write((char*)sdfData.geometries.data(), bufferSize);

        // SDF indices
        nbElements = sdfData.geometryIndices.size();
        file.write((char*)&nbElements, sizeof(size_t));
        for (const auto& geometryIndex : sdfData.geometryIndices)
        {
            size_t materialId = geometryIndex.first;
            file.write((char*)&materialId, sizeof(size_t));
            nbElements = geometryIndex.second.size();
            file.write((char*)&nbElements, sizeof(size_t));
            bufferSize = nbElements * sizeof(uint64_t);
            file.write((char*)geometryIndex.second.data(), bufferSize);
        }

        // Neighbours
        nbElements = sdfData.neighbours.size();
        file.write((char*)&nbElements, sizeof(size_t));
        for (const auto& neighbour : sdfData.neighbours)
        {
            nbElements = neighbour.size();
            file.write((char*)&nbElements, sizeof(size_t));
            bufferSize = nbElements * sizeof(size_t);
            file.write((char*)neighbour.data(), bufferSize);
        }

        // Neighbours flat
        nbElements = sdfData.neighboursFlat.size();
        file.write((char*)&nbElements, sizeof(size_t));
        bufferSize = nbElements * sizeof(uint64_t);
        file.write((char*)sdfData.neighboursFlat.data(), bufferSize);
    }
}

void CacheLoader::exportToCache(const std::string& filename) const
{
    PLUGIN_INFO << "Saving scene to BioExplorer file: " << filename
                << std::endl;
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.good())
    {
        const std::string msg = "Could not create BioExplorer file " + filename;
        PLUGIN_THROW(std::runtime_error(msg));
    }

    const size_t version = CACHE_VERSION_1;
    file.write((char*)&version, sizeof(size_t));

    const auto& modelDescriptors = _scene.getModelDescriptors();
    const size_t nbModels = modelDescriptors.size();
    file.write((char*)&nbModels, sizeof(size_t));

    for (const auto& modelDescriptor : modelDescriptors)
        _exportModel(modelDescriptor, file);

    file.close();
}

void CacheLoader::exportToXYZ(const std::string& filename,
                              const XYZFileFormat fileFormat) const
{
    PLUGIN_INFO << "Saving scene to XYZ file: " << filename << std::endl;
    std::ios_base::openmode flags = std::ios::out;
    if (fileFormat == XYZFileFormat::xyz_binary ||
        fileFormat == XYZFileFormat::xyzr_binary)
        flags |= std::ios::binary;

    std::ofstream file(filename, flags);
    if (!file.good())
    {
        const std::string msg = "Could not create XYZ file " + filename;
        PLUGIN_THROW(std::runtime_error(msg));
    }

    const auto& clippingPlanes = _scene.getClipPlanes();
    Vector4fs clipPlanes;
    for (const auto cp : clippingPlanes)
    {
        const auto& p = cp->getPlane();
        Vector4f plane{p[0], p[1], p[2], p[3]};
        clipPlanes.push_back(plane);
    }

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
                for (const auto& sphere : spheres.second)
                {
                    const Vector3d center =
                        tf.getTranslation() +
                        tf.getRotation() *
                            (Vector3d(sphere.center) - tf.getRotationCenter());

                    const Vector3f c = center;
                    if (isClipped(c, clipPlanes))
                        continue;

                    switch (fileFormat)
                    {
                    case XYZFileFormat::xyz_binary:
                    case XYZFileFormat::xyzr_binary:
                        file.write((char*)&c.x, sizeof(float));
                        file.write((char*)&c.y, sizeof(float));
                        file.write((char*)&c.z, sizeof(float));
                        if (fileFormat == XYZFileFormat::xyzr_binary ||
                            fileFormat == XYZFileFormat::xyzrv_binary)
                        {
                            file.write((char*)&sphere.radius, sizeof(float));
                            if (fileFormat == XYZFileFormat::xyzrv_binary)
                                file.write((char*)&sphere.radius,
                                           sizeof(float));
                        }
                        break;
                    case XYZFileFormat::xyz_ascii:
                    case XYZFileFormat::xyzr_ascii:
                        file << c.x << " " << c.y << " " << c.z;
                        if (fileFormat == XYZFileFormat::xyzr_ascii ||
                            fileFormat == XYZFileFormat::xyzrv_ascii)
                        {
                            file << " " << sphere.radius;
                            if (fileFormat == XYZFileFormat::xyzrv_ascii)
                                file << " " << sphere.radius;
                        }
                        file << std::endl;
                        break;
                    }
                }
            }
        }
    }
    file.close();
}

PropertyMap CacheLoader::getProperties() const
{
    return _defaults;
}

PropertyMap CacheLoader::getCLIProperties()
{
    PropertyMap pm("BioExplorerLoader");
    return pm;
}
} // namespace bioexplorer
