
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_CacheLoader.cpp:

Program Listing for File CacheLoader.cpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_CacheLoader.cpp>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/io/CacheLoader.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /*
    * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
    * scientific data from visualization
    *
    * Copyright 2020-2022 Blue BrainProject / EPFL
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
   #include <plugin/biology/Assembly.h>
   #include <plugin/biology/Protein.h>
   #include <plugin/common/CommonTypes.h>
   #include <plugin/common/GeneralSettings.h>
   #include <plugin/common/Utils.h>
   
   #include <brayns/common/scene/ClipPlane.h>
   
   #include <brayns/engineapi/Material.h>
   #include <brayns/engineapi/Model.h>
   #include <brayns/engineapi/Scene.h>
   #include <brayns/parameters/ParametersManager.h>
   
   #include <fstream>
   
   namespace
   {
   bool inBounds(const Vector3f& point, const Boxd& bounds)
   {
       const auto mi = bounds.getMin();
       const auto ma = bounds.getMax();
       return point.x >= mi.x && point.x < ma.x && point.y >= mi.y &&
              point.y < ma.y && point.z >= mi.z && point.z < ma.z;
   }
   } // namespace
   
   namespace bioexplorer
   {
   namespace io
   {
   using namespace common;
   
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
       PLUGIN_THROW("Loading molecular systems from blob is not supported");
   }
   
   ModelDescriptorPtr CacheLoader::_importModel(std::stringstream& buffer,
                                                const int32_t brickId) const
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
   
           Vector3f value3f;
           buffer.read((char*)&value3f, sizeof(Vector3f));
           material->setDiffuseColor(value3f);
           buffer.read((char*)&value3f, sizeof(Vector3f));
           material->setSpecularColor(value3f);
           float value;
           buffer.read((char*)&value, sizeof(float));
           material->setSpecularExponent(value);
           buffer.read((char*)&value, sizeof(float));
           material->setReflectionIndex(value);
           buffer.read((char*)&value, sizeof(float));
           material->setOpacity(value);
           buffer.read((char*)&value, sizeof(float));
           material->setRefractionIndex(value);
           buffer.read((char*)&value, sizeof(float));
           material->setEmission(value);
           buffer.read((char*)&value, sizeof(float));
           material->setGlossiness(value);
   
           brayns::PropertyMap props;
           double userParameter;
           buffer.read((char*)&userParameter, sizeof(double));
           props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, userParameter});
   
           int32_t shadingMode;
           buffer.read((char*)&shadingMode, sizeof(int32_t));
           // props.setProperty({MATERIAL_PROPERTY_SHADING_MODE, shadingMode});
           props.setProperty({MATERIAL_PROPERTY_SHADING_MODE, 1});
   
           int32_t chameleonMode;
           buffer.read((char*)&chameleonMode, sizeof(int32_t));
           props.setProperty({MATERIAL_PROPERTY_CHAMELEON_MODE, chameleonMode});
   
           int32_t nodeId;
           buffer.read((char*)&nodeId, sizeof(int32_t));
           props.setProperty({MATERIAL_PROPERTY_NODE_ID, nodeId});
           material->updateProperties(props);
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
               bufferSize = nbVertices * sizeof(Vector3f);
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
               bufferSize = nbNormals * sizeof(Vector3f);
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
               buffer.read((char*)sdfData.geometryIndices[materialId].data(),
                           bufferSize);
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
   
           const auto visible =
               GeneralSettings::getInstance()->getModelVisibilityOnCreation();
           modelDescriptor->setVisible(visible);
           return modelDescriptor;
       }
       return nullptr;
   }
   
   std::vector<ModelDescriptorPtr> CacheLoader::importModelsFromFile(
       const std::string& filename, const int32_t brickId,
       const LoaderProgress& callback, const PropertyMap& properties) const
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
   
           callback.updateProgress("Loading models", float(i) / float(nbModels));
       }
   
       return modelDescriptors;
   }
   
   ModelDescriptorPtr CacheLoader::importFromFile(
       const std::string& filename, const LoaderProgress& callback,
       const PropertyMap& properties) const
   {
       const auto modelDescriptors =
           importModelsFromFile(filename, UNDEFINED_BOX_ID, callback, properties);
       for (const auto modelDescriptor : modelDescriptors)
           _scene.addModel(modelDescriptor);
   
       return (!modelDescriptors.empty() ? modelDescriptors[0] : nullptr);
   }
   
   std::string CacheLoader::_readString(std::stringstream& buffer) const
   {
       size_t size;
       buffer.read((char*)&size, sizeof(size_t));
       char* str = new char[size + 1];
       buffer.read(str, size);
       str[size] = 0;
       std::string s{str};
       delete[] str;
       return s;
   }
   
   bool CacheLoader::_exportModel(const ModelDescriptorPtr modelDescriptor,
                                  std::stringstream& buffer,
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
           buffer.write((char*)&t, sizeof(brayns::Vector3d));
           const auto& rc = tf.getRotationCenter();
           buffer.write((char*)&rc, sizeof(brayns::Vector3d));
           const auto& q = tf.getRotation();
           buffer.write((char*)&q, sizeof(brayns::Quaterniond));
           const auto& s = tf.getScale();
           buffer.write((char*)&s, sizeof(brayns::Vector3d));
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
   
           brayns::Vector3f value3f;
           value3f = material.second->getDiffuseColor();
           buffer.write((char*)&value3f, sizeof(brayns::Vector3f));
           value3f = material.second->getSpecularColor();
           buffer.write((char*)&value3f, sizeof(brayns::Vector3f));
           float value = material.second->getSpecularExponent();
           buffer.write((char*)&value, sizeof(float));
           value = material.second->getReflectionIndex();
           buffer.write((char*)&value, sizeof(float));
           value = material.second->getOpacity();
           buffer.write((char*)&value, sizeof(float));
           value = material.second->getRefractionIndex();
           buffer.write((char*)&value, sizeof(float));
           value = material.second->getEmission();
           buffer.write((char*)&value, sizeof(float));
           value = material.second->getGlossiness();
           buffer.write((char*)&value, sizeof(float));
           double v = 1.0;
           try
           {
               v = material.second->getProperty<double>(
                   MATERIAL_PROPERTY_USER_PARAMETER);
           }
           catch (const std::runtime_error&)
           {
           }
           buffer.write((char*)&v, sizeof(double));
   
           int32_t shadingMode = MaterialShadingMode::undefined_shading_mode;
           try
           {
               shadingMode = material.second->getProperty<int32_t>(
                   MATERIAL_PROPERTY_SHADING_MODE);
           }
           catch (const std::runtime_error&)
           {
           }
           buffer.write((char*)&shadingMode, sizeof(int32_t));
   
           int32_t chameleonMode = MaterialChameleonMode::undefined_chameleon_mode;
           try
           {
               shadingMode = material.second->getProperty<int32_t>(
                   MATERIAL_PROPERTY_CHAMELEON_MODE);
           }
           catch (const std::runtime_error&)
           {
           }
           buffer.write((char*)&chameleonMode, sizeof(int32_t));
   
           int32_t nodeId = 0;
           try
           {
               shadingMode = material.second->getProperty<int32_t>(
                   MATERIAL_PROPERTY_NODE_ID);
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
           bufferSize = nbElements * sizeof(brayns::Sphere);
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
           bufferSize = nbElements * sizeof(brayns::Cylinder);
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
           bufferSize = nbElements * sizeof(brayns::Cone);
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
           bufferSize = nbElements * sizeof(brayns::Vector3f);
           buffer.write((char*)data.vertices.data(), bufferSize);
   
           // Indices
           nbElements = data.indices.size();
           buffer.write((char*)&nbElements, sizeof(size_t));
           bufferSize = nbElements * sizeof(brayns::Vector3ui);
           buffer.write((char*)data.indices.data(), bufferSize);
   
           // Normals
           nbElements = data.normals.size();
           buffer.write((char*)&nbElements, sizeof(size_t));
           bufferSize = nbElements * sizeof(brayns::Vector3f);
           buffer.write((char*)data.normals.data(), bufferSize);
   
           // Texture coordinates
           nbElements = data.textureCoordinates.size();
           buffer.write((char*)&nbElements, sizeof(size_t));
           bufferSize = nbElements * sizeof(brayns::Vector2f);
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
           bufferSize = nbElements * sizeof(brayns::Vector4f);
           buffer.write((char*)streamlineData.vertex.data(), bufferSize);
   
           // Vertex Color
           nbElements = streamlineData.vertexColor.size();
           buffer.write((char*)&nbElements, sizeof(size_t));
           bufferSize = nbElements * sizeof(brayns::Vector4f);
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
           bufferSize = nbElements * sizeof(brayns::SDFGeometry);
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
   
   void CacheLoader::exportToFile(const std::string& filename,
                                  const Boxd& bounds) const
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
   
   #ifdef USE_PQXX
   std::vector<ModelDescriptorPtr> CacheLoader::importBrickFromDB(
       DBConnector& connector, const int32_t brickId) const
   {
       std::vector<ModelDescriptorPtr> modelDescriptors;
       uint32_t nbModels = 0;
   
       auto buffer = connector.getBrick(brickId, CACHE_VERSION_1, nbModels);
   
       for (size_t i = 0; i < nbModels; ++i)
       {
           auto modelDescriptor = _importModel(buffer, brickId);
           if (modelDescriptor)
               modelDescriptors.push_back(modelDescriptor);
       }
   
       return modelDescriptors;
   }
   
   void CacheLoader::exportBrickToDB(DBConnector& connector, const int32_t brickId,
                                     const Boxd& bounds) const
   {
       std::stringstream buffer;
       uint32_t nbModels = 0;
       const auto& modelDescriptors = _scene.getModelDescriptors();
       for (const auto& modelDescriptor : modelDescriptors)
           nbModels += (_exportModel(modelDescriptor, buffer, bounds) ? 1 : 0);
   
       if (nbModels > 0)
       {
           PLUGIN_INFO("Saving brick " << brickId << " ( " << nbModels
                                       << " models) to database");
   
           connector.insertBrick(brickId, CACHE_VERSION_1, nbModels, buffer);
       }
   }
   #endif
   
   void CacheLoader::exportToXYZ(const std::string& filename,
                                 const XYZFileFormat fileFormat) const
   {
       PLUGIN_INFO("Saving scene to XYZ file: " << filename);
       std::ios_base::openmode flags = std::ios::out;
       if (fileFormat == XYZFileFormat::xyz_binary ||
           fileFormat == XYZFileFormat::xyzr_binary)
           flags |= std::ios::binary;
   
       std::ofstream file(filename, flags);
       if (!file.good())
           PLUGIN_THROW("Could not create XYZ file " + filename);
   
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
   } // namespace io
   } // namespace bioexplorer
