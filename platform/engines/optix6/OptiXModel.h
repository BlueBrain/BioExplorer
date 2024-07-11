/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "OptiXTypes.h"

#include <platform/core/engineapi/Model.h>

#include <optixu/optixpp_namespace.h>

#include <map>

namespace core
{
namespace engine
{
namespace optix
{
class OptiXModel : public Model
{
public:
    OptiXModel(AnimationParameters& animationParameters, VolumeParameters& volumeParameters,
               GeometryParameters& geometryParameters, FieldParameters& fieldParameters);

    ~OptiXModel();

    /** @copydoc Model::commit */
    void commitGeometry() final;

    /** @copydoc Model::buildBoundingBox */
    void buildBoundingBox() final;

    /** @copydoc Model::createMaterialImpl */
    virtual MaterialPtr createMaterialImpl(const PropertyMap& properties = {}) final;

    /** @copydoc Model::createSharedDataVolume */
    virtual SharedDataVolumePtr createSharedDataVolume(const Vector3ui& dimensions, const Vector3f& spacing,
                                                       const DataType type) final;

    /** @copydoc Model::createBrickedVolume */
    virtual BrickedVolumePtr createBrickedVolume(const Vector3ui& dimensions, const Vector3f& spacing,
                                                 const DataType type) final;

    /** @copydoc Model::createField */
    virtual FieldPtr createField(const Vector3ui& dimensions, const Vector3f& spacing, const Vector3f& offset,
                                 const uint32_ts& indices, const floats& values, const OctreeDataType dataType) final;

    ::optix::GeometryGroup getGeometryGroup() const { return _geometryGroup; }
    ::optix::GeometryGroup getBoundingBoxGroup() const { return _boundingBoxGroup; }

protected:
    size_t _createSampler(const size_t materialId, const ::optix::Buffer& buffer, const TextureType textureType,
                          const RTtextureindexmode textureIndexMode, const Vector2f& valueRange = Vector2f());

    void _commitTransferFunctionImpl(const Vector3fs& colors, const floats& opacities, const Vector2d valueRange) final;
    void _commitSimulationDataImpl(const float* frameData, const size_t frameSize) final;
    void _commitVolumesBuffers(const size_t materialId);
    void _commitFieldsBuffers(const size_t materialId);

private:
    void _resetUserDataBuffer();
    uint64_t _commitSpheres(const size_t materialId);
    uint64_t _commitCylinders(const size_t materialId);
    uint64_t _commitCones(const size_t materialId);
    uint64_t _commitSDFGeometries();
    uint64_t _commitMeshes(const size_t materialId);
    uint64_t _commitVolumes(const size_t materialId);
    uint64_t _commitFields(const size_t materialId);
    uint64_t _commitStreamlines(const size_t materialId);
    uint64_t _commitMaterials();
    uint64_t _commitSimulationData();
    uint64_t _commitTransferFunction();

    ::optix::GeometryGroup _geometryGroup{nullptr};
    ::optix::GeometryGroup _boundingBoxGroup{nullptr};

    // Spheres
    std::map<size_t, ::optix::Buffer> _spheresBuffers;
    std::map<size_t, ::optix::Geometry> _optixSpheres;

    // Cylinders
    std::map<size_t, ::optix::Buffer> _cylindersBuffers;
    std::map<size_t, ::optix::Geometry> _optixCylinders;

    // Cones
    std::map<size_t, ::optix::Buffer> _conesBuffers;
    std::map<size_t, ::optix::Geometry> _optixCones;

    // SDF geometries
    struct OptiXSDFGeometryBuffers
    {
        ::optix::Buffer indices_buffer{nullptr};
        ::optix::Buffer geometries_buffer{nullptr};
        ::optix::Buffer neighbours_buffer{nullptr};
    };
    std::map<size_t, OptiXSDFGeometryBuffers> _sdfGeometriesBuffers;
    std::map<size_t, ::optix::Geometry> _optixSdfGeometries;

    // Volumes
    std::map<size_t, ::optix::Buffer> _volumesBuffers;
    std::map<size_t, ::optix::Geometry> _optixVolumes;

    // Fields
    std::map<size_t, ::optix::Buffer> _fieldsBuffers;
    std::map<size_t, ::optix::Geometry> _optixFields;

    // Meshes
    struct OptiXTriangleMeshBuffers
    {
        ::optix::Buffer vertices_buffer{nullptr};
        ::optix::Buffer normal_buffer{nullptr};
        ::optix::Buffer texcoord_buffer{nullptr};
        ::optix::Buffer indices_buffer{nullptr};
    };

    std::map<size_t, OptiXTriangleMeshBuffers> _optixMeshBuffers;
    std::map<size_t, ::optix::Geometry> _optixMeshes;

    // Volume
    ::optix::Buffer _volumeBuffer{nullptr};
    std::map<size_t, VolumeGeometry> _volumeGeometries;

    // Fields
    ::optix::Buffer _fieldBuffer{nullptr};
    std::map<size_t, FieldGeometry> _fieldGeometries;

    // Streamlines
    struct Streamlines
    {
        ::optix::Buffer vertices_buffer{nullptr};
        ::optix::Buffer color_buffer{nullptr};
        ::optix::Buffer indices_buffer{nullptr};
    };
    std::map<size_t, Streamlines> _streamlinesBuffers;
    std::map<size_t, ::optix::Geometry> _optixStreamlines;

    // Materials and textures
    std::map<std::string, ::optix::Buffer> _optixTextures;
    std::map<std::string, ::optix::TextureSampler> _optixTextureSamplers;

    // User Data
    ::optix::Buffer _userDataBuffer{nullptr};

    bool _boundingBoxBuilt = false;
};
} // namespace optix
} // namespace engine
} // namespace core