/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
               GeometryParameters& geometryParameters);

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

    ::optix::GeometryGroup getGeometryGroup() const { return _geometryGroup; }
    ::optix::GeometryGroup getBoundingBoxGroup() const { return _boundingBoxGroup; }

protected:
    void _createSampler(const size_t materialId, const ::optix::Buffer& buffer, const size_t size,
                        const TextureType textureType, const RTtextureindexmode textureIndexMode,
                        const Vector2f& valueRange = Vector2f());

    void _commitTransferFunctionImpl(const Vector3fs& colors, const floats& opacities, const Vector2d valueRange) final;
    void _commitSimulationDataImpl(const float* frameData, const size_t frameSize) final;
    void _commitVolumesBuffers(const size_t materialId);

private:
    uint64_t _commitSpheres(const size_t materialId);
    uint64_t _commitCylinders(const size_t materialId);
    uint64_t _commitCones(const size_t materialId);
    uint64_t _commitSDFGeometries();
    uint64_t _commitMeshes(const size_t materialId);
    uint64_t _commitVolumes(const size_t materialId);
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