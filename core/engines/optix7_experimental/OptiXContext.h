/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "OptiXCamera.h"
#include "OptiXUtils.h"

#include <core/brayns/common/Types.h>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace brayns
{
enum class OptixGeometryType
{
    sphere,
    cone,
    cylinder,
    triangleMesh
};

class OptiXContext
{
public:
    static OptiXContext& getInstance()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_instance)
            _instance = new OptiXContext();
        return *_instance;
    }

    // Camera
    void addCamera(const std::string& name, OptiXCameraPtr camera);
    OptiXCameraPtr getCamera(const std::string& name);
    void setCamera(const std::string& name);

// Geometry
#if 0
    Geometry createGeometry(const OptixGeometryType type);
    GeometryGroup createGeometryGroup(const bool compact);
    Group createGroup();
    Material createMaterial();

    // Textures
    TextureSampler createTextureSampler(Texture2DPtr texture);

    // Others
    void addRenderer(const std::string& name, OptiXShaderProgramPtr program);
    OptiXShaderProgramPtr getRenderer(const std::string& name);
#else
    OptixModule createModule(const OptixGeometryType type);
#endif

    State& getState() { return _state; }
    std::vector<OptixProgramGroup>& getProgramGroups() { return _programGroups; }

    void linkPipeline();
    const bool pipelineInitialized() const { return _pipelineInitialized; }

    std::unique_lock<std::mutex> getScopeLock() { return std::unique_lock<std::mutex>(_mutex); }

    static OptiXContext* _instance;
    static std::mutex _mutex;

private:
    OptiXContext();
    ~OptiXContext();

    void _initialize();

    void _createCameraModules();
    void _createCameraPrograms();

    void _createShadingModules();

    void _createGeometryModules();
    void _createGeometryPrograms();

    void _initLaunchParams();

    std::vector<OptixProgramGroup> _programGroups;
    std::map<std::string, OptiXCameraPtr> _cameras;
    std::string _currentCamera;

    State _state;
    bool _pipelineInitialized{false};
#if 0
    std::unordered_map<void*, TextureSampler> _optixTextureSamplers;
#endif
};
} // namespace brayns
