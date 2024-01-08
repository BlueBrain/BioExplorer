/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Based on OSPRay implementation
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

#include "AdvancedMaterial.h"

#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/render/Renderer.h>

#include <vector>

namespace core
{
namespace engine
{
namespace ospray
{
/**
 * The AbstractRenderer class implements a base renderer for all Core custom implementations
 */
class AbstractRenderer : public ::ospray::Renderer
{
public:
    void commit() override;

protected:
    std::vector<void*> _lightArray;
    void** _lightPtr{nullptr};
    ::ospray::Data* _lightData{nullptr};
    AdvancedMaterial* _bgMaterial{nullptr};
    bool _showBackground{false};
    double _exposure{1.0};
    float _timestamp;
    bool _useHardwareRandomizer{false};
    ::ospray::uint32 _randomNumber{0};
    bool _anaglyphEnabled{false};
    ::ospray::vec3f _anaglyphIpdOffset;
};
} // namespace ospray
} // namespace engine
} // namespace core