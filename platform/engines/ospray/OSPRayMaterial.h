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

#include <ospray.h>
#include <platform/core/engineapi/Material.h>

namespace core
{
namespace engine
{
namespace ospray
{
class OSPRayMaterial : public Material
{
public:
    OSPRayMaterial(const PropertyMap& properties = {}, const bool backgroundMaterial = false)
        : Material(properties)
        , _isBackGroundMaterial(backgroundMaterial)
    {
    }
    ~OSPRayMaterial();

    /** Noop until commit(renderer) is called. */
    void commit() final;

    /** Instance the actual renderer specific object for this material.
        This operation always creates a new ISPC side material.
     */
    void commit(const std::string& renderer);

    OSPMaterial getOSPMaterial() { return _ospMaterial; }

private:
    OSPTexture _createOSPTexture2D(Texture2DPtr texture);
    OSPMaterial _ospMaterial{nullptr};
    bool _isBackGroundMaterial{false};
    std::string _renderer;
};
} // namespace ospray
} // namespace engine
} // namespace core