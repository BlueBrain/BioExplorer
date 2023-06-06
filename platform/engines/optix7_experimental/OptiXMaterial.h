/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <platform/core/engineapi/Material.h>

#include <map>

namespace core
{
class OptiXMaterial : public Material
{
public:
    OptiXMaterial();
    ~OptiXMaterial();

    void commit() final;
    bool isTextured() const;

#if 0
    ::optix::Material getOptixMaterial() { return _optixMaterial; }
#endif
    auto getTextureSampler(const TextureType type) const
    {
#if 0
        return _textureSamplers.at(type);
#else
        return nullptr;
#endif
    }

private:
#if 0
    ::optix::Material _optixMaterial{nullptr};
    std::map<TextureType, ::optix::TextureSampler> _textureSamplers;
#endif
};
} // namespace core
