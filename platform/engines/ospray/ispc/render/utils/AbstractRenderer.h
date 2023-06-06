/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#ifndef ABSTRACTRENDERER_H
#define ABSTRACTRENDERER_H

// obj
#include "AdvancedMaterial.h"

// ospray
#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/render/Renderer.h>

// system
#include <vector>

namespace core
{
/**
 * The AbstractRenderer class implements a base renderer for all Core custom
 * implementations
 */
class AbstractRenderer : public ospray::Renderer
{
public:
    void commit() override;

protected:
    std::vector<void*> _lightArray;
    void** _lightPtr;

    ospray::Data* _lightData;

    core::AdvancedMaterial* _bgMaterial;
    float _timestamp;

    bool _useHardwareRandomizer;
    ospray::uint32 _randomNumber;
};
} // namespace core

#endif // ABSTRACTRENDERER_H
