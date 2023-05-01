/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#pragma once

// obj
#include "SonataExplorerMaterial.h"

// ospray
#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/render/Renderer.h>

// system
#include <vector>

namespace sonataexplorer
{
/**
 * The SonataExplorerAbstractRenderer class implements a base renderer for all
 * Brayns custom implementations
 */
class SonataExplorerAbstractRenderer : public ospray::Renderer
{
public:
    void commit() override;

protected:
    bool _useHardwareRandomizer;

    std::vector<void*> _lightArray;
    void** _lightPtr;
    ospray::Data* _lightData;

    SonataExplorerMaterial* _bgMaterial;

    float _timestamp{0.f};
    ospray::uint32 _maxBounces{10};
    float _exposure{1.f};
};
} // namespace sonataexplorer
