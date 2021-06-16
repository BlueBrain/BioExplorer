/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#include "AlbedoRenderer.h"

// ospray
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "AlbedoRenderer_ispc.h"

using namespace ospray;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void AlbedoRenderer::commit()
{
    Renderer::commit();

    ispc::AlbedoRenderer_set(getIE(), spp);
}

AlbedoRenderer::AlbedoRenderer()
{
    ispcEquivalent = ispc::AlbedoRenderer_create(this);
}

OSP_REGISTER_RENDERER(AlbedoRenderer, albedo);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer