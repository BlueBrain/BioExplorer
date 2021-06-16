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

#include "DepthRenderer.h"

// ospray
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "DepthRenderer_ispc.h"

using namespace ospray;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void DepthRenderer::commit()
{
    Renderer::commit();

    _infinity = getParam1f("infinity", 1e6f);

    ispc::DepthRenderer_set(getIE(), spp, _infinity);
}

DepthRenderer::DepthRenderer()
{
    ispcEquivalent = ispc::DepthRenderer_create(this);
}

OSP_REGISTER_RENDERER(DepthRenderer, depth);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer