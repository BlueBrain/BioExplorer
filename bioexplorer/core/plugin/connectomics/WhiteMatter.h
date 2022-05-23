/* Copyright (c) 2020-2022, EPFL/Blue Brain Project
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

#pragma once

#include <plugin/common/Node.h>

namespace bioexplorer
{
namespace connectomics
{
using namespace brayns;
using namespace common;
using namespace details;

/**
 * Load whitematter from database
 */
class WhiteMatter : public Node
{
public:
    /**
     * @brief Construct a new WhiteMatter object
     *
     * @param scene 3D scene into which the white matter should be loaded
     * @param details Set of attributes defining how the whitematter should be
     * loaded
     */
    WhiteMatter(Scene& scene, const WhiteMatterDetails& details);

private:
    void _buildModel();

    void _addStreamline(ThreadSafeContainer& container, const Vector3fs& points,
                        const uint64_t materialId);

    const WhiteMatterDetails _details;
    Scene& _scene;
};
} // namespace connectomics
} // namespace bioexplorer
