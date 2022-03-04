/* Copyright (c) 2018-2021, EPFL/Blue Brain Project
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

#include <plugin/api/Params.h>
#include <plugin/common/Node.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace morphology
{
using namespace brayns;
using namespace common;

/**
 * Load astrocytes from database
 */
class Astrocytes : public common::Node
{
public:
    Astrocytes(Scene& scene, const AstrocytesDetails& details);

private:
    void _buildModel();
    const AstrocytesDetails _details;
    Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
