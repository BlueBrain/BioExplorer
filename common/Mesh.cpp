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

#include "Mesh.h"

#include <brayns/io/MeshLoader.h>

Mesh::Mesh(brayns::Scene& scene, const MeshDescriptor& md)
    : Node()
{
    const auto loader = brayns::MeshLoader(scene);
    uint8_ts contentAsChars;
    for (size_t i = 0; i < md.contents.length(); ++i)
        contentAsChars.push_back(md.contents[i]);
    brayns::Blob blob{"obj", md.name, contentAsChars};

    _modelDescriptor =
        loader.importFromBlob(std::move(blob), brayns::LoaderProgress(),
                              brayns::PropertyMap());
}
