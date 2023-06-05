/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Core <https://github.com/BlueBrain/Core>
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

#ifndef XYZBLOADER_H
#define XYZBLOADER_H

#include <platform/core/common/loader/Loader.h>

namespace core
{
class XYZBLoader : public Loader
{
public:
    XYZBLoader(Scene& scene);

    std::vector<std::string> getSupportedExtensions() const final;
    std::string getName() const final;

    bool isSupported(const std::string& filename, const std::string& extension) const final;
    ModelDescriptorPtr importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;

    ModelDescriptorPtr importFromFile(const std::string& filename, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;
};
} // namespace core

#endif // XYZBLOADER_H
