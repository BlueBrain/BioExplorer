/*
 * Copyright (c) 2018 EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Jonas Karlsson <jonas.karlsson@epfl.ch>
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

#include <platform/core/common/loader/Loader.h>
#include <platform/core/common/loader/LoaderRegistry.h>

#include <set>

namespace core
{
class ArchiveLoader : public Loader
{
public:
    ArchiveLoader(Scene& scene, LoaderRegistry& registry);

    std::vector<std::string> getSupportedStorage() const final;
    std::string getName() const final;

    bool isSupported(const std::string& storage, const std::string& extension) const final;
    ModelDescriptorPtr importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;

    ModelDescriptorPtr importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;

private:
    ModelDescriptorPtr loadExtracted(const std::string& path, const LoaderProgress& callback,
                                     const PropertyMap& properties) const;
    LoaderRegistry& _registry;
};
} // namespace core
