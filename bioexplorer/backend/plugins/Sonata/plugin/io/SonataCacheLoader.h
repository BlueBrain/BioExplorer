/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <plugin/api/SonataExplorerParams.h>

#include <platform/core/common/Types.h>
#include <platform/core/common/loader/Loader.h>

#include <set>
#include <vector>

namespace sonataexplorer
{
namespace io
{
namespace loader
{
namespace servus
{
class URI;
}

/**
 * Load circuit from BlueConfig or CircuitConfig file, including simulation.
 */
class SonataCacheLoader : public core::Loader
{
public:
    SonataCacheLoader(core::Scene& scene, core::PropertyMap&& loaderParams = {});

    std::string getName() const final;

    strings getSupportedStorage() const final;

    bool isSupported(const std::string& storage, const std::string& extension) const final;

    static core::PropertyMap getCLIProperties();

    core::PropertyMap getProperties() const final;

    core::ModelDescriptorPtr importFromBlob(core::Blob&& blob, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final;

    core::ModelDescriptorPtr importFromStorage(const std::string& storage, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final;

    void exportToFile(const core::ModelDescriptorPtr modelDescriptor, const std::string& filename);

private:
    std::string _readString(std::ifstream& f) const;
    core::PropertyMap _defaults;
};
} // namespace loader
} // namespace io
} // namespace sonataexplorer
