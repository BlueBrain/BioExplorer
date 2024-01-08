/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

#include <platform/core/common/Types.h>
#include <platform/core/common/loader/Loader.h>
#include <platform/core/parameters/GeometryParameters.h>

#include <set>

namespace medicalimagingexplorer
{
namespace dicom
{
using uint8_ts = std::vector<uint8_t>;

struct DICOMImageDescriptor
{
    std::string path;
    core::DataType dataType;
    core::Vector2ui dimensions{0, 0};
    core::Vector3f position{0, 0, 0};
    core::Vector2f pixelSpacing{1, 1};
    core::Vector2f dataRange;
    uint8_ts buffer;
    uint16_t nbFrames;
};
using DICOMImageDescriptors = std::vector<DICOMImageDescriptor>;

class DICOMLoader : public core::Loader
{
public:
    DICOMLoader(core::Scene& scene, const core::GeometryParameters& geometryParameters,
                core::PropertyMap&& loaderParams);

    std::string getName() const final;

    strings getSupportedStorage() const final;

    bool isSupported(const std::string& storage, const std::string& extension) const final;

    static core::PropertyMap getCLIProperties();

    core::ModelDescriptorPtr importFromStorage(const std::string& storage, const core::LoaderProgress& callback,
                                               const core::PropertyMap& properties) const final;

    core::ModelDescriptorPtr importFromBlob(core::Blob&& blob, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final;

    core::ModelDescriptorPtr importFromFolder(const std::string& path);

private:
    void _setDefaultTransferFunction(core::Model& model, const core::Vector2f& dataRange) const;

    std::string _dataTypeToString(const core::DataType& dataType) const;

    DICOMImageDescriptors _parseDICOMImagesData(const std::string& fileName, core::ModelMetadata& metadata) const;

    void _readDICOMFile(const std::string& path, DICOMImageDescriptor& imageDescriptor) const;

    core::ModelDescriptorPtr _readDirectory(const std::string& path, const core::LoaderProgress& callback) const;

    core::ModelDescriptorPtr _readFile(const std::string& path) const;

    const core::GeometryParameters& _geometryParameters;
    core::PropertyMap _loaderParams;
};
} // namespace dicom
} // namespace medicalimagingexplorer