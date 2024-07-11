/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
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