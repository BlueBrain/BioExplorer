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

#include <science/io/db/DBConnector.h>

#include <platform/core/parameters/GeometryParameters.h>

namespace bioexplorer
{
namespace io
{
const int32_t UNDEFINED_BOX_ID = std::numeric_limits<int32_t>::max();

/**
 * Load molecular systems from an optimized binary representation of the 3D
 * scene
 */
class CacheLoader : public core::Loader
{
public:
    /**
     * @brief Construct a new Bio Explorer Loader object
     *
     * @param scene Scene to which the file contents should be loaded
     * @param loaderParams Loader parameters
     */
    CacheLoader(core::Scene& scene, core::PropertyMap&& loaderParams = {});

    /**
     * @brief Get the name of the loader
     *
     * @return A string containing the name of the loader
     */
    std::string getName() const final;

    /**
     * @brief Get the list of extensions supported by loaded
     *
     * @return The list of extensions supported by loaded
     */
    strings getSupportedStorage() const final;

    /**
     * @brief Returns whever a file extention is supported by the loader
     *
     * @param filename Name of the file
     * @param extension Extension of the file
     * @return true if the file extension is supported by the loader, false
     * otherwise
     */
    bool isSupported(const std::string& storage, const std::string& extension) const final;

    /**
     * @brief Returns the list of loader command line arguments
     *
     * @return The list of loader command line arguments
     */
    static core::PropertyMap getCLIProperties();

    /**
     * @brief Returns the list of loader properties
     *
     * @return The list of loader properties
     */
    core::PropertyMap getProperties() const final;

    /**
     * @brief Imports a 3D scene from an in-memory blob storage
     *
     * @param blob In-memory blob storage
     * @param callback Callback object providing the status of the loading
     * process
     * @param properties Loader properties
     * @return A core model if loading is successful
     */
    core::ModelDescriptorPtr importFromBlob(core::Blob&& blob, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final;

    /**
     * @brief Imports a 3D scene from file
     *
     * @param filename Full path of the file
     * @param callback Callback object providing the status of the loading
     * process
     * @param properties Loader properties
     * @return A core model if loading is successful
     */
    core::ModelDescriptorPtr importFromStorage(const std::string& storage, const core::LoaderProgress& callback,
                                               const core::PropertyMap& properties) const final;

    /**
     * @brief
     *
     * @param filename
     * @param callback
     * @param properties
     * @return std::vector<ModelDescriptorPtr>
     */
    std::vector<core::ModelDescriptorPtr> importModelsFromFile(
        const std::string& filename, const int32_t brickId = UNDEFINED_BOX_ID,
        const core::LoaderProgress& callback = core::LoaderProgress(),
        const core::PropertyMap& properties = core::PropertyMap()) const;

    /**
     * @brief Exports an optimized binary representation the 3D scene to a file
     *
     * @param filename Full path of the file
     */
    void exportToFile(const std::string& filename, const core::Boxd& bounds) const;

    /**
     * @brief
     *
     * @param brickId
     * @return std::vector<ModelDescriptorPtr>
     */
    std::vector<core::ModelDescriptorPtr> importBrickFromDB(const int32_t brickId) const;

    /**
     * @brief Exports an optimized binary representation the 3D scene to a DB
     *
     */
    void exportBrickToDB(const int32_t brickId, const core::Boxd& bounds) const;

    /**
     * @brief Exports atom information from the 3D scene to a file
     *
     * @param filename Full path of the file
     * @param format File format to be used for the export
     */
    void exportToXYZ(const std::string& filename, const common::XYZFileFormat format) const;

#ifdef USE_LASLIB
    /**
     * @brief Exports spheres in 3D scene to LAS file
     *
     * @param filename Full path of the file
     * @param format File format to be used for the export
     */
    void exportToLas(const std::string& filename, const uint32_ts& modelIds, const uint32_ts& materialIds,
                     const bool exportColors = false) const;
#endif

private:
    std::string _readString(std::stringstream& f) const;

    core::ModelDescriptorPtr _importModel(std::stringstream& buffer, const int32_t brickId) const;

    bool _exportModel(const core::ModelDescriptorPtr modelDescriptor, std::stringstream& buffer,
                      const core::Boxd& bounds) const;

    core::PropertyMap _defaults;
};
} // namespace io
} // namespace bioexplorer
