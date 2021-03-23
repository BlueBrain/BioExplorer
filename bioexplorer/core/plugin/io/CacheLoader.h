/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <plugin/common/Types.h>

#include <brayns/common/loader/Loader.h>
#include <brayns/common/types.h>
#include <brayns/parameters/GeometryParameters.h>

namespace bioexplorer
{
using namespace brayns;

const int32_t UNDEFINED_BOX_ID = std::numeric_limits<int32_t>::max();

/**
 * Load molecular systems from an optimized binary representation of the 3D
 * scene
 */
class CacheLoader : public Loader
{
public:
    /**
     * @brief Construct a new Bio Explorer Loader object
     *
     * @param scene Scene to which the file contents should be loaded
     * @param loaderParams Loader parameters
     */
    CacheLoader(Scene& scene, PropertyMap&& loaderParams = {});

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
    std::vector<std::string> getSupportedExtensions() const final;

    /**
     * @brief Returns whever a file extention is supported by the loader
     *
     * @param filename Name of the file
     * @param extension Extension of the file
     * @return true if the file extension is supported by the loader, false
     * otherwise
     */
    bool isSupported(const std::string& filename,
                     const std::string& extension) const final;

    /**
     * @brief Returns the list of loader command line arguments
     *
     * @return The list of loader command line arguments
     */
    static PropertyMap getCLIProperties();

    /**
     * @brief Returns the list of loader properties
     *
     * @return The list of loader properties
     */
    PropertyMap getProperties() const final;

    /**
     * @brief Imports a 3D scene from an in-memory blob storage
     *
     * @param blob In-memory blob storage
     * @param callback Callback object providing the status of the loading
     * process
     * @param properties Loader properties
     * @return A brayns model if loading is successfull
     */
    ModelDescriptorPtr importFromBlob(
        Blob&& blob, const LoaderProgress& callback,
        const PropertyMap& properties) const final;

    /**
     * @brief Imports a 3D scene from file
     *
     * @param filename Full path of the file
     * @param callback Callback object providing the status of the loading
     * process
     * @param properties Loader properties
     * @return A brayns model if loading is successfull
     */
    ModelDescriptorPtr importFromFile(
        const std::string& filename, const LoaderProgress& callback,
        const PropertyMap& properties) const final;

    /**
     * @brief
     *
     * @param filename
     * @param callback
     * @param properties
     * @return std::vector<ModelDescriptorPtr>
     */
    std::vector<ModelDescriptorPtr> importModelsFromFile(
        const std::string& filename, const int32_t boxId = UNDEFINED_BOX_ID,
        const LoaderProgress& callback = LoaderProgress(),
        const PropertyMap& properties = PropertyMap()) const;

    /**
     * @brief Exports an optimized binary representation the 3D scene to a file
     *
     * @param filename Full path of the file
     */
    void exportToCache(const std::string& filename) const;

    /**
     * @brief Exports atom information from the 3D scene to a file
     *
     * @param filename Full path of the file
     * @param format File format to be used for the export
     */
    void exportToXYZ(const std::string& filename,
                     const XYZFileFormat format) const;

private:
    std::string _readString(std::ifstream& f) const;

    ModelDescriptorPtr _importModel(std::ifstream& file,
                                    const int32_t boxId) const;

    bool _exportModel(const ModelDescriptorPtr modelDescriptor,
                      std::ofstream& file) const;

    PropertyMap _defaults;
};
} // namespace bioexplorer
