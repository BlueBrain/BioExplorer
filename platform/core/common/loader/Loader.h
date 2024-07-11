/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include <platform/core/common/PropertyMap.h>
#include <platform/core/common/Types.h>

#include <functional>

#ifdef BRAYNS_USE_OPENMP
#include <omp.h>
#endif

namespace core
{
/**
 * A class for providing progress feedback
 */
class LoaderProgress
{
public:
    /**
     * The callback for each progress update with the signature (message,
     * fraction of progress in 0..1 range)
     */
    using CallbackFn = std::function<void(const std::string&, float)>;

    LoaderProgress(CallbackFn callback)
        : _callback(std::move(callback))
    {
    }

    LoaderProgress() = default;
    ~LoaderProgress() = default;

    /**
     * Update the current progress of an operation and call the callback
     */
    void updateProgress(const std::string& message, const float fraction) const
    {
#ifdef BRAYNS_USE_OPENMP
        if (omp_get_thread_num() == 0)
#endif
            if (_callback)
                _callback(message, fraction);
    }

    CallbackFn _callback;
};

/**
 * A base class for data loaders to unify loading data from blobs and files, and
 * provide progress feedback.
 */
class Loader
{
public:
    Loader(Scene& scene)
        : _scene(scene)
    {
    }

    virtual ~Loader() = default;

    /**
     * @return The loaders supported file extensions
     */
    virtual std::vector<std::string> getSupportedStorage() const = 0;

    /**
     * @return The loader name
     */
    virtual std::string getName() const = 0;

    /**
     * @return The loader properties
     */
    virtual PropertyMap getProperties() const { return {}; }
    /**
     * Import the data from the blob and return the created model.
     *
     * @param blob the blob containing the data to import
     * @param callback Callback for loader progress
     * @param properties Properties used for loading
     * @return the model that has been created by the loader
     */
    virtual ModelDescriptorPtr importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                              const PropertyMap& properties) const = 0;

    /**
     * Import the data from the given file or database schema and return the created model.
     *
     * @param storage the file or database schema containing the data to import
     * @param callback Callback for loader progress
     * @param properties Properties used for loading
     * @return the model that has been created by the loader
     */
    virtual ModelDescriptorPtr importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                 const PropertyMap& properties) const = 0;

    /**
     * Query the loader if it can load the given file
     */
    virtual bool isSupported(const std::string& filename, const std::string& extension) const = 0;

protected:
    Scene& _scene;
};
} // namespace core
