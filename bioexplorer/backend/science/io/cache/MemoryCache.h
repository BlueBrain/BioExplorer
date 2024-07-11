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

#include <science/common/Types.h>

namespace bioexplorer
{
namespace io
{
/**
 * @brief MemoryCache is a singleton class that caches morphologies in memory when enabled
 *
 */
class MemoryCache
{
public:
    /**
     * @brief Construct a MemoryCache object
     *
     */
    MemoryCache() {}

    /**
     * @brief Get the Instance object
     *
     * @return MemoryCache* Pointer to the object
     */
    static MemoryCache* getInstance()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_instance)
            _instance = new MemoryCache();
        return _instance;
    }

    void setEnabled(const bool enabled);
    const bool getEnabled() const { return _enabled; };

    /**
     * @brief Get morphology sections from cache
     *
     * @return Sections
     */
    const morphology::SectionMap& getNeuronSections(const db::DBConnector& connector, const uint64_t neuronId,
                                                    const details::NeuronsDetails& details);

    static std::mutex _mutex;
    static MemoryCache* _instance;

private:
    ~MemoryCache() {}

    bool _enabled{false};
    std::map<uint64_t, morphology::SectionMap> _sections;
};
} // namespace io
} // namespace bioexplorer
