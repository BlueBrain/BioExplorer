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

#include "Types.h"

namespace sonataexplorer
{
namespace neuroscience
{
namespace common
{
/**
 * @brief MorphologyCache is a singleton class that caches morphologies in memory when enabled
 *
 */
class MorphologyCache
{
public:
    /**
     * @brief Construct a MorphologyCache object
     *
     */
    MorphologyCache() {}

    /**
     * @brief Get the Instance object
     *
     * @return MorphologyCache* Pointer to the object
     */
    static MorphologyCache* getInstance()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_instance)
            _instance = new MorphologyCache();
        return _instance;
    }

    void setEnabled(const bool enabled);
    const bool getEnabled() const { return _enabled; };

    /**
     * @brief Get morphology from cache
     *
     * @return Morphology object
     */
    const brain::neuron::MorphologyPtr getMorphology(const std::string& source);

    const brain::neuron::Sections& getSections(const std::string& source,
                                               const brain::neuron::SectionTypes& sectionTypes);

#if 0 
const brain::neuron::Soma & getSoma(const std::string& source);
#endif

    static std::mutex _mutex;
    static MorphologyCache* _instance;

private:
    ~MorphologyCache() {}

    bool _enabled{false};
    std::map<std::string, brain::neuron::MorphologyPtr> _morphologies;
    std::map<std::pair<std::string, const brain::neuron::SectionTypes>, brain::neuron::Sections> _sections;
    std::map<std::string, brain::neuron::Soma> _somas;
};
} // namespace common
} // namespace neuroscience
} // namespace sonataexplorer
