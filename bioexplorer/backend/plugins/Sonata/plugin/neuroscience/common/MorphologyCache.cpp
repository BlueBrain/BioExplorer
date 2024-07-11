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

#include "MorphologyCache.h"

namespace sonataexplorer
{
namespace neuroscience
{
namespace common
{
MorphologyCache* MorphologyCache::_instance = nullptr;
std::mutex MorphologyCache::_mutex;

const brain::neuron::MorphologyPtr MorphologyCache::getMorphology(const std::string& uri)
{
    if (_enabled)
        if (_morphologies.find(uri) != _morphologies.end())
            return _morphologies[uri];
    const brion::URI source(uri);
    _morphologies[uri] = std::shared_ptr<brain::neuron::Morphology>(new brain::neuron::Morphology(source));
    return _morphologies[uri];
}

const brain::neuron::Sections& MorphologyCache::getSections(const std::string& uri,
                                                            const brain::neuron::SectionTypes& sectionTypes)
{
    const auto key = std::pair<std::string, const brain::neuron::SectionTypes>(uri, sectionTypes);
    if (_enabled)
        if (_sections.find(key) != _sections.end())
            return _sections[key];
    _sections[key] = getMorphology(uri)->getSections(sectionTypes);
    return _sections[key];
}

#if 0
const brain::neuron::Soma& MorphologyCache::getSoma(const std::string& uri)
{
    if (_enabled)
        if (_somas.find(uri) != _somas.end())
            return _somas[uri];
    _somas[uri] = getMorphology(uri)->getSoma();
    return _somas[uri];
}
#endif

void MorphologyCache::setEnabled(const bool enabled)
{
    _enabled = enabled;
    if (!enabled)
        _morphologies.clear();
}
} // namespace common
} // namespace neuroscience
} // namespace sonataexplorer
