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

#include "MemoryCache.h"

#include <science/common/Logs.h>
#include <science/io/db/DBConnector.h>

namespace bioexplorer
{
namespace io
{
using namespace details;
using namespace db;

MemoryCache* MemoryCache::_instance = nullptr;
std::mutex MemoryCache::_mutex;

const morphology::SectionMap& MemoryCache::getNeuronSections(const DBConnector& connector, const uint64_t neuronId,
                                                             const NeuronsDetails& details)
{
    if (_enabled)
    {
        const auto it = _sections.find(neuronId);
        if (it != _sections.end())
            return (*it).second;
    }

    _sections[neuronId] = connector.getNeuronSections(details.populationName, neuronId, details.sqlSectionFilter);
    return _sections[neuronId];
}

void MemoryCache::setEnabled(const bool enabled)
{
    _enabled = enabled;
    PLUGIN_INFO(1, "Memory cache is " << (_enabled ? "ON" : "OFF"));
    if (!enabled)
        _sections.clear();
}
} // namespace io
} // namespace bioexplorer
