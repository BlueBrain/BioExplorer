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
