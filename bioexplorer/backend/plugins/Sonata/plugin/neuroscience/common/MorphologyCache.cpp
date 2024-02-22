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
