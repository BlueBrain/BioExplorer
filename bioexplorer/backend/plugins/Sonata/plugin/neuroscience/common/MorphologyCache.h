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
