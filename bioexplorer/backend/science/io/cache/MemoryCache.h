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
