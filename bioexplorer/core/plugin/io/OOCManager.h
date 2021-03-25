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

namespace bioexplorer
{
using namespace brayns;

class OOCManager
{
public:
    OOCManager(Scene& scene, const Camera& camera,
               const CommandLineArguments& arguments);
    ~OOCManager() {}

    /**
     * @brief
     *
     */
    void loadBricks();

    /**
     * @brief Get the Description object
     *
     * @return const std::string&
     */
    const std::string& getDescription() { return _description; }

private:
    void _parseArguments(const CommandLineArguments& arguments);
    void _loadBricks();

    Scene& _scene;
    const Camera& _camera;

    std::string _description;
    Vector3f _sceneSize;
    Vector3f _brickSize;
    uint32_t _nbBricks{0};
    float _updateFrequency{1.f};
    int32_t _nbVisibleBricks{0};
    bool _unloadBricks{false};

    // IO
#ifdef USE_PQXX
    std::string _dbConnectionString;
    std::string _dbSchema;
#else
    std::string _bricksFolder;
#endif
};
} // namespace bioexplorer
