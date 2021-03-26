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
    /**
     * @brief Construct a new OOCManager object
     *
     * @param scene 3D Scene
     * @param camera Camera
     * @param arguments Command line arguments
     */
    OOCManager(Scene& scene, const Camera& camera,
               const CommandLineArguments& arguments);

    /**
     * @brief Destroy the OOCManager object
     *
     */
    ~OOCManager() {}

    /**
     * @brief Set the Frame Buffer object
     *
     * @param frameBuffer A reference to the Brayns frame buffer
     */
    void setFrameBuffer(FrameBuffer* frameBuffer)
    {
        _frameBuffer = frameBuffer;
    }

    const FrameBuffer* getFrameBuffer() const { return _frameBuffer; }

    /**
     * @brief Starts a thread that takes care of loading the bricks according to
     * the current camera position
     *
     */
    void loadBricks();

    /**
     * @brief Get the Description object
     *
     * @return const std::string&
     */
    const std::string& getDescription() const { return _description; }

    /**
     * @brief Get the Scene Size
     *
     * @return const Vector3f& Scene size
     */
    const Vector3f& getSceneSize() const { return _sceneSize; }

    /**
     * @brief Get the Brick Size
     *
     * @return const Vector3f& Brick size
     */
    const Vector3f& getBrickSize() const { return _brickSize; }

    /**
     * @brief Get the Show Grid value, read from the command line parameters
     *
     * @return true a grid should be shown, false otherwise
     */
    const bool getShowGrid() const { return _showGrid; }

private:
    void _parseArguments(const CommandLineArguments& arguments);
    void _loadBricks();

    Scene& _scene;
    const Camera& _camera;
    FrameBuffer* _frameBuffer{nullptr};

    std::string _description;
    Vector3f _sceneSize;
    Vector3f _brickSize;
    uint32_t _nbBricks{0};
    float _updateFrequency{1.f};
    int32_t _nbVisibleBricks{0};
    bool _unloadBricks{false};
    bool _showGrid{false};
    uint32_t _nbBricksPerCycle{5};

    // IO
#ifdef USE_PQXX
    std::string _dbConnectionString;
    std::string _dbSchema;
#else
    std::string _bricksFolder;
#endif
};
} // namespace bioexplorer
