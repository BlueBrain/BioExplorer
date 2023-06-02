/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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
namespace io
{
using namespace core;
using namespace details;

/**
 * @brief The OOCManager classes manager the out-of-core engine of the
 * BioExplorer. The scene is devided into bricks of a given size that are, in
 * the current implementation, stored in a PostgreSQL database. The out-of-core
 * engine is activated via command line parameters only. By settings the
 * --ooc-enabled command line parameter, the out-of-core engine is enabled and,
 * in order to avoid unnecessary network traffic, the scene contents are not
 * broadcasted anymore to the BioExplorer clients (Python notebook or User
 * Inferface). Note that ENV_ROCKETS_DISABLE_SCENE_BROADCASTING environment
 * variable has to be set to order Brayns not to broadcast the scene changes.
 * The out-of-core mode is a read-only feature and does not allow any
 * modification of the scene. When the camera position changes, the OOCManager
 * identifies the visible bricks and loads them as a background task. For
 * performance reasons, models are loaded as invisible. This allows the
 * ray-tracing engine to create the BVH before the model is added to the scene.
 * Once all models are loaded, they are added to the scene. Note that the
 * OOCManager only does work when the frame buffer accumulation is greater than
 * 1, which means that the camera is not moving.
 * If the --ooc-unload-bricks command line parameter is set, invisible bricks
 * are unloaded from memory and removed from the scene. The
 * --ooc-update-frequency command line parameters defines (in seconds) at which
 * frequency models should be loaded. The --ooc-nb-bricks-per-cycle command line
 * parameter defines how many bricks can be loaded at each rendering cycle. The
 * connection to the database is defined by the --ooc-db-host, --ooc-db-port,
 * --ooc-db-name,--ooc-db-user,--ooc-db-password and --ooc-db-schema parameters.
 * If the BioExplorer fails to find the pqxx dependency to connect to the
 * PostgreSQL database, it falls back to a file-based mode where bricks can be
 * stored as BioExplorer cache files, in the folder defined by the
 * --ooc-bricks-folder command line argument. The --ooc-show-grid command line
 * parameter can be used to show a grid corresponding the positions of the
 * bricks in the scene.
 *
 */
class OOCManager
{
public:
    /**
     * @brief Construct a new OOCManager object
     *
     * @param scene 3D Scene to wich bricks are added
     * @param camera Camera used to determine the position of the viewer in the
     * scene.
     * @param arguments Command line arguments (See class description for
     * details)
     */
    OOCManager(Scene& scene, const Camera& camera, const CommandLineArguments& arguments);

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
    void setFrameBuffer(FrameBuffer* frameBuffer) { _frameBuffer = frameBuffer; }

    /**
     * @brief Get the frame buffer
     *
     * @return const FrameBuffer* A pointer to the frame buffer
     */
    const FrameBuffer* getFrameBuffer() const { return _frameBuffer; }

    /**
     * @brief Starts a thread that takes care of loading the bricks according to
     * the current camera position
     *
     */
    void loadBricks();

    /**
     * @brief Get the Scene Configuration object
     *
     * @return const std::string&
     */
    const OOCSceneConfigurationDetails& getSceneConfiguration() const { return _sceneConfiguration; }

    /**
     * @brief Get the Show Grid value, read from the command line parameters
     *
     * @return true a grid should be shown, false otherwise
     */
    bool getShowGrid() const { return _showGrid; }

    /**
     * @return Get the number of visible bricks surrounding the camera position
     */
    int32_t getVisibleBricks() const { return _nbVisibleBricks; }

    /**
     * @return Get the update frequency (in seconds) of the bricks in the scene
     */
    double getUpdateFrequency() const { return _updateFrequency; }

    /**
     * @return Get current loading progress
     */
    double getProgress() const { return _progress; }

    /**
     * @return Get average loading time (in milliseconds)
     */
    double getAverageLoadingTime() const { return _averageLoadingTime; }

private:
    void _parseArguments(const CommandLineArguments& arguments);
    void _loadBricks();

    Scene& _scene;
    const Camera& _camera;
    FrameBuffer* _frameBuffer{nullptr};

    OOCSceneConfigurationDetails _sceneConfiguration;
    double _updateFrequency{1.f};
    int32_t _nbVisibleBricks{0};
    bool _unloadBricks{false};
    bool _showGrid{false};
    uint32_t _nbBricksPerCycle{5};
    double _progress{0.f};
    double _averageLoadingTime{0.f};

    // IO
    std::string _dbConnectionString;
    std::string _dbSchema;
};
} // namespace io
} // namespace bioexplorer
