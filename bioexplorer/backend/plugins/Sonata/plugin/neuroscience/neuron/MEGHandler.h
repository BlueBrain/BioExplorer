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

#include <plugin/neuroscience/common/Types.h>

#include <platform/core/common/simulation/AbstractAnimationHandler.h>

#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Scene.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
/**
 * @brief The MEGHandler class handles electro-magnetic fields data
 * structures
 */
class MEGHandler : public core::AbstractAnimationHandler
{
public:
    /**
     * @brief Constructs a MEGHandler object from a circuit configuration and a
     * report name.
     *
     * @param circuitConfiguration A string representing the circuit
     * configuration.
     * @param reportName A string representing the voltage report name.
     * @param synchronous A boolean indicating whether synchronous mode is
     * enabled or not.
     */
    MEGHandler(const std::string& circuitConfiguration, const std::string& reportName, const bool synchronous);

    /**
     * @brief Construct a new MEGHandler object
     *
     * @param rhs A copy of the MEGHandler object
     */
    MEGHandler(const MEGHandler& rhs);

    /**
     * @brief Destroy the Fields Handler object
     *
     */
    ~MEGHandler();

    /**
     * @brief Builds a Model object based on simulation parameters.
     *
     * @param model A reference to a Model object.
     * @param voxelSize The size of a single voxel in the MEG data.
     * @param density The density of the MEG data.
     * @return A ModelMetadata object representing the simulation parameters.
     * @throws An exception if density is greater than 1 or less than or equal
     * to 0.
     */
    core::ModelMetadata buildModel(core::Model& model, const double voxelSize, const double density);

    /**
     * @brief Get the Frame Data object
     *
     * @return void* A buffer of the data for the current frame
     */
    void* getFrameData(const uint32_t) final;

    /**
     * @brief Current state of the handler
     *
     * @return true The data is loaded in memory and available
     * @return false The data is not yet available
     */
    bool isReady() const final { return true; }

    /**
     * @brief Clone the AbstractAnimationHandler
     *
     * @return AbstractSimulationHandlerPtr Clone of the
     * AbstractAnimationHandler
     */
    core::AbstractSimulationHandlerPtr clone() const final;

    /**
     * @brief Get the Dimensions of the octree
     *
     * @return const Vector3ui& Dimensions of the octree
     */
    const core::Vector3ui& getDimensions() const { return _dimensions; }

    /**
     * @brief Get the voxel spacing information
     *
     * @return const Vector3f& The voxel spacing information
     */
    const core::Vector3f& getSpacing() const { return _spacing; }

    /**
     * @brief Get the offset of the octree
     *
     * @return const Vector3f& Offset of the octree
     */
    const core::Vector3f& getOffset() const { return _offset; }

    /**
     * @brief Returns whether the object is operating in synchronous mode.
     *
     * @return true if the object is operating in synchronous mode, false
     * otherwise.
     */
    bool isSynchronized() const { return _synchronousMode; }

private:
    /**
     * @brief This method takes a Model instance and builds its metadata with
     * specified voxel size and density.
     *
     * @param model the Model instance to build metadata for.
     * @param voxelSize the desired size of each voxel in the octree.
     * @param density the desired density for the octree, represented as the
     * inverse of voxel density.
     * @return the ModelMetadata instance containing octree, voxel size,
     * density, volume size, offset, dimensions, spacing, frame data size, and
     * flat indices and data. */
    void _buildOctree();

    /**
     * @brief Checks if the current frame is loaded.
     *
     * @return True if the frame is loaded, false otherwise.
     */
    bool _isFrameLoaded() const;

    /**
     * @brief Triggers the loading process for a specific frame.
     *
     * @param frame The frame to be loaded.
     */
    void _triggerLoading(const uint32_t frame);

    /**
     * @brief Makes a frame ready for processing.
     *
     * @param frame An integer specifying the frame number to be processed.
     * @return 'true' if the frame has been made ready successfully, 'false'
     * otherwise.
     */
    bool _makeFrameReady(const uint32_t frame);

    bool _synchronousMode{false};
    core::Vector3ui _dimensions;
    core::Vector3f _spacing;
    core::Vector3f _offset;
    uint64_t _startDataIndex{0};
    uint64_t _startFrame{0};
    core::Boxd _bounds;
    double _voxelSize{0.1};
    double _density{1.0};
    common::Matrix4fs _transformations;

    common::CompartmentReportPtr _report{nullptr};
    std::future<brion::Frame> _currentFrameFuture;
    bool _ready{false};
};
typedef std::shared_ptr<MEGHandler> MEGHandlerPtr;
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
