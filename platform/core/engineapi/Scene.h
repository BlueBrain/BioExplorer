/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <platform/core/common/Api.h>
#include <platform/core/common/BaseObject.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/loader/LoaderRegistry.h>
#include <platform/core/engineapi/LightManager.h>

#include <shared_mutex>

SERIALIZATION_ACCESS(Scene)

namespace core
{
/**
 * @brief Scene object
 * This object contains collections of geometries, materials and light sources that are used to describe the 3D scene to
 * be rendered. Scene is the base class for rendering-engine-specific inherited scenes.
 */
class Scene : public BaseObject
{
public:
    /**
     * @brief Called after scene-related changes have been made before rendering the scene.
     */
    PLATFORM_API virtual void commit();

    /**
     * @brief Commits lights to renderers.
     * @return True if lights were committed, false otherwise
     */
    PLATFORM_API virtual bool commitLights() = 0;

    /**
    @brief Factory method to create an engine-specific model.
    */
    PLATFORM_API virtual ModelPtr createModel() const = 0;

    /**
     * @brief Creates a scene object responsible for handling models, simulations and light sources.
     */
    PLATFORM_API Scene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                       VolumeParameters& volumeParameters);
    /**
     * @brief Returns the bounding box of the scene
     */
    PLATFORM_API const Boxd& getBounds() const { return _bounds; }

    /**
     * @brief Gets the light manager
     */
    PLATFORM_API LightManager& getLightManager() { return _lightManager; }

    /**
     * @brief Adds a model to the scene
     * @param model The model to add
     * @throw std::runtime_error if model is empty
     * @return The ID of the added model
     * */
    PLATFORM_API size_t addModel(ModelDescriptorPtr model);

    /**
     * @brief Removes a model from the scene
     * @param id ID of the model to remove
     * @return True if model was found and removed, false otherwise
     */

    PLATFORM_API bool removeModel(const size_t id);
    /**
     * @brief Get all model descriptors
     * @return The model descriptors
     */

    PLATFORM_API const ModelDescriptors& getModelDescriptors() const { return _modelDescriptors; }

    /**
     * @brief Get a model descriptor given its ID
     * @param id The ID of the model descriptor
     * @return The model descriptor
     */
    PLATFORM_API ModelDescriptorPtr getModel(const size_t id) const;

    /**
     * @brief Builds a default scene made of a Cornell box, a reflective cube, and a transparent sphere
     */
    PLATFORM_API void buildDefault();

    /**
     * @brief Checks whether the scene is empty
     * @return True if the scene does not contain any geometry, false otherwise
     */
    PLATFORM_API bool empty() const;

    /**
     * @brief Add a clip plane to the scene.
     * @param plane The coefficients of the clip plane equation.
     * @return The clip plane ID.
     */
    PLATFORM_API size_t addClipPlane(const Plane& plane);

    /**
     * @brief Get a clip plane by its ID.
     * @param id The ID of the clip plane
     * @return A pointer to the clip plane or null if not found.
     * */
    PLATFORM_API ClipPlanePtr getClipPlane(const size_t id) const;

    /**
     * @brief Remove a clip plane by its ID, or no-op if not found.
     * @param id The ID of the clip plane to remove
     */
    PLATFORM_API void removeClipPlane(const size_t id);

    /**
     * @brief Get all clip planes in the scene
     * @return The clip planes
     */

    PLATFORM_API const ClipPlanes& getClipPlanes() const { return _clipPlanes; }

    /**
     * @brief Get the current size in bytes of the loaded geometry
     * @return The current size in bytes of the loaded geometry
     */
    PLATFORM_API size_t getSizeInBytes() const;

    /**
     * @brief Get the current number of models in the scene
     * @return The current number of models in the scene
     */
    PLATFORM_API size_t getNumModels() const;

    /**
     * @brief Initializes materials for all models in the scene
     * @param colorMap Color map to use for every individual model
     */
    PLATFORM_API void setMaterialsColorMap(MaterialsColorMap colorMap);

    /**
     * @brief Set a new environment map as the background image
     * @param envMap Filepath to the environment map
     * @return False if the new map could not be set, true otherwise
     */
    PLATFORM_API bool setEnvironmentMap(const std::string& envMap);

    /**
     * @brief Get the current environment map texture file, or empty if no environment is set
     * @return The environment map texture file, or empty if no environment is set
     * */
    PLATFORM_API const std::string& getEnvironmentMap() const { return _environmentMap; }

    /**
     * @brief Check if an environment map is currently set in the scene
     * @return True if an environment map is currently set in the scene, false otherwise
     */
    PLATFORM_API bool hasEnvironmentMap() const;

    /**
     * @brief Get the background material
     * @return The background material
     */
    PLATFORM_API MaterialPtr getBackgroundMaterial() const { return _backgroundMaterial; }

    /**
     * @brief Load a model from the given blob
     * @param blob The blob containing the data to import
     * @param params Parameters for the model to be loaded
     * @param cb The callback for progress updates from the loader
     * @return The model that has been added to the scene
     */
    PLATFORM_API ModelDescriptorPtr loadModel(Blob&& blob, const ModelParams& params, LoaderProgress cb);

    /**
     * @brief Load a model from the given file
     * @param path The file or folder containing the data to import
     * @param params Parameters for the model to be loaded
     * @param cb The callback for progress updates from the loader
     * @return The model that has been added to the scene
     */
    PLATFORM_API ModelDescriptorPtr loadModel(const std::string& path, const ModelParams& params, LoaderProgress cb);

    /**
     * @brief Apply the given functor to every model in the scene
     * @param functor The functor to be applied
     */
    PLATFORM_API void visitModels(const std::function<void(Model&)>& functor);

    /**
     * @brief Get the registry for all supported loaders of this scene.
     * @return The loader registry
     */
    PLATFORM_API LoaderRegistry& getLoaderRegistry() { return _loaderRegistry; }

    /** @internal */
    PLATFORM_API auto acquireReadAccess() const { return std::shared_lock<std::shared_timed_mutex>(_modelMutex); }

    /**
     * @brief Copy the scene from another scene
     * @param rhs Scene to copy from
     */
    PLATFORM_API void copyFrom(const Scene& rhs);

    /**
     * @brief Compute the bounds of the geometry handled by the scene
     */
    PLATFORM_API void computeBounds();

protected:
    /**
     * @brief Check whether this scene supports scene updates from any thread
     * @return True if this scene supports scene updates from any thread, false otherwise
     */
    virtual bool supportsConcurrentSceneUpdates() const { return false; }

    void _loadIBLMaps(const std::string& envMap);

    AnimationParameters& _animationParameters;
    GeometryParameters& _geometryParameters;
    VolumeParameters& _volumeParameters;
    MaterialPtr _backgroundMaterial{nullptr};
    std::string _environmentMap;

    size_t _modelID{0};
    ModelDescriptors _modelDescriptors;
    mutable std::shared_timed_mutex _modelMutex;

    LightManager _lightManager;
    ClipPlanes _clipPlanes;

    LoaderRegistry _loaderRegistry;
    Boxd _bounds;

private:
    SERIALIZATION_FRIEND(Scene)
};
} // namespace core
