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
#include <platform/core/common/PropertyMap.h>
#include <platform/core/common/Transformation.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/geometry/Cone.h>
#include <platform/core/common/geometry/Curve.h>
#include <platform/core/common/geometry/Cylinder.h>
#include <platform/core/common/geometry/SDFBezier.h>
#include <platform/core/common/geometry/SDFGeometry.h>
#include <platform/core/common/geometry/Sphere.h>
#include <platform/core/common/geometry/Streamline.h>
#include <platform/core/common/geometry/TriangleMesh.h>
#include <platform/core/common/transferFunction/TransferFunction.h>

#include <set>

SERIALIZATION_ACCESS(Model)
SERIALIZATION_ACCESS(ModelParams)
SERIALIZATION_ACCESS(ModelDescriptor)
SERIALIZATION_ACCESS(ModelInstance)

namespace core
{
/**
 * @brief A struct that holds data for Signed Distance Field (SDF) calculations.
 * Since this struct holds data for SDF calculations, it is assumed that the geometries have been converted to Signed
 * Distance Fields beforehand.
 */
struct SDFGeometryData
{
    std::vector<SDFGeometry> geometries;                     // A vector of SDFGeometry objects
    std::map<size_t, std::vector<uint64_t>> geometryIndices; // A map where the keys are size_t values that represent
                                                             // the indices of geometries vector. The map values are
                                                             // vectors of uint64_t values that represent the indices of
                                                             // vertices that belong to the geometry.
    std::vector<std::vector<uint64_t>> neighbours; // A vector of vectors of uint64_t values that represent the indices
                                                   // of neighbouring vertices for each  vertex.
    std::vector<uint64_t> neighboursFlat; // A vector of uint64_t values that represents the indices of neighbouring
                                          // vertices flattened into a single vector.
};

/**
 * @brief A class representing an instance of a 3D model
 * @extends BaseObject
 */
class ModelInstance : public BaseObject
{
public:
    /**
     * @brief Default constructor
     */
    PLATFORM_API ModelInstance() = default;

    /**
    @brief Constructor with parameters
    @param visible A boolean indicating if the model instance is visible
    @param boundingBox A boolean indicating if the model instance has a bounding box
    @param transformation A Transformation object representing the model's transformation
    */
    PLATFORM_API ModelInstance(const bool visible, const bool boundingBox, const Transformation& transformation)
        : _visible(visible)
        , _boundingBox(boundingBox)
        , _transformation(transformation)
    {
    }

    /**
     * @brief Get the value of _visible
     * @return A boolean indicating if the model instance is visible
     */
    PLATFORM_API bool getVisible() const { return _visible; }

    /**
     * @brief Set the value of _visible
     * @param visible A boolean indicating if the model instance is visible
     */
    PLATFORM_API void setVisible(const bool visible) { _updateValue(_visible, visible); }

    /**
     * @brief Get the value of _boundingBox
     * @return A boolean indicating if the model instance has a bounding box
     */
    PLATFORM_API bool getBoundingBox() const { return _boundingBox; }

    /**
     * @brief Set the value of _boundingBox
     * @param enabled A boolean indicating if the model instance has a bounding box
     */
    PLATFORM_API void setBoundingBox(const bool enabled) { _updateValue(_boundingBox, enabled); }

    /**
     * @brief Get the value of _transformation
     * @return A constant reference to a Transformation object representing the model's transformation
     */
    PLATFORM_API const Transformation& getTransformation() const { return _transformation; }

    /**
     * @brief Set the value of _transformation
     * @param transformation A Transformation object representing the model's transformation
     */
    PLATFORM_API void setTransformation(const Transformation& transformation)
    {
        _updateValue(_transformation, transformation);
    }

    /**
     * @brief Set the value of _modelID
     * @param id A size_t representing the model ID
     */
    PLATFORM_API void setModelID(const size_t id) { _updateValue(_modelID, id); }

    /**
     * @brief Get the value of _modelID
     * @return A size_t representing the model ID
     */
    PLATFORM_API size_t getModelID() const { return _modelID; }

    /**
     * @brief Set the value of _instanceID
     * @param id A size_t representing the instance ID
     */
    PLATFORM_API void setInstanceID(const size_t id) { _updateValue(_instanceID, id); }

    /**
     * @brief Get the value of _instanceID
     * @return A size_t representing the instance ID
     */
    PLATFORM_API size_t getInstanceID() const { return _instanceID; }

protected:
    size_t _modelID{0};             // A size_t representing the ID of the model
    size_t _instanceID{0};          // A size_t representing the ID of the instance
    bool _visible{true};            // A boolean indicating if the model instance is visible
    bool _boundingBox{false};       // A boolean indicating if the model instance has a bounding box
    Transformation _transformation; // A Transformation object representing the model's transformation

    SERIALIZATION_FRIEND(ModelInstance)
};

/**
 * @brief The ModelParams class represents the parameters needed for initializing a model instance.
 * @extends ModelInstance.
 */
class ModelParams : public ModelInstance
{
public:
    /* Default constructor */
    PLATFORM_API ModelParams() = default;

    /**
     * @brief Constructor to create ModelParams object from a given path.
     * @param path path of the model
     */
    PLATFORM_API ModelParams(const std::string& path);

    /**
     * @brief Constructor to create ModelParams object with a given name and path.
     * @param name name of the model
     * @param path path of the model
     */
    PLATFORM_API ModelParams(const std::string& name, const std::string& path);

    /**
     * @brief Constructor to create ModelParams object with a given name, path and loader properties.
     * @param name name of the model
     * @param path path of the model
     * @param loaderProperties loader properties of the model
     */
    PLATFORM_API ModelParams(const std::string& name, const std::string& path, const PropertyMap& loaderProperties);

    /** Move constructor */
    PLATFORM_API ModelParams(ModelParams&& rhs) = default;

    /** Move assignment operator */
    PLATFORM_API ModelParams& operator=(ModelParams&& rhs) = default;

    /** Copy constructor */
    PLATFORM_API ModelParams(const ModelParams& rhs) = default;

    /** Copy assignment operator */
    PLATFORM_API ModelParams& operator=(const ModelParams& rhs) = default;

    /**
     * @brief setName sets the name of the model
     * @param name name of the model
     */
    PLATFORM_API void setName(const std::string& name) { _updateValue(_name, name); }

    /**
     * @brief getName gets the name of the model
     * @return name of the model
     */
    PLATFORM_API const std::string& getName() const { return _name; }

    /**
     * @brief setPath sets the path of the model
     * @param path path of the model
     */
    PLATFORM_API void setPath(const std::string& path) { _updateValue(_path, path); }

    /**
     * @brief getPath gets the path of the model
     * @return path of the model
     */
    PLATFORM_API const std::string& getPath() const { return _path; }

    /**
     * @brief setLoaderName sets the loader name of the model
     * @param loaderName loader name of the model
     */
    PLATFORM_API void setLoaderName(const std::string& loaderName) { _updateValue(_loaderName, loaderName); }

    /**
     * @brief getLoaderName gets the loader name of the model
     * @return loader name of the model
     */
    PLATFORM_API const std::string& getLoaderName() const { return _loaderName; }

    /**
     * @brief getLoaderProperties gets the loader properties of the model
     * @return loader properties of the model
     */
    PLATFORM_API const PropertyMap& getLoaderProperties() const { return _loaderProperties; }

    /**
     * @brief setLoaderProperties sets the loader properties of the model
     * @param pm loader properties of the model
     */
    PLATFORM_API void setLoaderProperties(const PropertyMap& pm) { _loaderProperties = pm; }

protected:
    std::string _name;             // name of the model
    std::string _path;             // path of the model
    std::string _loaderName;       // loader name of the model
    PropertyMap _loaderProperties; // loader properties of the model

    SERIALIZATION_FRIEND(ModelParams)
};

/**
 * @brief The ModelDescriptor struct defines the metadata attached to a model.
 * @extends ModelParams
 * Model descriptor are exposed via the HTTP/WS interface.
 * Enabling a model means that the model is part of scene. If disabled, the
 * model still exists in Core, but is removed from the rendered scene.
 * The visible attribute defines if the model should be visible or not. If
 * invisible, the model is removed from the BVH.
 * If set to true, the bounding box attribute displays a bounding box for the
 * current model
 */
class ModelDescriptor : public ModelParams
{
public:
    /** * Default constructor for a ModelDescriptor. */
    ModelDescriptor() = default;

    /**
     * Default move constructor for a ModelDescriptor.
     * @param rhs The r-value reference of the ModelDescriptor.
     */
    ModelDescriptor(ModelDescriptor&& rhs) = default;

    /**
     * Default move operator for a ModelDescriptor.
     * @param rhs The r-value reference of the ModelDescriptor.
     * @return A reference to this ModelDescriptor.
     */
    ModelDescriptor& operator=(ModelDescriptor&& rhs) = default;

    /**
     * Constructor for a ModelDescriptor with a given model and path.
     * @param model The model pointer.
     * @param path The path of the model.
     */
    ModelDescriptor(ModelPtr model, const std::string& path);

    /**
     * Constructor for a ModelDescriptor with a given model, path and metadata.
     * @param model The model pointer.
     * @param path The path of the model.
     * @param metadata The metadata of the model.
     */
    ModelDescriptor(ModelPtr model, const std::string& path, const ModelMetadata& metadata);

    /**
     * Constructor for a ModelDescriptor with a given model, name, path and metadata.
     * @param model The model pointer.
     * @param name The name of the model.
     * @param path The path of the model.
     * @param metadata The metadata of the model.
     */
    ModelDescriptor(ModelPtr model, const std::string& name, const std::string& path, const ModelMetadata& metadata);

    /**
     * Copy assignment operator for a ModelParams.
     * @param rhs The ModelParams to copy from.
     * @return A reference to this ModelDescriptor.
     */
    ModelDescriptor& operator=(const ModelParams& rhs);

    /**
     * Checks if the model is enabled.
     * @return True if the model is visible or has a bounding box.
     */
    bool getEnabled() const { return _visible || _boundingBox; }

    /**
     * Sets the metadata of this model.
     * @param metadata The metadata of the model.
     */
    void setMetadata(const ModelMetadata& metadata)
    {
        _metadata = metadata;
        markModified();
    }

    /**
     * Gets the metadata of this model.
     * @return The metadata of the model.
     */
    const ModelMetadata& getMetadata() const { return _metadata; }

    /**
     * Gets the const reference to the model.
     * @return The model reference.
     */
    const Model& getModel() const { return *_model; }

    /**
     * Gets the reference to the model.
     * @return The model reference.
     */
    Model& getModel() { return *_model; }

    /**
     * Adds an instance of the model.
     * @param instance The instance to add.
     */
    void addInstance(const ModelInstance& instance);

    /**
     * Removes an instance of the model.
     * @param id The id of the instance to remove.
     */
    void removeInstance(const size_t id);

    /**
     * Gets the instance of the model with the given id.
     * @param id The id of the instance to get.
     * @return The instance of the model with the given id.
     */
    ModelInstance* getInstance(const size_t id);

    /**
     * Gets all instances of the model.
     * @return All instances of the model.
     */
    const ModelInstances& getInstances() const { return _instances; }

    /**
     * Gets the bounding box of the model.
     * @return The bounding box of the model.
     */
    Boxd getBounds() const { return _bounds; }

    /**
     * Computes the bounding box of the model.
     */
    void computeBounds();

    /**
     * Sets the properties of this model.
     * @param properties The properties of the model.
     */
    void setProperties(const PropertyMap& properties)
    {
        _properties = properties;
        markModified();
    }

    /**
     * Gets the properties of this model.
     * @return The properties of the model.
     */
    const PropertyMap& getProperties() const { return _properties; }

    /**
     * Set a function that is called when this model is about to be removed.
     * @param callback The callback function.
     */
    using RemovedCallback = std::function<void(const ModelDescriptor&)>;
    void onRemoved(const RemovedCallback& callback) { _onRemovedCallback = callback; }

    /** @internal */
    void callOnRemoved()
    {
        if (_onRemovedCallback)
            _onRemovedCallback(*this);
    }

    /** @internal */
    void markForRemoval() { _markedForRemoval = true; }

    /** @internal */
    bool isMarkedForRemoval() const { return _markedForRemoval; }

    /** @internal */
    ModelDescriptorPtr clone(ModelPtr model) const;

private:
    size_t _nextInstanceID{0};
    Boxd _bounds;
    ModelMetadata _metadata;
    ModelPtr _model;
    ModelInstances _instances;
    PropertyMap _properties;
    RemovedCallback _onRemovedCallback;
    bool _markedForRemoval = false;

    SERIALIZATION_FRIEND(ModelDescriptor)
};

/**
 * @brief The abstract Model class holds the geometry attached to an asset of
 * the scene (mesh, circuit, volume, etc). The model handles resources attached
 * to the geometry such as implementation specific classes, and acceleration
 * structures). Models provide a simple API to manipulate primitives (spheres,
 * cylinders, triangle meshes, etc).
 */
class Model
{
public:
    /**
     * @brief Constructor for Model class
     * @param animationParameters Parameters for animation
     * @param volumeParameters Parameters for volume
     */
    PLATFORM_API Model(AnimationParameters& animationParameters, VolumeParameters& volumeParameters,
                       GeometryParameters& geometryParameters);

    /**
     * @brief Virtual destructor for Model class
     */
    PLATFORM_API virtual ~Model();

    /**
     * @brief Pure virtual function to commit geometry
     */
    PLATFORM_API virtual void commitGeometry() = 0;

    /**
     * @brief Function to commit transfer function
     * @return True if successful, false otherwise
     */
    PLATFORM_API bool commitTransferFunction();

    /**
     * @brief Function to commit simulation data
     * @return True if successful, false otherwise
     */
    PLATFORM_API bool commitSimulationData();

    /**
     * @brief Factory method to create an engine-specific material.
     * @param materialId ID of material
     * @param name Name of material
     * @param properties Properties of material
     * @return Pointer to created material
     */
    PLATFORM_API MaterialPtr createMaterial(const size_t materialId, const std::string& name,
                                            const PropertyMap& properties = {});

    /**
     * @brief Create a volume with the given dimensions, voxel spacing and data type
     * where the voxels are set via setVoxels() from any memory location.
     * @param dimensions Dimensions of volume
     * @param spacing Spacing of volume
     * @param type Data type of volume
     * @return Pointer to created SharedDataVolume
     */
    PLATFORM_API virtual SharedDataVolumePtr createSharedDataVolume(const Vector3ui& dimensions,
                                                                    const Vector3f& spacing, const DataType type) = 0;

    /**
     * @brief Create a volume with the given dimensions, voxel spacing and data type where the voxels are copied via
     * setBrick() into an optimized internal storage.
     * @param dimensions Dimensions of volume
     * @param spacing Spacing of volume
     * @param type Data type of volume
     * @return Pointer to created BrickedVolume
     */
    PLATFORM_API virtual BrickedVolumePtr createBrickedVolume(const Vector3ui& dimensions, const Vector3f& spacing,
                                                              const DataType type) = 0;

    /**
     * @brief Pure virtual function to build bounding box
     */
    PLATFORM_API virtual void buildBoundingBox() = 0;

    /**
     * @return True if the geometry Model does not contain any geometry, false otherwise
     */
    PLATFORM_API bool empty() const;

    /**
     * @return true if the geometry Model is dirty, false otherwise
     */
    PLATFORM_API bool isDirty() const;

    /**
     * @brief Returns the bounds for the Model
     * @return Bounds of Model
     */
    PLATFORM_API const Boxd& getBounds() const { return _bounds; }

    /**
     * @brief Merges model bounds with the given bounds
     * @param bounds Bounds to merge
     */
    void mergeBounds(const Boxd& bounds) { _bounds.merge(bounds); }

    /**
     * @brief Returns spheres handled by the Model
     * @return Map of Spheres
     */
    PLATFORM_API const SpheresMap& getSpheres() const { return _geometries->_spheres; }
    PLATFORM_API SpheresMap& getSpheres()
    {
        _spheresDirty = true;
        return _geometries->_spheres;
    }

    /**
     * @brief Adds a sphere to the model
     * @param materialId ID of material
     * @param sphere Sphere to add
     * @return Index of sphere for the specified material
     */
    PLATFORM_API uint64_t addSphere(const size_t materialId, const Sphere& sphere);

    /**
     * @brief Returns cylinders handled by the model
     * @return Map of Cylinders
     */
    PLATFORM_API const CylindersMap& getCylinders() const { return _geometries->_cylinders; }
    PLATFORM_API CylindersMap& getCylinders()
    {
        _cylindersDirty = true;
        return _geometries->_cylinders;
    }

    /**
     * @brief Adds a cylinder to the model
     * @param materialId ID of material
     * @param cylinder Cylinder to add
     * @return Index of cylinder for the specified material
     */
    PLATFORM_API uint64_t addCylinder(const size_t materialId, const Cylinder& cylinder);

    /**
     * @brief Returns cones handled by the model
     * @return Map of Cones
     */
    PLATFORM_API const ConesMap& getCones() const { return _geometries->_cones; }
    PLATFORM_API ConesMap& getCones()
    {
        _conesDirty = true;
        return _geometries->_cones;
    }

    /**
     * @brief Adds a cone to the model
     * @param materialId ID of material
     * @param cone Cone to add
     * @return Index of cone for the specified material
     */
    PLATFORM_API uint64_t addCone(const size_t materialId, const Cone& cone);

    /**
     * @brief Returns SDFBezier handled by the model
     * @return Map of SDFBeziers
     */
    PLATFORM_API const SDFBeziersMap& getSDFBeziers() const { return _geometries->_sdfBeziers; }

    PLATFORM_API SDFBeziersMap& getSDFBeziers()
    {
        _sdfBeziersDirty = true;
        return _geometries->_sdfBeziers;
    }

    /**
     * @brief Adds an SDFBezier to the model
     * @param materialId ID of material
     * @param sdfBezier SDFBezier to add
     * @return Index of bezier for the specified material
     */
    PLATFORM_API uint64_t addSDFBezier(const size_t materialId, const SDFBezier& sdfBezier);

    /**
     * @brief Adds a streamline to the model
     * @param materialId ID of material
     * @param streamline Streamline to add
     */
    PLATFORM_API void addStreamline(const size_t materialId, const Streamline& streamline);

    /**
     * @brief Returns streamlines handled by the model
     * @return Map of Streamlines
     */
    PLATFORM_API const StreamlinesDataMap& getStreamlines() const { return _geometries->_streamlines; }
    PLATFORM_API StreamlinesDataMap& getStreamlines()
    {
        _streamlinesDirty = true;
        return _geometries->_streamlines;
    }

    /**
     * @brief Adds a curve to the model
     * @param materialId ID of material
     * @param curve Curve to add
     */
    PLATFORM_API void addCurve(const size_t materialId, const Curve& curve);

    /**
     * @brief Returns curves handled by the model
     * @return Map of Curves
     */
    PLATFORM_API const CurvesMap& getCurves() const { return _geometries->_curves; }
    PLATFORM_API CurvesMap& getCurves()
    {
        _curvesDirty = true;
        return _geometries->_curves;
    }

    /**
     * @brief Adds an SDFGeometry to the scene
     * @param materialId ID of material
     * @param geom Geometry to add
     * @param neighbourIndices Global indices of the geometries to smoothly blend together with
     * @return Global index of the geometry
     */
    PLATFORM_API uint64_t addSDFGeometry(const size_t materialId, const SDFGeometry& geom,
                                         const uint64_ts& neighbourIndices);

    /**
     * @brief Returns SDF geometry data handled by the model
     * @return SDF geometry data
     */
    PLATFORM_API const SDFGeometryData& getSDFGeometryData() const { return _geometries->_sdf; }
    PLATFORM_API SDFGeometryData& getSDFGeometryData()
    {
        _sdfGeometriesDirty = true;
        return _geometries->_sdf;
    }

    /**
     * @brief Update the list of neighbours for an SDF geometry
     * @param geometryIdx Index of the geometry
     * @param neighbourIndices Global indices of the geometries to smoothly blend together with
     */
    PLATFORM_API void updateSDFGeometryNeighbours(size_t geometryIdx, const uint64_ts& neighbourIndices);

    /**
     * Returns triangle meshes handled by the model
     * @return Map of TriangleMeshes
     */
    PLATFORM_API const TriangleMeshMap& getTriangleMeshes() const { return _geometries->_triangleMeshes; }
    PLATFORM_API TriangleMeshMap& getTriangleMeshes()
    {
        _triangleMeshesDirty = true;
        return _geometries->_triangleMeshes;
    }

    /**
     * @brief Add a volume to the model
     * @param materialId ID of material
     * @param volume Pointer to volume to add
     */
    PLATFORM_API void addVolume(const size_t materialId, VolumePtr);

    /**
     * @brief Remove a volume from the model
     * @param materialId ID of material
     */
    PLATFORM_API void removeVolume(const size_t materialId);

    /**
     * @brief Logs information about the model, like the number of primitives, and the associated memory footprint.
     */
    PLATFORM_API void logInformation();

    /**
     * @brief Sets the materials handled by the model, and available to the geometry
     * @param colorMap Specifies the algorithm that is used to create the materials
     */
    PLATFORM_API void setMaterialsColorMap(const MaterialsColorMap colorMap);

    /**
     * @brief Returns a reference to the map of materials handled by the model
     * @return const MaterialMap& The map of materials handled by the model
     */
    PLATFORM_API const MaterialMap& getMaterials() const { return _materials; }

    /**
     * @brief Returns a pointer to a specific material
     * @param materialId The ID of the material
     * @return MaterialPtr A pointer to the material
     */
    PLATFORM_API MaterialPtr getMaterial(const size_t materialId) const;

    /**
     * @brief Returns the transfer function used for volumes and simulations
     * @return TransferFunction& The transfer function used for volumes and simulations
     */
    PLATFORM_API TransferFunction& getTransferFunction() { return _transferFunction; }
    /**
     * @brief Returns the transfer function used for volumes and simulations
     * @return const TransferFunction& The transfer function used for volumes and simulations
     */
    PLATFORM_API const TransferFunction& getTransferFunction() const { return _transferFunction; }

    /**
     * @brief Returns the simulation handler
     * @return AbstractSimulationHandlerPtr The simulation handler
     */
    PLATFORM_API AbstractSimulationHandlerPtr getSimulationHandler() const;

    /**
     * @brief Sets the simulation handler
     * @param handler The simulation handler
     */
    PLATFORM_API void setSimulationHandler(AbstractSimulationHandlerPtr handler);

    /**
     * @brief Returns the size in bytes of all geometries
     * @return size_t The size in bytes of all geometries
     */
    PLATFORM_API size_t getSizeInBytes() const;

    /**
     * @brief Marks the instances as dirty
     */
    PLATFORM_API void markInstancesDirty() { _instancesDirty = true; }

    /**
     * @brief Marks the instances as clean
     */
    PLATFORM_API void markInstancesClean() { _instancesDirty = false; }

    /**
     * @brief Returns a const reference to the list of volumes
     * @return const Volumes& The list of volumes
     */
    PLATFORM_API const VolumesMap& getVolumes() const { return _geometries->_volumes; }

    /**
     * @brief Returns whether the volumes are dirty
     * @return bool Whether the volumes are dirty
     */
    PLATFORM_API bool isVolumesDirty() const { return _volumesDirty; }

    /**
     * @brief Resets the dirty status of the volumes
     */
    PLATFORM_API void resetVolumesDirty() { _volumesDirty = false; }

    /**
     * @brief Sets the BVH flags
     * @param bvhFlags The BVH flags to set
     */
    PLATFORM_API void setBVHFlags(std::set<BVHFlag> bvhFlags) { _bvhFlags = std::move(bvhFlags); }

    /**
     * @brief Gets the BVH flags
     * @return const std::set<BVHFlag>& The BVH flags
     */
    PLATFORM_API const std::set<BVHFlag>& getBVHFlags() const { return _bvhFlags; }

    /**
     * @brief Updates the bounds of the geometries
     */
    PLATFORM_API void updateBounds();

    /**
     * @brief Copies the model data from another model
     * @param rhs The model to copy the data from
     */
    PLATFORM_API void copyFrom(const Model& rhs);

    /**
     * @brief Applies a default color map (rainbow) to the model
     */
    PLATFORM_API void applyDefaultColormap();

protected:
    void _updateSizeInBytes();

    /** Factory method to create an engine-specific material. */
    PLATFORM_API virtual MaterialPtr createMaterialImpl(const PropertyMap& properties = {}) = 0;

    /** Mark all geometries as clean. */
    void _markGeometriesClean();

    virtual void _commitTransferFunctionImpl(const Vector3fs& colors, const floats& opacities,
                                             const Vector2d valueRange) = 0;
    virtual void _commitSimulationDataImpl(const float* frameData, const size_t frameSize) = 0;

    AnimationParameters& _animationParameters;
    VolumeParameters& _volumeParameters;
    GeometryParameters& _geometryParameters;

    AbstractSimulationHandlerPtr _simulationHandler;
    TransferFunction _transferFunction;

    MaterialMap _materials;

    struct Geometries
    {
        SpheresMap _spheres;
        CylindersMap _cylinders;
        ConesMap _cones;
        SDFBeziersMap _sdfBeziers;
        TriangleMeshMap _triangleMeshes;
        StreamlinesDataMap _streamlines;
        SDFGeometryData _sdf;
        VolumesMap _volumes;
        CurvesMap _curves;

        Boxd _sphereBounds;
        Boxd _cylindersBounds;
        Boxd _conesBounds;
        Boxd _sdfBeziersBounds;
        Boxd _triangleMeshesBounds;
        Boxd _streamlinesBounds;
        Boxd _sdfGeometriesBounds;
        Boxd _volumesBounds;
        Boxd _curvesBounds;

        bool isEmpty() const
        {
            return _spheres.empty() && _cylinders.empty() && _cones.empty() && _sdfBeziers.empty() &&
                   _triangleMeshes.empty() && _sdf.geometries.empty() && _streamlines.empty() && _volumes.empty() &&
                   _curves.empty();
        }
    };

    // the model clone actually shares all geometries to save memory. It will
    // still create engine specific copies though (BVH only ideally) as part of
    // commitGeometry()
    std::shared_ptr<Geometries> _geometries{std::make_shared<Geometries>()};

    bool _spheresDirty{false};
    bool _cylindersDirty{false};
    bool _conesDirty{false};
    bool _sdfBeziersDirty{false};
    bool _triangleMeshesDirty{false};
    bool _streamlinesDirty{false};
    bool _sdfGeometriesDirty{false};
    bool _volumesDirty{false};
    bool _curvesDirty{false};

    bool _areGeometriesDirty() const
    {
        return _spheresDirty || _cylindersDirty || _conesDirty || _sdfBeziersDirty || _triangleMeshesDirty ||
               _sdfGeometriesDirty || _curvesDirty;
    }

    Boxd _bounds;
    bool _instancesDirty{true};
    std::set<BVHFlag> _bvhFlags;
    size_t _sizeInBytes{0};

    // Whether this model has set the AnimationParameters "is ready" callback
    bool _isReadyCallbackSet{false};

    SERIALIZATION_FRIEND(Model)
};
} // namespace core
