/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include <platform/core/common/Api.h>
#include <platform/core/common/Types.h>
#include <platform/core/parameters/AnimationParameters.h>
#include <platform/core/parameters/ApplicationParameters.h>
#include <platform/core/parameters/FieldParameters.h>
#include <platform/core/parameters/GeometryParameters.h>
#include <platform/core/parameters/RenderingParameters.h>
#include <platform/core/parameters/VolumeParameters.h>

namespace core
{
/**
   Class managing all parameters registered by the application. By default
   this class create an instance of Application, Rendering, and Geometry
   parameters are registered. Other parameters can also be added using the
   registerParameters method for as long as they are inherited from
   AbstractParameters.
 */
class ParametersManager
{
public:
    ParametersManager(int argc, const char** argv);

    /**
       Registers specific parameters to the manager
       @param parameters to be registered
     */
    PLATFORM_API void registerParameters(AbstractParameters* parameters);

    /**
       Displays usage of registered parameters
     */
    PLATFORM_API void usage();

    /**
       Displays values registered parameters
     */
    PLATFORM_API void print();

    /**
       Gets animation parameters
       @return Animation parameters for the current scene
    */
    PLATFORM_API AnimationParameters& getAnimationParameters();
    PLATFORM_API const AnimationParameters& getAnimationParameters() const;

    /**
       Gets rendering parameters
       @return Rendering parameters for the current scene
    */
    PLATFORM_API RenderingParameters& getRenderingParameters();
    PLATFORM_API const RenderingParameters& getRenderingParameters() const;

    /**
       Gets geometry parameters
       @return Geometry parameters for the current scene
    */
    PLATFORM_API GeometryParameters& getGeometryParameters();
    PLATFORM_API const GeometryParameters& getGeometryParameters() const;

    /**
       Gets application parameters
       @return Application parameters for the current scene
    */
    PLATFORM_API ApplicationParameters& getApplicationParameters();
    PLATFORM_API const ApplicationParameters& getApplicationParameters() const;

    /**
       Gets volume parameters
       @return Parameters for the current volume
    */
    PLATFORM_API VolumeParameters& getVolumeParameters();
    PLATFORM_API const VolumeParameters& getVolumeParameters() const;

    /**
       Gets field parameters
       @return Parameters for the current field
    */
    PLATFORM_API FieldParameters& getFieldParameters();
    PLATFORM_API const FieldParameters& getFieldParameters() const;

    /** Call resetModified on all parameters. */
    void resetModified();

    /**
     * @return true if any of the parameters has been modified since the last
     * resetModified().
     */
    bool isAnyModified() const;

private:
    void _parse(int argc, const char** argv);
    void _processUnrecognizedOptions(const std::vector<std::string>& unrecognizedOptions) const;

    po::options_description _allOptions;

    std::vector<AbstractParameters*> _parameterSets;
    AnimationParameters _animationParameters;
    ApplicationParameters _applicationParameters;
    GeometryParameters _geometryParameters;
    RenderingParameters _renderingParameters;
    VolumeParameters _volumeParameters;
    FieldParameters _fieldParameters;
};
} // namespace core
