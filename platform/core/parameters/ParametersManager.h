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
#include <platform/core/common/Types.h>
#include <platform/core/parameters/AnimationParameters.h>
#include <platform/core/parameters/ApplicationParameters.h>
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
};
} // namespace core
