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

#include "AbstractParameters.h"

SERIALIZATION_ACCESS(VolumeParameters)

namespace core
{
class VolumeParameters final : public AbstractParameters
{
public:
    VolumeParameters();

    /** @copydoc AbstractParameters::print */
    void print() final;

    /** Volume dimensions  */
    const Vector3ui& getDimensions() const { return _dimensions; }
    void setDimensions(const Vector3ui& dim) { _updateValue(_dimensions, dim); }

    /** Volume scale  */
    const Vector3d& getElementSpacing() const { return _elementSpacing; }
    void setElementSpacing(const Vector3d& spacing) { _updateValue(_elementSpacing, spacing); }

    /** Volume offset */
    const Vector3d& getOffset() const { return _offset; }

    void setGradientShading(const bool enabled) { _updateValue(_gradientShading, enabled); }
    bool getGradientShading() const { return _gradientShading; }

    void setGradientOffset(const double value) { _updateValue(_gradientOffset, value); }
    double getGradientOffset() const { return _gradientOffset; }

    void setSingleShade(const bool enabled) { _updateValue(_singleShade, enabled); }
    bool getSingleShade() const { return _singleShade; }

    void setPreIntegration(const bool enabled) { _updateValue(_preIntegration, enabled); }
    bool getPreIntegration() const { return _preIntegration; }

    void setAdaptiveSampling(const bool enabled) { _updateValue(_adaptiveSampling, enabled); }
    bool getAdaptiveSampling() const { return _adaptiveSampling; }

    void setAdaptiveMaxSamplingRate(const double value) { _updateValue(_adaptiveMaxSamplingRate, value); }
    double getAdaptiveMaxSamplingRate() const { return _adaptiveMaxSamplingRate; }

    void setSamplingRate(const double value) { _updateValue(_samplingRate, value); }
    double getSamplingRate() const { return _samplingRate; }

    void setSpecular(const Vector3d& value) { _updateValue(_specular, value); }
    const Vector3d& getSpecular() const { return _specular; }

    void setUserParameters(const Vector3d& value) { _updateValue(_userParameters, value); }
    const Vector3d& getUserParameters() const { return _userParameters; }

    void setClipBox(const Boxd& value) { _updateValue(_clipBox, value); }
    const Boxd& getClipBox() const { return _clipBox; }

protected:
    void parse(const po::variables_map& vm) final;

    Vector3ui _dimensions;
    Vector3d _elementSpacing;
    Vector3d _offset;

    bool _gradientShading{false};
    double _gradientOffset{0.001};
    bool _singleShade{true};
    bool _preIntegration{false};
    double _adaptiveMaxSamplingRate{2.};
    bool _adaptiveSampling{true};
    double _samplingRate{0.125};
    Vector3d _specular{0.3, 0.3, 0.3};
    Boxd _clipBox;
    Vector3d _userParameters;

    SERIALIZATION_FRIEND(VolumeParameters)
};
} // namespace core
