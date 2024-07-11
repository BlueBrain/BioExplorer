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

SERIALIZATION_ACCESS(FieldParameters)

namespace core
{
class FieldParameters final : public AbstractParameters
{
public:
    FieldParameters();

    /** @copydoc AbstractParameters::print */
    void print() final;

    void setGradientShading(const bool enabled) { _updateValue(_gradientShading, enabled); }
    bool getGradientShading() const { return _gradientShading; }

    void setGradientOffset(const double value) { _updateValue(_gradientOffset, value); }
    double getGradientOffset() const { return _gradientOffset; }

    void setSamplingRate(const double value) { _updateValue(_samplingRate, value); }
    double getSamplingRate() const { return _samplingRate; }

    void setDistance(const double value) { _updateValue(_distance, value); }
    double getDistance() const { return _distance; }

    void setCutoff(const double value) { _updateValue(_cutoff, value); }
    double getCutoff() const { return _cutoff; }

    void setEpsilon(const double value) { _updateValue(_epsilon, value); }
    double getEpsilon() const { return _epsilon; }

    void setAccumulationSteps(const uint64_t value) { _updateValue(_randomAccumulation, value); }
    uint64_t getAccumulationSteps() const { return _randomAccumulation; }

    void setUseOctree(const bool value) { _updateValue(_useOctree, value); }
    bool getUseOctree() const { return _useOctree; }

protected:
    void parse(const po::variables_map& vm) final;

    Vector3ui _dimensions;
    Vector3d _elementSpacing;
    Vector3d _offset;

    bool _gradientShading{false};
    double _gradientOffset{0.001};
    double _samplingRate{0.125};
    double _distance{1.f};
    double _cutoff{1500.f};
    double _epsilon{1e-6};
    uint64_t _randomAccumulation{0};
    bool _useOctree{true};

    SERIALIZATION_FRIEND(FieldParameters)
};
} // namespace core
