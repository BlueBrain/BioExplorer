/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
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
