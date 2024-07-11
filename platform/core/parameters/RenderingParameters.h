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

#include <platform/core/common/Types.h>

#include "AbstractParameters.h"
#include <deque>

SERIALIZATION_ACCESS(RenderingParameters)

namespace core
{
class AbstractParameters;

/** Manages rendering parameters
 */
class RenderingParameters : public AbstractParameters
{
public:
    RenderingParameters();

    /** @copydoc AbstractParameters::print */
    void print() final;

    /** All registered renderers */
    const auto& getRenderers() const { return _renderers; }
    void addRenderer(const std::string& renderer)
    {
        if (std::find(_renderers.begin(), _renderers.end(), renderer) == _renderers.end())
            _renderers.push_front(renderer);
    }
    const std::string& getCurrentCamera() const { return _camera; }

    /** All registered cameras */
    const auto& getCameras() const { return _cameras; }
    void addCamera(const std::string& camera) { _cameras.push_front(camera); }
    /**
     * @return the threshold where accumulation stops if the variance error
     * reaches this value.
     */
    double getVarianceThreshold() const { return _varianceThreshold; }
    /**
     * The threshold where accumulation stops if the variance error reaches this
     * value.
     */
    void setVarianceThreshold(const double value) { _updateValue(_varianceThreshold, value); }

    /** If the rendering should be refined by accumulating multiple passes */
    AccumulationType getAccumulationType() const { return _accumulationType; }
    const std::string getAccumulationTypeAsString(const AccumulationType value);

    /**
     *  Denoising parameters: Used by the Optix 6 engine only
     */
    /** Number of frames that show the original image before switching on denoising */
    void setNumNonDenoisedFrames(const uint32_t value) { _updateValue(_numNonDenoisedFrames, value); }
    uint32_t getNumNonDenoisedFrames() const { return _numNonDenoisedFrames; }

    /* Amount of the original image that is blended with the denoised result ranging from 0.0 to 1.0 */
    void setDenoiseBlend(const float value) { _updateValue(_denoiseBlend, value); }
    float getDenoiseBlend() const { return _denoiseBlend; }

    /* Tone mapper exposure */
    void setToneMapperExposure(const float value) { _updateValue(_toneMapperExposure, value); }
    float getToneMapperExposure() const { return _toneMapperExposure; }

    /* Tone mapper gamma */
    void setToneMapperGamma(const float value) { _updateValue(_toneMapperGamma, value); }
    float getToneMapperGamma() const { return _toneMapperGamma; }

protected:
    void parse(const po::variables_map& vm) final;

    std::deque<std::string> _renderers;
    std::string _camera{"perspective"};
    std::deque<std::string> _cameras;
    double _varianceThreshold{-1.};
    uint32_t _numNonDenoisedFrames{2};
    float _denoiseBlend{0.1f};
    float _toneMapperExposure{1.5f};
    float _toneMapperGamma{1.f};
    AccumulationType _accumulationType{AccumulationType::linear};

    SERIALIZATION_FRIEND(RenderingParameters)
};
} // namespace core
