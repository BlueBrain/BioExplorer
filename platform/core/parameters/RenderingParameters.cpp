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

#include "RenderingParameters.h"
#include <platform/core/common/Logs.h>

namespace
{
const std::string PARAM_ACCUMULATION = "disable-accumulation";
const std::string PARAM_BACKGROUND_COLOR = "background-color";
const std::string PARAM_CAMERA = "camera";
const std::string PARAM_RENDERER = "renderer";
const std::string PARAM_VARIANCE_THRESHOLD = "variance-threshold";
const std::string PARAM_NUM_NON_DENOISED_FRAMES = "num-non-denoised-frames";
const std::string PARAM_DENOISE_BLEND = "denoise-blend";
const std::string PARAM_TONE_MAPPER_EXPOSURE = "tone-mapper-exposure";
const std::string PARAM_TONE_MAPPER_GAMMA = "tone-mapper-gamma";

const std::string ACCUMULATION_TYPES[3] = {"none", "linear", "ai-denoised"};
} // namespace

namespace core
{
RenderingParameters::RenderingParameters()
    : AbstractParameters("Rendering")
{
    _parameters.add_options() //
        (PARAM_RENDERER.c_str(), po::value<std::string>(),
         "The renderer to use") //
        (PARAM_ACCUMULATION.c_str(), po::bool_switch()->default_value(false),
         "Disable accumulation") //
        (PARAM_CAMERA.c_str(), po::value<std::string>(),
         "The camera to use") //
        (PARAM_VARIANCE_THRESHOLD.c_str(), po::value<double>(&_varianceThreshold),
         "Threshold for adaptive accumulation [float]")                                      //
        (PARAM_NUM_NON_DENOISED_FRAMES.c_str(), po::value<uint32_t>(&_numNonDenoisedFrames), //
         "Optix 6 only: Number of frames that show the original image before switching on denoising"),
        (PARAM_DENOISE_BLEND.c_str(), po::value<float>(&_denoiseBlend), //
         "Optix 6 only: Amount of the original image that is blended with the denoised result ranging from 0.0 to 1.0"),
        (PARAM_TONE_MAPPER_EXPOSURE.c_str(), po::value<float>(&_toneMapperExposure), //
         "Optix 6 only: Tone mapper exposure"),
        (PARAM_TONE_MAPPER_GAMMA.c_str(), po::value<float>(&_toneMapperGamma), //
         "Optix 6 only: Tone mapper gamma");
}

void RenderingParameters::parse(const po::variables_map& vm)
{
    if (vm.count(PARAM_RENDERER))
    {
        const std::string& rendererName = vm[PARAM_RENDERER].as<std::string>();
        addRenderer(rendererName);
    }
    if (vm.count(PARAM_CAMERA))
    {
        const std::string& cameraName = vm[PARAM_CAMERA].as<std::string>();
        _camera = cameraName;
        if (std::find(_cameras.begin(), _cameras.end(), cameraName) == _cameras.end())
            _cameras.push_front(cameraName);
    }
    markModified();
}

const std::string RenderingParameters::getAccumulationTypeAsString(const AccumulationType value)
{
    return ACCUMULATION_TYPES[static_cast<size_t>(value)];
}

void RenderingParameters::print()
{
    AbstractParameters::print();
    CORE_INFO("Supported renderers               : ");
    for (const auto& renderer : _renderers)
        CORE_INFO("- " << renderer);
    CORE_INFO("Camera                            : " << _camera);
    CORE_INFO("Accumulation type                 : " << getAccumulationTypeAsString(_accumulationType));
    CORE_INFO("Number of non-denoised frames     : " << _numNonDenoisedFrames);
    CORE_INFO("Denoise blend                     : " << _denoiseBlend);
    CORE_INFO("Tone mapper exposure              : " << _toneMapperExposure);
    CORE_INFO("Tone mapper gamma                 : " << _toneMapperGamma);
}
} // namespace core
