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

#include "OptiXFrameBuffer.h"
#include "OptiXContext.h"
#include "OptiXTypes.h"
#include "OptiXUtils.h"

#include <platform/core/common/Logs.h>
#include <platform/core/parameters/RenderingParameters.h>

#include <optixu/optixu_math_namespace.h>

namespace core
{
namespace engine
{
namespace optix
{
OptiXFrameBuffer::OptiXFrameBuffer(const std::string& name, const Vector2ui& frameSize,
                                   FrameBufferFormat frameBufferFormat, const RenderingParameters& renderingParameters)
    : FrameBuffer(name, frameSize, frameBufferFormat)
    , _renderingParameters(renderingParameters)
{
    resize(frameSize);
}

OptiXFrameBuffer::~OptiXFrameBuffer()
{
    auto lock = getScopeLock();
    _unmapUnsafe();
    _cleanup();
}

void OptiXFrameBuffer::_cleanup()
{
    // Buffers
    RT_DESTROY(_outputBuffer);
    RT_DESTROY(_accumBuffer);
    RT_DESTROY(_tonemappedBuffer);
    RT_DESTROY(_denoisedBuffer);

    // Post processing
    RT_DESTROY(_denoiserStage);
    RT_DESTROY(_tonemapStage);
    RT_DESTROY(_denoiserWithMappingStage);
    RT_DESTROY(_commandListWithDenoiser);
    RT_DESTROY(_commandListWithDenoiserAndToneMapper);
    _postprocessingStagesInitialized = false;
}

void OptiXFrameBuffer::resize(const Vector2ui& frameSize)
{
    if (_outputBuffer && frameSize == _frameSize)
        return;

    if (glm::compMul(frameSize) == 0)
        throw std::runtime_error("Invalid size for frame buffer resize");

    _frameSize = frameSize;
    _cleanup();

    RTformat format;
    switch (_frameBufferFormat)
    {
    case FrameBufferFormat::rgb_i8:
        format = RT_FORMAT_UNSIGNED_BYTE3;
        break;
    case FrameBufferFormat::rgba_i8:
    case FrameBufferFormat::bgra_i8:
        format = RT_FORMAT_UNSIGNED_BYTE4;
        break;
    case FrameBufferFormat::rgb_f32:
        format = RT_FORMAT_FLOAT4;
        break;
    default:
        format = RT_FORMAT_UNKNOWN;
    }

    auto context = OptiXContext::get().getOptixContext();
    _outputBuffer = context->createBuffer(RT_BUFFER_OUTPUT, format, _frameSize.x, _frameSize.y);
    context[CONTEXT_OUTPUT_BUFFER]->set(_outputBuffer);

    _accumBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, _frameSize.x, _frameSize.y);
    context[CONTEXT_ACCUMULATION_BUFFER]->set(_accumBuffer);

    _tonemappedBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, _frameSize.x, _frameSize.y);
    context[CONTEXT_TONEMAPPED_BUFFER]->set(_tonemappedBuffer);

    _denoisedBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, _frameSize.x, _frameSize.y);
    context[CONTEXT_DENOISED_BUFFER]->set(_denoisedBuffer);
    clear();
}

void OptiXFrameBuffer::map()
{
    _mapMutex.lock();
    _mapUnsafe();
}

void OptiXFrameBuffer::_mapUnsafe()
{
    // Initialization
    if (!_postprocessingStagesInitialized)
        _initializePostProcessingStages();

    rtBufferMap(_outputBuffer->get(), &_colorData);

    auto context = OptiXContext::get().getOptixContext();
    context[CONTEXT_FRAME_NUMBER]->setUint(_accumulationFrameNumber);

    _colorBuffer = (uint8_t*)(_colorData);

    // Post processing
    if (_accumulationType == AccumulationType::ai_denoised)
    {
        const bool useDenoiser = (_accumulationFrameNumber >= _renderingParameters.getNumNonDenoisedFrames());

        if (useDenoiser)
            rtBufferMap(_denoisedBuffer->get(), &_floatData);
        else
            rtBufferMap(_tonemappedBuffer->get(), &_floatData);

        _floatBuffer = (float*)_floatData;
    }
    ++_accumulationFrameNumber;
}

void OptiXFrameBuffer::unmap()
{
    _unmapUnsafe();
    _mapMutex.unlock();
}

void OptiXFrameBuffer::_unmapUnsafe()
{
    // Post processing stages
    const bool useDenoiser = (_accumulationFrameNumber >= _renderingParameters.getNumNonDenoisedFrames());
    if (_accumulationType == AccumulationType::ai_denoised)
    {
        if (_commandListWithDenoiser && _commandListWithDenoiserAndToneMapper)
        {
            ::optix::Variable(_denoiserStage->queryVariable(CONTEXT_DENOISE_BLEND))
                ->setFloat(_renderingParameters.getDenoiseBlend());
            if (useDenoiser)
            {
                if (_renderingParameters.getToneMapperExposure() > 0.f)
                    _commandListWithDenoiserAndToneMapper->execute();
                else
                    _commandListWithDenoiser->execute();
            }
        }

        if (useDenoiser)
            rtBufferUnmap(_denoisedBuffer->get());
        rtBufferUnmap(_tonemappedBuffer->get());
    }

    rtBufferUnmap(_outputBuffer->get());

    if (_postprocessingStagesInitialized)
    {
        auto context = OptiXContext::get().getOptixContext();
        context[CONTEXT_TONE_MAPPER_EXPOSURE]->setFloat(_renderingParameters.getToneMapperExposure());
        context[CONTEXT_TONE_MAPPER_GAMMA]->setFloat(_renderingParameters.getToneMapperGamma());
        context[CONTEXT_DENOISE_BLEND]->setFloat(_renderingParameters.getDenoiseBlend());
    }
}

void OptiXFrameBuffer::setAccumulation(const bool accumulation)
{
    if (_accumulation != accumulation)
    {
        FrameBuffer::setAccumulation(accumulation);
        resize(_frameSize);
    }
}

void OptiXFrameBuffer::_initializePostProcessingStages()
{
    auto context = OptiXContext::get().getOptixContext();

    _tonemapStage = context->createBuiltinPostProcessingStage(CONTEXT_STAGE_TONE_MAPPER);
    _tonemapStage->declareVariable(CONTEXT_INPUT_BUFFER)->set(_accumBuffer);
    _tonemapStage->declareVariable(CONTEXT_OUTPUT_BUFFER)->set(_tonemappedBuffer);
    _tonemapStage->declareVariable(CONTEXT_TONE_MAPPER_EXPOSURE)
        ->setFloat(_renderingParameters.getToneMapperExposure());
    _tonemapStage->declareVariable(CONTEXT_TONE_MAPPER_GAMMA)->setFloat(_renderingParameters.getToneMapperGamma());

    _denoiserStage = context->createBuiltinPostProcessingStage(CONTEXT_STAGE_DENOISER);
    _denoiserStage->declareVariable(CONTEXT_INPUT_BUFFER)->set(_accumBuffer);
    _denoiserStage->declareVariable(CONTEXT_OUTPUT_BUFFER)->set(_denoisedBuffer);
    _denoiserStage->declareVariable(CONTEXT_DENOISE_BLEND)->setFloat(_renderingParameters.getDenoiseBlend());
    _denoiserStage->declareVariable(CONTEXT_INPUT_ALBEDO_BUFFER);
    _denoiserStage->declareVariable(CONTEXT_INPUT_NORMAL_BUFFER);

    _denoiserWithMappingStage = context->createBuiltinPostProcessingStage(CONTEXT_STAGE_DENOISER);
    _denoiserWithMappingStage->declareVariable(CONTEXT_INPUT_BUFFER)->set(_tonemappedBuffer);
    _denoiserWithMappingStage->declareVariable(CONTEXT_OUTPUT_BUFFER)->set(_denoisedBuffer);
    _denoiserWithMappingStage->declareVariable(CONTEXT_DENOISE_BLEND)->setFloat(_renderingParameters.getDenoiseBlend());
    _denoiserWithMappingStage->declareVariable(CONTEXT_INPUT_ALBEDO_BUFFER);
    _denoiserWithMappingStage->declareVariable(CONTEXT_INPUT_NORMAL_BUFFER);

    // With denoiser
    _commandListWithDenoiser = context->createCommandList();
    _commandListWithDenoiser->appendLaunch(0, _frameSize.x, _frameSize.y);
    _commandListWithDenoiser->appendPostprocessingStage(_denoiserStage, _frameSize.x, _frameSize.y);
    _commandListWithDenoiser->finalize();

    // With denoiser and tone mapper
    _commandListWithDenoiserAndToneMapper = context->createCommandList();
    _commandListWithDenoiserAndToneMapper->appendLaunch(0, _frameSize.x, _frameSize.y);
    _commandListWithDenoiserAndToneMapper->appendPostprocessingStage(_tonemapStage, _frameSize.x, _frameSize.y);
    _commandListWithDenoiserAndToneMapper->appendPostprocessingStage(_denoiserWithMappingStage, _frameSize.x,
                                                                     _frameSize.y);
    _commandListWithDenoiserAndToneMapper->finalize();

    _postprocessingStagesInitialized = true;
}
} // namespace optix
} // namespace engine
} // namespace core
