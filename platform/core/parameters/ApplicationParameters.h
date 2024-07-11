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

#include <platform/core/common/Types.h>
#include <string>
#include <vector>

SERIALIZATION_ACCESS(ApplicationParameters)

namespace core
{
/** Manages application parameters
 */
class ApplicationParameters : public AbstractParameters
{
public:
    ApplicationParameters();

    /** @copydoc AbstractParameters::print */
    void print() final;

    /** Engine*/
    const std::string& getEngine() const { return _engine; }
    /** OSPRay modules */
    const std::vector<std::string>& getOsprayModules() const { return _modules; }

    bool getDynamicLoadBalancer() const { return _dynamicLoadBalancer; }
    void setDynamicLoadBalancer(const bool value) { _updateValue(_dynamicLoadBalancer, value); }

    /** window size */
    const Vector2ui getWindowSize() const { return Vector2ui(_windowSize); }
    void setWindowSize(const Vector2ui& size)
    {
        Vector2d value(size);
        _updateValue(_windowSize, value);
    }
    /** Benchmarking */
    bool isBenchmarking() const { return _benchmarking; }
    void setBenchmarking(bool enabled) { _benchmarking = enabled; }
    /** JPEG compression quality */
    void setJpegCompression(const size_t compression) { _updateValue(_jpegCompression, compression); }
    size_t getJpegCompression() const { return _jpegCompression; }
    /** Image stream FPS */
    size_t getImageStreamFPS() const { return _imageStreamFPS; }
    void setImageStreamFPS(const size_t fps) { _updateValue(_imageStreamFPS, fps); }

    bool useVideoStreaming() const { return _useVideoStreaming; }
    /** Max render FPS to limit */
    size_t getMaxRenderFPS() const { return _maxRenderFPS; }
    bool isStereo() const { return _stereo; }
    bool getParallelRendering() const { return _parallelRendering; }
    const std::string& getHttpServerURI() const { return _httpServerURI; }
    void setHttpServerURI(const std::string& httpServerURI) { _updateValue(_httpServerURI, httpServerURI); }

    const std::string& getEnvMap() const { return _envMap; }
    const std::string& getSandboxPath() const { return _sandBoxPath; }
    const strings& getInputPaths() const { return _inputPaths; }
    po::positional_options_description& posArgs() { return _positionalArgs; }

protected:
    void parse(const po::variables_map& vm) final;

    std::string _engine{ENGINE_OSPRAY};
    std::vector<std::string> _modules;
    Vector2d _windowSize;
    bool _benchmarking{false};
    size_t _jpegCompression;
    bool _stereo{false};
    size_t _imageStreamFPS{60};
    size_t _maxRenderFPS{std::numeric_limits<size_t>::max()};
    std::string _httpServerURI;
    bool _parallelRendering{false};
    bool _dynamicLoadBalancer{false};
    bool _useVideoStreaming{false};
    std::string _envMap;
    std::string _sandBoxPath;

    strings _inputPaths;

    po::positional_options_description _positionalArgs;

    SERIALIZATION_FRIEND(ApplicationParameters)
};
} // namespace core
