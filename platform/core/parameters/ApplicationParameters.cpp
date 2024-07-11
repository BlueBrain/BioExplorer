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

#include "ApplicationParameters.h"
#include <platform/core/common/Logs.h>
#include <platform/core/common/Properties.h>
#include <platform/core/common/Types.h>
#include <platform/core/parameters/ParametersManager.h>

namespace
{
const std::string PARAM_BENCHMARKING = "enable-benchmark";
const std::string PARAM_ENGINE = "engine";
const std::string PARAM_HTTP_SERVER = "http-server";
const std::string PARAM_IMAGE_STREAM_FPS = "image-stream-fps";
const std::string PARAM_INPUT_PATHS = "input-paths";
const std::string PARAM_JPEG_COMPRESSION = "jpeg-compression";
const std::string PARAM_MAX_RENDER_FPS = "max-render-fps";
const std::string PARAM_MODULE = "module";
const std::string PARAM_PARALLEL_RENDERING = "parallel-rendering";
const std::string PARAM_PLUGIN = "plugin";
const std::string PARAM_WINDOW_SIZE = "window-size";
const std::string PARAM_ENV_MAP = "env-map";
const std::string PARAM_SANDBOX_PATH = "sandbox-path";
#ifdef BRAYNS_USE_FFMPEG
const std::string PARAM_VIDEOSTREAMING = "videostreaming";
#endif

const size_t DEFAULT_WINDOW_WIDTH = 800;
const size_t DEFAULT_WINDOW_HEIGHT = 600;
const size_t DEFAULT_JPEG_COMPRESSION = 90;
const std::string DEFAULT_SANDBOX_PATH = "/gpfs/bbp.cscs.ch/project";
} // namespace

namespace core
{
ApplicationParameters::ApplicationParameters()
    : AbstractParameters("Application")
    , _windowSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    , _jpegCompression(DEFAULT_JPEG_COMPRESSION)
    , _sandBoxPath(DEFAULT_SANDBOX_PATH)
{
    _parameters.add_options() //
        (PARAM_ENGINE.c_str(), po::value<std::string>(&_engine),
         "Engine name [ospray|optix]") //
        (PARAM_MODULE.c_str(), po::value<strings>(&_modules)->composing(),
         "OSPRay module name [string]") //
        (PARAM_HTTP_SERVER.c_str(), po::value<std::string>(&_httpServerURI),
         "HTTP interface") //
        (PARAM_INPUT_PATHS.c_str(), po::value<strings>(&_inputPaths),
         "List of files/folders to load data from") //
        (PARAM_PLUGIN.c_str(), po::value<strings>()->composing(),
         "Dynamic plugin to load from LD_LIBRARY_PATH; "
         "can be repeated to load multiple plugins. "
         "Arguments to plugins can be added by inserting a space followed by "
         "the arguments like: --plugin 'myPluginName arg0 arg1'") //
        (PARAM_WINDOW_SIZE.c_str(), po::fixed_tokens_value<uints>(2, 2),
         "Window size [uint uint]") //
        (PARAM_BENCHMARKING.c_str(), po::bool_switch(&_benchmarking)->default_value(false),
         "Enable benchmarking") //
        (PARAM_JPEG_COMPRESSION.c_str(), po::value<size_t>(&_jpegCompression),
         "JPEG compression rate (100 is full quality) [int]") //
        (PARAM_PARALLEL_RENDERING.c_str(), po::bool_switch(&_parallelRendering)->default_value(false),
         "Enable parallel rendering, equivalent to --osp:mpi") //
        (CAMERA_PROPERTY_STEREO.name.c_str(), po::bool_switch(&_stereo)->default_value(DEFAULT_CAMERA_STEREO),
         "Enable stereo rendering") //
        (PARAM_IMAGE_STREAM_FPS.c_str(), po::value<size_t>(&_imageStreamFPS),
         "Image stream FPS (60 default), [int]") //
        (PARAM_MAX_RENDER_FPS.c_str(), po::value<size_t>(&_maxRenderFPS),
         "Max. render FPS") //
        (PARAM_ENV_MAP.c_str(), po::value<std::string>(&_envMap),
         "Path to environment map")(PARAM_SANDBOX_PATH.c_str(), po::value<std::string>(&_sandBoxPath),
                                    "Path to sandbox directory")
#ifdef BRAYNS_USE_FFMPEG
            (PARAM_VIDEOSTREAMING.c_str(), po::bool_switch(&_useVideoStreaming)->default_value(false),
             "Use videostreaming over websockets instead of JPEG")
#endif
        ;

    _positionalArgs.add(PARAM_INPUT_PATHS.c_str(), -1);
}

void ApplicationParameters::parse(const po::variables_map& vm)
{
    if (vm.count(PARAM_WINDOW_SIZE))
    {
        uints values = vm[PARAM_WINDOW_SIZE].as<uints>();
        _windowSize.x = values[0];
        _windowSize.y = values[1];
    }
    markModified();
}

void ApplicationParameters::print()
{
    AbstractParameters::print();
    CORE_INFO("Engine                      : " << _engine);
    CORE_INFO("Ospray modules              : ");
    for (const auto& module : _modules)
        CORE_INFO("- " << module);
    CORE_INFO("Window size                 : " << _windowSize);
    CORE_INFO("Benchmarking                : " << asString(_benchmarking));
    CORE_INFO("JPEG Compression            : " << _jpegCompression);
    CORE_INFO("Image stream FPS            : " << _imageStreamFPS);
    CORE_INFO("Max. render  FPS            : " << _maxRenderFPS);
    CORE_INFO("Sandbox directory           : " << _sandBoxPath);
}
} // namespace core
