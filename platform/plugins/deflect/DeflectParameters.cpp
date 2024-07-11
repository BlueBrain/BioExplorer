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

#include "DeflectParameters.h"
#include "utils.h"

#include <deflect/Stream.h>

namespace core
{
DeflectParameters::DeflectParameters()
    : _props("Deflect plugin parameters")
{
    // clang-format off
    _props.setProperty(
        {PARAM_ENABLED, true,
         Property::MetaData{"Enable streaming", "Enable/disable streaming"}});
    _props.setProperty(
        {PARAM_ID, std::string(),
         {"Stream ID", "The ID/name of the stream, equivalent to DEFLECT_ID"}});
    _props.setProperty({PARAM_HOSTNAME, std::string(),
                        {"Stream hostname", "Hostname of Deflect server"}});
    _props.setProperty({PARAM_PORT, (int32_t)deflect::Stream::defaultPortNumber,
                        1, 65535, {"Stream port", "Port of Deflect server"}});
    _props.setProperty({PARAM_COMPRESSION, true,
                        {"Use JPEG compression", "Use JPEG compression"}});
    _props.setProperty({PARAM_TOP_DOWN, false,
                        {"Stream image top-down",
                         "Top-down image orientation instead of bottom-up"}});
    _props.setProperty(
        {PARAM_RESIZING, true,
         {"Allow resizing",
          "Allow resizing of framebuffers from EVT_VIEW_SIZE_CHANGED"}});
    _props.setProperty(
        {PARAM_QUALITY, (int32_t)80, 1, 100, {"JPEG quality", "JPEG quality"}});
    _props.setProperty(
        {PARAM_USE_PIXEL_OP, false,
         {"Use per-tile direct streaming", "Use per-tile direct streaming"}});
    _props.setProperty({PARAM_CHROMA_SUBSAMPLING,
                        int32_t(deflect::ChromaSubsampling::YUV444),
                        enumNames<deflect::ChromaSubsampling>(),
                        {"Chroma subsampling",
                         "Chroma subsampling modes: yuv444, yuv422, yuv420"}});
    // clang-format on
}
} // namespace core
