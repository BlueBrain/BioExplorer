/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include <platform/core/common/PropertyMap.h>

using namespace core;

namespace medicalimagingexplorer
{
namespace dicom
{
static constexpr double DICOM_DEFAULT_RENDERER_SURFACE_OFFSET = 1.0;

static const Property DICOM_RENDERER_PROPERTY_SURFACE_OFFSET = {
    "surfaceOffset", DICOM_DEFAULT_RENDERER_SURFACE_OFFSET, 0.01, 10., {"Surface offset"}};
} // namespace dicom
} // namespace medicalimagingexplorer