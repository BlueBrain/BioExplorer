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

#include "VolumeParameters.h"

namespace
{
const std::string PARAM_VOLUME_DIMENSIONS = "volume-dimensions";
const std::string PARAM_VOLUME_ELEMENT_SPACING = "volume-element-spacing";
const std::string PARAM_VOLUME_OFFSET = "volume-offset";
} // namespace

namespace core
{
VolumeParameters::VolumeParameters()
    : AbstractParameters("Volume")
    , _dimensions(0, 0, 0)
    , _elementSpacing(1.f, 1.f, 1.f)
    , _offset(0.f, 0.f, 0.f)
{
    _parameters.add_options()(PARAM_VOLUME_DIMENSIONS.c_str(), po::fixed_tokens_value<uints>(3, 3),
                              "Volume dimensions [uint uint uint]")(
        PARAM_VOLUME_ELEMENT_SPACING.c_str(), po::fixed_tokens_value<floats>(3, 3),
        "Element spacing in the volume [float float float]")(PARAM_VOLUME_OFFSET.c_str(),
                                                             po::fixed_tokens_value<floats>(3, 3),
                                                             "Volume offset [float float float]");
}

void VolumeParameters::parse(const po::variables_map& vm)
{
    if (vm.count(PARAM_VOLUME_DIMENSIONS))
    {
        auto values = vm[PARAM_VOLUME_DIMENSIONS].as<uints>();
        _dimensions = Vector3ui(values[0], values[1], values[2]);
    }
    if (vm.count(PARAM_VOLUME_ELEMENT_SPACING))
    {
        auto values = vm[PARAM_VOLUME_ELEMENT_SPACING].as<floats>();
        _elementSpacing = Vector3f(values[0], values[1], values[2]);
    }
    if (vm.count(PARAM_VOLUME_OFFSET))
    {
        auto values = vm[PARAM_VOLUME_OFFSET].as<floats>();
        _offset = Vector3f(values[0], values[1], values[2]);
    }
    markModified();
}

void VolumeParameters::print()
{
    AbstractParameters::print();
    CORE_INFO("Dimensions      : " << _dimensions);
    CORE_INFO("Element spacing : " << _elementSpacing);
    CORE_INFO("Offset          : " << _offset);
}
} // namespace core
