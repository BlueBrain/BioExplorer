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

#include <platform/core/common/BaseObject.h>
#include <platform/core/common/PropertyMap.h>

namespace core
{
constexpr auto PARAM_RESOLUTION_SCALING = "resolution-scaling";
constexpr auto PARAM_CAMERA_SCALING = "camera-scaling";

class OpenDeckParameters : public BaseObject
{
public:
    OpenDeckParameters();

    double getResolutionScaling() const { return _props.getProperty<double>(PARAM_RESOLUTION_SCALING); }
    void setResolutionScaling(const double resScaling) { _updateProperty(PARAM_RESOLUTION_SCALING, resScaling); }

    double getCameraScaling() const { return _props.getProperty<double>(PARAM_CAMERA_SCALING); }
    void setCameraScaling(const double cameraScaling) { _updateProperty(PARAM_CAMERA_SCALING, cameraScaling); }

    const PropertyMap& getPropertyMap() const { return _props; }
    PropertyMap& getPropertyMap() { return _props; }

private:
    PropertyMap _props;

    template <typename T>
    void _updateProperty(const char* property, const T& newValue)
    {
        if (!_isEqual(_props.getProperty<T>(property), newValue))
        {
            _props.updateProperty(property, newValue);
            markModified();
        }
    }
};
} // namespace core
