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

#include "OpenDeckParameters.h"

namespace core
{
OpenDeckParameters::OpenDeckParameters()
    : _props("OpenDeck plugin parameters")
{
    _props.setProperty({PARAM_RESOLUTION_SCALING, 1.0,
                        Property::MetaData{"OpenDeck native resolution scale", "OpenDeck native resolution scale"}});
    _props.setProperty(
        {PARAM_CAMERA_SCALING, 1.0, Property::MetaData{"OpenDeck camera scaling", "OpenDeck camera scaling"}});
}
} // namespace core
