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

#include "utils/AbstractRenderer.h"

namespace core
{
namespace engine
{
namespace ospray
{
class BasicRenderer : public AbstractRenderer
{
public:
    BasicRenderer();

    /**
       Returns the class name as a string
       @return string containing the name of the object in the OSPRay context
    */
    std::string toString() const final { return RENDERER_PROPERTY_TYPE_BASIC; }
    void commit() final;
};
} // namespace ospray
} // namespace engine
} // namespace core