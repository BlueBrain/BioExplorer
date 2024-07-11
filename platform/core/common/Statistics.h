/*
    Copyright 2018 - 0211 Blue Brain Project / EPFL

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
#include <platform/core/common/Types.h>

SERIALIZATION_ACCESS(Statistics)

namespace core
{
/** Captures various statistics about rendering, scenes, etc. */
class Statistics : public BaseObject
{
public:
    double getFPS() const { return _fps; }
    void setFPS(const double fps) { _updateValue(_fps, fps); }
    size_t getSceneSizeInBytes() const { return _sceneSizeInBytes; }
    void setSceneSizeInBytes(const size_t sceneSizeInBytes) { _updateValue(_sceneSizeInBytes, sceneSizeInBytes); }

private:
    double _fps{0.0};
    size_t _sceneSizeInBytes{0};

    SERIALIZATION_FRIEND(Statistics)
};
} // namespace core
