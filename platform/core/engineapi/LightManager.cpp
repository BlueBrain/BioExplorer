/*
    Copyright 2019 - 2024 Blue Brain Project / EPFL

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

#include "LightManager.h"

#include <platform/core/common/light/Light.h>
#include <platform/core/common/utils/Utils.h>

#include <algorithm>

namespace core
{
size_t LightManager::addLight(LightPtr light)
{
    // If light already added, return id
    auto itInv = _lightsInverse.find(light);
    if (itInv != _lightsInverse.end())
    {
        markModified();
        return itInv->second;
    }

    // If lights are empty we reset id counter to avoid huge numbers
    if (_lights.empty())
        _IDctr = 0;

    const size_t id = _IDctr++;
    _lights.insert({id, light});
    _lightsInverse.insert({light, id});
    markModified();
    return id;
}

void LightManager::removeLight(const size_t id)
{
    auto it = _lights.find(id);
    if (it != _lights.end())
    {
        auto light = it->second;

        auto itInv = _lightsInverse.find(light);
        assert(itInv != _lightsInverse.end());
        if (itInv != _lightsInverse.end())
            _lightsInverse.erase(itInv);

        _lights.erase(it);

        markModified();
    }
}

void LightManager::removeLight(LightPtr light)
{
    auto itInv = _lightsInverse.find(light);

    if (itInv != _lightsInverse.end())
    {
        const size_t id = itInv->second;
        auto it = _lights.find(id);
        assert(it != _lights.end());
        if (it != _lights.end())
            _lights.erase(it);
    }
}

LightPtr LightManager::getLight(const size_t id)
{
    auto it = _lights.find(id);
    if (it != _lights.end())
        return it->second;

    return nullptr;
}

const std::map<size_t, LightPtr>& LightManager::getLights() const
{
    return _lights;
}

void LightManager::clearLights()
{
    _lights.clear();
    _lightsInverse.clear();
    markModified();
}

} // namespace core
