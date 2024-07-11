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

#include <platform/core/common/Types.h>

namespace core
{
/** The API that plugins can use to  interact with Core. */
class PluginAPI
{
public:
    virtual ~PluginAPI() = default;

    virtual Engine& getEngine() = 0;

    /** @return access to the scene of Core. */
    virtual Scene& getScene() = 0;

    /** @return access to the parameters of Core. */
    virtual ParametersManager& getParametersManager() = 0;

    /** @return access to the action interface of Core. */
    virtual ActionInterface* getActionInterface() = 0;

    /** @return access to the keyboard handler of Core. */
    virtual KeyboardHandler& getKeyboardHandler() = 0;

    /** @return access to the camera manipulator of Core. */
    virtual AbstractManipulator& getCameraManipulator() = 0;

    /** @return access to the camera of Core. */
    virtual Camera& getCamera() = 0;

    /** @return access to the renderer of Core. */
    virtual Renderer& getRenderer() = 0;

    /** Triggers a new preRender() and potentially render() and postRender(). */
    virtual void triggerRender() = 0;

    /** Set the action interface to be used by Core main loop. */
    virtual void setActionInterface(const ActionInterfacePtr& interface) = 0;
};
} // namespace core
