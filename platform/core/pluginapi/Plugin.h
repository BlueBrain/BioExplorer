/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
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
