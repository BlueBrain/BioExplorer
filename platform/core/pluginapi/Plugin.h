/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *                     Jafet Villafranca <jafet.villafrancadiaz@epfl.ch>
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
