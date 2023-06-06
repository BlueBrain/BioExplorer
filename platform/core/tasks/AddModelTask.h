/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel.Nachbaur@epfl.ch
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

#include <platform/core/common/loader/Loader.h>
#include <platform/core/common/tasks/Task.h>

namespace core
{
/**
 * A task which loads data from the path of the given params and adds the loaded
 * model to the engines' scene.
 */
class AddModelTask : public Task<ModelDescriptorPtr>
{
public:
    AddModelTask(const ModelParams& model, Engine& engine);
};
} // namespace core
