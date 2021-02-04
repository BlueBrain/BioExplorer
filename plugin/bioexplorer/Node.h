/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
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

#include <plugin/common/Types.h>

namespace bioexplorer
{
/**
 * @brief The Node class
 */
class Node
{
public:
    /**
     * @brief Construct a new Node object
     *
     */
    Node();

    /**
     * @brief Destroy the Node object
     *
     */
    virtual ~Node();

    /**
     * @brief Get the Model Descriptor object
     *
     * @return ModelDescriptorPtr
     */
    ModelDescriptorPtr getModelDescriptor() { return _modelDescriptor; }

protected:
    ModelDescriptorPtr _modelDescriptor{nullptr};
};
} // namespace bioexplorer
