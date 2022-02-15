/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <plugin/common/Node.h>

namespace bioexplorer
{
namespace molecularsystems
{
using namespace brayns;
using namespace common;
using namespace details;

/**
 * @brief
 */
class EnzymeReaction : public Node
{
public:
    /**
     * @brief Construct a new Enzyme Reaction object
     *
     * @param scene
     * @param details
     * @param enzyme
     * @param substrate
     * @param product
     */
    EnzymeReaction(Scene& scene, const EnzymeReactionDetails& details,
                   ProteinPtr enzyme, ProteinPtr substrate, ProteinPtr product);

    /**
     * @brief Set the Progress object
     *
     * @param progress
     */
    void setProgress(const uint64_t instanceId, const double progress);

protected:
    Scene& _scene;
    ProteinPtr _enzyme{nullptr};
    ProteinPtr _substrate{nullptr};
    ProteinPtr _product{nullptr};
    const EnzymeReactionDetails& _details;
};
} // namespace molecularsystems
} // namespace bioexplorer
