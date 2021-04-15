/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <plugin/biology/Node.h>

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace biology
{
using namespace details;

/**
 * @brief The RNASequence class
 */
class RNASequence : public Node
{
public:
    /**
     * @brief Construct a new RNASequence object
     *
     * @param scene Scene to which the RNA sequence should be added
     * @param details Details of the RNA sequence
     * @param position Relative position of the RNA sequence in the assembly
     */
    RNASequence(Scene& scene, const RNASequenceDetails& details);

    /**
     * @brief Get the map of RNA sequences
     *
     * @return The map of RNA sequences
     */
    RNASequenceMap getRNASequences() { return _rnaSequenceMap; }

private:
    Vector3f _trefoilKnot(float R, float t, const Vector3f& params);
    Vector3f _torus(float R, float t, const Vector3f& params);
    Vector3f _star(float R, float t);
    Vector3f _spring(float R, float t);
    Vector3f _heart(float R, float u);
    Vector3f _thing(float R, float t, const Vector3f& a);
    Vector3f _moebius(float R, float u, float v);

    RNASequenceMap _rnaSequenceMap;
};
} // namespace biology
} // namespace bioexplorer
