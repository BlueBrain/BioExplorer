/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_RNASEQUENCE_H
#define BIOEXPLORER_RNASEQUENCE_H

#include <brayns/engineapi/Model.h>
#include <plugin/bioexplorer/Node.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
typedef std::map<std::string, std::string> RNASequenceMap;

class RNASequence : public Node
{
public:
    RNASequence(Scene& scene, const RNASequenceDescriptor& rnaDescriptor,
                const Vector2f& range, const Vector3f& params,
                const Vector3f& position);

    // Class member accessors
    RNASequenceMap getRNASequences() { return _rnaSequenceMap; }

    ModelDescriptorPtr getModelDescriptor() { return _modelDescriptor; }

private:
    Vector3f _trefoilKnot(float R, float t, const Vector3f& params);
    Vector3f _torus(float R, float t, const Vector3f& params);
    Vector3f _star(float R, float t);
    Vector3f _spring(float R, float t);
    Vector3f _heart(float R, float u);
    Vector3f _thing(float R, float t, const Vector3f& a);
    Vector3f _moebius(float R, float u, float v);

    ModelDescriptorPtr _modelDescriptor{nullptr};

    RNASequenceMap _rnaSequenceMap;
};
} // namespace bioexplorer
#endif // BIOEXPLORER_RNASEQUENCE_H
