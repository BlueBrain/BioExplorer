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

#ifndef RNASEQUENCE_H
#define RNASEQUENCE_H

#include <brayns/engineapi/Model.h>
#include <common/Node.h>
#include <common/types.h>

typedef std::map<std::string, std::string> RNASequenceMap;

class RNASequence : public Node
{
public:
    RNASequence(brayns::Scene& scene,
                const RNASequenceDescriptor& rnaDescriptor,
                const brayns::Vector2f& range, const brayns::Vector3f& params);

    // Class member accessors
    RNASequenceMap getRNASequences() { return _rnaSequenceMap; }

    brayns::ModelDescriptorPtr getModelDescriptor() { return _modelDescriptor; }

private:
    brayns::Vector3f _trefoilKnot(float R, float t,
                                  const brayns::Vector3f& params);
    brayns::Vector3f _torus(float R, float t, const brayns::Vector3f& params);
    brayns::Vector3f _star(float R, float t);
    brayns::Vector3f _spring(float R, float t);
    brayns::Vector3f _heart(float R, float u);
    brayns::Vector3f _thing(float R, float t, const brayns::Vector3f& a);
    brayns::Vector3f _moebius(float R, float u, float v);

    brayns::ModelDescriptorPtr _modelDescriptor{nullptr};

    RNASequenceMap _rnaSequenceMap;
};

#endif // RNASEQUENCE_H
