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

#include <science/common/SDFGeometries.h>

namespace bioexplorer
{
namespace connectomics
{

struct SynaptomeNode
{
    core::Vector3f position;
    core::Vector3f direction;
    float mass{1.f};
};
/**
 * Load Synaptome from database
 */
class Synaptome : public common::SDFGeometries
{
public:
    /**
     * @brief Construct a new Synaptome object
     *
     * @param scene 3D scene into which the Synaptome should be loaded
     * @param details Set of attributes defining how the Synaptome should be loaded
     */
    Synaptome(core::Scene& scene, const details::SynaptomeDetails& details, const core::Vector3d& position,
              const core::Quaterniond& rotation, const core::LoaderProgress& callback = core::LoaderProgress());

protected:
    void _addNode(const uint64_t id, const core::Vector3f& position, float mass);
    void _addEdge(uint64_t source, uint64_t target, const core::Vector3f& direction);

private:
    double _getDisplacementValue(const DisplacementElement& element) final { return 0; }

    void _buildModel(const core::LoaderProgress& callback);

    const details::SynaptomeDetails _details;
    core::Scene& _scene;

    std::map<uint64_t, SynaptomeNode> _nodes;
    std::vector<core::Vector2ui> _edges;
};
} // namespace connectomics
} // namespace bioexplorer
