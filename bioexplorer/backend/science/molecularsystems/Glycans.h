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

#include <platform/core/engineapi/Model.h>
#include <science/api/Params.h>
#include <science/molecularsystems/Molecule.h>

namespace bioexplorer
{
namespace molecularsystems
{
/**
 * @brief The Glycans class
 */
class Glycans : public Molecule
{
public:
    /**
     * @brief Construct a new Glycans object
     *
     * @param scene The 3D scene where the glycans are added
     * @param details The data structure describing the glycans
     */
    Glycans(core::Scene& scene, const details::SugarDetails& details);

private:
    details::SugarDetails _details;
};
} // namespace molecularsystems
} // namespace bioexplorer
