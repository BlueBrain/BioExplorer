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

namespace bioexplorer
{
enum class DisplacementElement
{
    vasculature_segment_strength = 0,
    vasculature_segment_frequency = 1,
    morphology_soma_strength = 2,
    morphology_soma_frequency = 3,
    morphology_section_strength = 4,
    morphology_section_frequency = 5,
    morphology_nucleus_strength = 6,
    morphology_nucleus_frequency = 7,
    morphology_mitochondrion_strength = 8,
    morphology_mitochondrion_frequency = 9,
    morphology_myelin_steath_strength = 10,
    morphology_myelin_steath_frequency = 11,
    morphology_spine_strength = 12,
    morphology_spine_frequency = 13
};

const double DEFAULT_VASCULATURE_SEGMENT_STRENGTH = 0.3;
const double DEFAULT_VASCULATURE_SEGMENT_FREQUENCY = 0.5;

const double DEFAULT_MORPHOLOGY_SOMA_STRENGTH = 0.1;
const double DEFAULT_MORPHOLOGY_SOMA_FREQUENCY = 3.0;
const double DEFAULT_MORPHOLOGY_SECTION_STRENGTH = 0.15;
const double DEFAULT_MORPHOLOGY_SECTION_FREQUENCY = 2.0;
const double DEFAULT_MORPHOLOGY_NUCLEUS_STRENGTH = 0.01;
const double DEFAULT_MORPHOLOGY_NUCLEUS_FREQUENCY = 2.0;
const double DEFAULT_MORPHOLOGY_MITOCHONDRION_STRENGTH = 0.2;
const double DEFAULT_MORPHOLOGY_MITOCHONDRION_FREQUENCY = 100.0;
const double DEFAULT_MORPHOLOGY_MYELIN_STEATH_STRENGTH = 0.1;
const double DEFAULT_MORPHOLOGY_MYELIN_STEATH_FREQUENCY = 2.5;
const double DEFAULT_MORPHOLOGY_SPINE_STRENGTH = 0.01;
const double DEFAULT_MORPHOLOGY_SPINE_FREQUENCY = 25.0;
} // namespace bioexplorer
