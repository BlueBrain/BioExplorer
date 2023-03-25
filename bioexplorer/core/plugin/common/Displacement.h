/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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
