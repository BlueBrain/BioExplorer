/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
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

#include "FieldParameters.h"

namespace
{
const std::string PARAM_FIELD_DISTANCE = "field-distance";
const std::string PARAM_FIELD_CUTOFF = "field-cutoff";
const std::string PARAM_FIELD_GRADIENT_SHADING = "field-gradient-shading";
const std::string PARAM_FIELD_GRADIENT_OFFSET = "field-gradient-offset";
const std::string PARAM_FIELD_SAMPLING_RATE = "field-sampling-rate";
const std::string PARAM_FIELD_EPSILON = "field-epsilon";
} // namespace

namespace core
{
FieldParameters::FieldParameters()
    : AbstractParameters("Field")
{
    _parameters.add_options()
        //
        (PARAM_FIELD_GRADIENT_SHADING.c_str(), po::value<bool>(), "Gradient shading [bool]")
        //
        (PARAM_FIELD_GRADIENT_OFFSET.c_str(), po::value<float>(), "Gradient shading offset [float]")
        //
        (PARAM_FIELD_SAMPLING_RATE.c_str(), po::value<float>(), "Gradient shading sampling rate [float]")
        //
        (PARAM_FIELD_DISTANCE.c_str(), po::value<float>(), "Initial distance between leafs in the octree [float]")
        //
        (PARAM_FIELD_CUTOFF.c_str(), po::value<float>(), "Cutoff distance between leafs in the octree [float]")
        //
        (PARAM_FIELD_EPSILON.c_str(), po::value<float>(), "Epsilon between intersections [float]");
}

void FieldParameters::parse(const po::variables_map& vm)
{
    if (vm.count(PARAM_FIELD_GRADIENT_SHADING))
        _gradientShading = vm[PARAM_FIELD_GRADIENT_SHADING].as<bool>();
    if (vm.count(PARAM_FIELD_GRADIENT_OFFSET))
        _gradientOffset = vm[PARAM_FIELD_GRADIENT_OFFSET].as<float>();
    if (vm.count(PARAM_FIELD_SAMPLING_RATE))
        _samplingRate = vm[PARAM_FIELD_SAMPLING_RATE].as<float>();
    if (vm.count(PARAM_FIELD_CUTOFF))
        _cutoff = vm[PARAM_FIELD_CUTOFF].as<float>();
    if (vm.count(PARAM_FIELD_CUTOFF))
        _cutoff = vm[PARAM_FIELD_CUTOFF].as<float>();
    if (vm.count(PARAM_FIELD_EPSILON))
        _epsilon = vm[PARAM_FIELD_EPSILON].as<float>();
    markModified();
}

void FieldParameters::print()
{
    AbstractParameters::print();
    CORE_INFO("Gradient shading: " << asString(_gradientShading));
    CORE_INFO("Gradient offset : " << _gradientOffset);
    CORE_INFO("Sampling rate   : " << _samplingRate);
    CORE_INFO("Distance        : " << _distance);
    CORE_INFO("Cutoff          : " << _cutoff);
    CORE_INFO("Epsilon         : " << _epsilon);
}
} // namespace core
