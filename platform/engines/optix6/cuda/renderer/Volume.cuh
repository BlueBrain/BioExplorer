/*
 * Copyright (c) 2020-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/renderer/TransferFunction.cuh>

using namespace optix;

const float DEFAULT_SHADING_ALPHA_RATIO = 1.5f;
const float DEFAULT_SHADING_AMBIENT = 0.2f;
const float DEFAULT_GRADIENT_OFFSET = 0.001f;

rtDeclareVariable(float, specularExponent, , );
