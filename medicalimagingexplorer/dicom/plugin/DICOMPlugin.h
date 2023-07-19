/* Copyright (c) 2018, Cyrille Favreau
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@gmail.com>
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

#pragma once

#include <plugin/io/DICOMLoader.h>

#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace medicalimagingexplorer
{
namespace dicom
{
using namespace core;

/**
 * @brief The DICOM plugin class manages the loading of DICOM datasets
 */
class DICOMPlugin : public ExtensionPlugin
{
public:
    DICOMPlugin(PropertyMap&& dicomParams);

    void init() final;

private:
    PropertyMap _dicomParams;
    bool _dirty{false};
};
} // namespace dicom
} // namespace medicalimagingexplorer