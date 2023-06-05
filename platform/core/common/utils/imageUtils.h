/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#pragma once

#include <platform/core/common/Types.h>

#include <FreeImage.h>

namespace core
{
namespace freeimage
{
struct ImageDeleter
{
    inline void operator()(FIBITMAP* image)
    {
        if (image)
            FreeImage_Unload(image);
    }
};

struct MemoryDeleter
{
    inline void operator()(FIMEMORY* memory)
    {
        if (memory)
            FreeImage_CloseMemory(memory);
    }
};

using ImagePtr = std::unique_ptr<FIBITMAP, ImageDeleter>;
using MemoryPtr = std::unique_ptr<FIMEMORY, MemoryDeleter>;

bool SwapRedBlue32(FIBITMAP* freeImage);
std::string getBase64Image(ImagePtr image, const std::string& format, const int quality);
ImagePtr mergeImages(const std::vector<ImagePtr>& images);
} // namespace freeimage
} // namespace core
