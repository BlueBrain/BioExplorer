/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "SharedDataVolume.h"

#include <platform/core/common/Logs.h>

#include <fcntl.h>
#include <fstream>
#include <future>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
const int NO_DESCRIPTOR = -1;
}

namespace core
{
SharedDataVolume::~SharedDataVolume()
{
    if (_memoryMapPtr)
    {
        ::munmap((void*)_memoryMapPtr, _size);
        _memoryMapPtr = nullptr;
    }
    if (_cacheFileDescriptor != NO_DESCRIPTOR)
    {
        ::close(_cacheFileDescriptor);
        _cacheFileDescriptor = NO_DESCRIPTOR;
    }
}

void SharedDataVolume::mapData(const std::string& filename)
{
    _cacheFileDescriptor = open(filename.c_str(), O_RDONLY);
    if (_cacheFileDescriptor == NO_DESCRIPTOR)
        throw std::runtime_error("Failed to open volume file " + filename);

    struct stat sb;
    if (::fstat(_cacheFileDescriptor, &sb) == NO_DESCRIPTOR)
    {
        ::close(_cacheFileDescriptor);
        _cacheFileDescriptor = NO_DESCRIPTOR;
        throw std::runtime_error("Failed to open volume file " + filename);
    }

    _size = sb.st_size;
    _memoryMapPtr = ::mmap(0, _size, PROT_READ, MAP_PRIVATE, _cacheFileDescriptor, 0);
    if (_memoryMapPtr == MAP_FAILED)
    {
        _memoryMapPtr = nullptr;
        ::close(_cacheFileDescriptor);
        _cacheFileDescriptor = NO_DESCRIPTOR;
        throw std::runtime_error("Failed to open volume file " + filename);
    }

    setVoxels(_memoryMapPtr);
}

void SharedDataVolume::mapData(const uint8_ts& buffer)
{
    _memoryBuffer.insert(_memoryBuffer.begin(), buffer.begin(), buffer.end());
    setVoxels(_memoryBuffer.data());
}

void SharedDataVolume::mapData(uint8_ts&& buffer)
{
    _memoryBuffer = std::move(buffer);
    setVoxels(_memoryBuffer.data());
}
} // namespace core
