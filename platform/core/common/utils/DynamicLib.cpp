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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/times.h>
#endif

#include "DynamicLib.h"

namespace core
{
DynamicLib::DynamicLib(const std::string& name)
{
    std::string file = name;
    std::string errorMessage;
#ifdef _WIN32
    std::string filename = name + ".dll";
    _handler = LoadLibrary(fullName.c_str());
    if (!_handler)
    {
        DWORD err = GetLastError();
        LPTSTR buffer;
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                      err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, NULL);
        errorMessage = buffer;
        LocalFree(buffer);
    }

#else
#if defined(__MACOSX__) || defined(__APPLE__)
    std::string filename = "lib" + file + ".dylib";
#else
    std::string filename = "lib" + file + ".so";
#endif
    _handler = dlopen(filename.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!_handler)
        errorMessage = dlerror();
#endif
    if (!_handler)
        throw std::runtime_error("Error opening dynamic library: " + filename + ": " + errorMessage);
}

DynamicLib::~DynamicLib()
{
    if (!_handler)
        return;

#ifdef _WIN32
    if (_handler != GetModuleHandle(0))
        FreeLibrary(_handler);
#else
    if (_handler != RTLD_DEFAULT)
        dlclose(_handler);
#endif
    _handler = 0;
}

DynamicLib::DynamicLib(DynamicLib&& other)
{
    _handler = other._handler;
    other._handler = nullptr;
}

DynamicLib& DynamicLib::operator=(DynamicLib&& other)
{
    _handler = other._handler;
    other._handler = nullptr;
    return *this;
}

void* DynamicLib::getSymbolAddress(const std::string& name) const
{
    if (!_handler)
        return nullptr;

#ifdef _WIN32 //_MSC_VER
    return GetProcAddress((HMODULE)_handler, name.c_str());
#else
    return dlsym(_handler, name.c_str());
#endif
}
} // namespace core
