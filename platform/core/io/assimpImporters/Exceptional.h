/*
    Copyright 2006 - 2008 Blue Brain Project / EPFL

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

#ifndef INCLUDED_EXCEPTIONAL_H
#define INCLUDED_EXCEPTIONAL_H

#include <assimp/DefaultIOStream.h>
#include <stdexcept>
using std::runtime_error;

#ifdef _MSC_VER
#pragma warning(disable : 4275)
#endif

// ---------------------------------------------------------------------------
/** FOR IMPORTER PLUGINS ONLY: Simple exception class to be thrown if an
 *  unrecoverable error occurs while importing. Loading APIs return
 *  NULL instead of a valid aiScene then.  */
class DeadlyImportError : public runtime_error
{
public:
    /** Constructor with arguments */
    explicit DeadlyImportError(const std::string& errorText)
        : runtime_error(errorText)
    {
    }

private:
};

typedef DeadlyImportError DeadlyExportError;

#ifdef _MSC_VER
#pragma warning(default : 4275)
#endif

// ---------------------------------------------------------------------------
template <typename T>
struct ExceptionSwallower
{
    T operator()() const { return T(); }
};

// ---------------------------------------------------------------------------
template <typename T>
struct ExceptionSwallower<T*>
{
    T* operator()() const { return NULL; }
};

// ---------------------------------------------------------------------------
template <>
struct ExceptionSwallower<aiReturn>
{
    aiReturn operator()() const
    {
        try
        {
            throw;
        }
        catch (std::bad_alloc&)
        {
            return aiReturn_OUTOFMEMORY;
        }
        catch (...)
        {
            return aiReturn_FAILURE;
        }
    }
};

// ---------------------------------------------------------------------------
template <>
struct ExceptionSwallower<void>
{
    void operator()() const { return; }
};

#define ASSIMP_BEGIN_EXCEPTION_REGION() \
    {                                   \
        try                             \
        {
#define ASSIMP_END_EXCEPTION_REGION(type)    \
    }                                        \
    catch (...)                              \
    {                                        \
        return ExceptionSwallower<type>()(); \
    }                                        \
    }

#endif // INCLUDED_EXCEPTIONAL_H
