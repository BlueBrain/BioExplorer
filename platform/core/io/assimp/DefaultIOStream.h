/*
    Copyright 2006 - 2024 Blue Brain Project / EPFL

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

/** @file Default file I/O using fXXX()-family of functions */
#ifndef AI_DEFAULTIOSTREAM_H_INC
#define AI_DEFAULTIOSTREAM_H_INC

#include <assimp/Defines.h>
#include <assimp/IOStream.hpp>
#include <assimp/importerdesc.h>
#include <stdio.h>

namespace Assimp
{
// ----------------------------------------------------------------------------------
//! @class  DefaultIOStream
//! @brief  Default IO implementation, use standard IO operations
//! @note   An instance of this class can exist without a valid file handle
//!         attached to it. All calls fail, but the instance can nevertheless be
//!         used with no restrictions.
class ASSIMP_API DefaultIOStream : public IOStream
{
    friend class DefaultIOSystem;
#if __ANDROID__
#if __ANDROID_API__ > 9
#if defined(AI_CONFIG_ANDROID_JNI_ASSIMP_MANAGER_SUPPORT)
    friend class AndroidJNIIOSystem;
#endif // defined(AI_CONFIG_ANDROID_JNI_ASSIMP_MANAGER_SUPPORT)
#endif // __ANDROID_API__ > 9
#endif // __ANDROID__

protected:
    DefaultIOStream();
    DefaultIOStream(FILE* pFile, const std::string& strFilename);

public:
    /** Destructor public to allow simple deletion to close the file. */
    ~DefaultIOStream();

    // -------------------------------------------------------------------
    /// Read from stream
    size_t Read(void* pvBuffer, size_t pSize, size_t pCount);

    // -------------------------------------------------------------------
    /// Write to stream
    size_t Write(const void* pvBuffer, size_t pSize, size_t pCount);

    // -------------------------------------------------------------------
    /// Seek specific position
    aiReturn Seek(size_t pOffset, aiOrigin pOrigin);

    // -------------------------------------------------------------------
    /// Get current seek position
    size_t Tell() const;

    // -------------------------------------------------------------------
    /// Get size of file
    size_t FileSize() const;

    // -------------------------------------------------------------------
    /// Flush file contents
    void Flush();

private:
    //  File data-structure, using clib
    FILE* mFile;
    //  Filename
    std::string mFilename;

    // Cached file size
    mutable size_t mCachedSize;
};

// ----------------------------------------------------------------------------------
inline DefaultIOStream::DefaultIOStream()
    : mFile(NULL)
    , mFilename("")
    , mCachedSize(SIZE_MAX)
{
    // empty
}

// ----------------------------------------------------------------------------------
inline DefaultIOStream::DefaultIOStream(FILE* pFile, const std::string& strFilename)
    : mFile(pFile)
    , mFilename(strFilename)
    , mCachedSize(SIZE_MAX)
{
    // empty
}
// ----------------------------------------------------------------------------------

} // namespace Assimp

#endif //!!AI_DEFAULTIOSTREAM_H_INC
