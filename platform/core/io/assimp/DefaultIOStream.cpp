/*
    Copyright 2006 - 2017 Blue Brain Project / EPFL

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
/** @file  DefaultIOStream.cpp
 *  @brief Default File I/O implementation for #Importer
 */

#include <assimp/DefaultIOStream.h>
#include <assimp/ai_assert.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace Assimp;

// ----------------------------------------------------------------------------------
DefaultIOStream::~DefaultIOStream()
{
    if (mFile)
    {
        ::fclose(mFile);
        mFile = nullptr;
    }
}

// ----------------------------------------------------------------------------------
size_t DefaultIOStream::Read(void* pvBuffer, size_t pSize, size_t pCount)
{
    ai_assert(NULL != pvBuffer && 0 != pSize && 0 != pCount);
    return (mFile ? ::fread(pvBuffer, pSize, pCount, mFile) : 0);
}

// ----------------------------------------------------------------------------------
size_t DefaultIOStream::Write(const void* pvBuffer, size_t pSize, size_t pCount)
{
    ai_assert(NULL != pvBuffer && 0 != pSize && 0 != pCount);
    return (mFile ? ::fwrite(pvBuffer, pSize, pCount, mFile) : 0);
}

// ----------------------------------------------------------------------------------
aiReturn DefaultIOStream::Seek(size_t pOffset, aiOrigin pOrigin)
{
    if (!mFile)
    {
        return AI_FAILURE;
    }

    // Just to check whether our enum maps one to one with the CRT constants
    static_assert(aiOrigin_CUR == SEEK_CUR && aiOrigin_END == SEEK_END && aiOrigin_SET == SEEK_SET,
                  "aiOrigin_CUR == SEEK_CUR && \
        aiOrigin_END == SEEK_END && aiOrigin_SET == SEEK_SET");

    // do the seek
    return (0 == ::fseek(mFile, (long)pOffset, (int)pOrigin) ? AI_SUCCESS : AI_FAILURE);
}

// ----------------------------------------------------------------------------------
size_t DefaultIOStream::Tell() const
{
    if (!mFile)
    {
        return 0;
    }
    return ::ftell(mFile);
}

// ----------------------------------------------------------------------------------
size_t DefaultIOStream::FileSize() const
{
    if (!mFile || mFilename.empty())
    {
        return 0;
    }

    if (SIZE_MAX == mCachedSize)
    {
// Although fseek/ftell would allow us to reuse the existing file handle here,
// it is generally unsafe because:
//  - For binary streams, it is not technically well-defined
//  - For text files the results are meaningless
// That's why we use the safer variant fstat here.
//
// See here for details:
// https://www.securecoding.cert.org/confluence/display/seccode/FIO19-C.+Do+not+use+fseek()+and+ftell()+to+compute+the+size+of+a+regular+file
#if defined _WIN32 && (!defined __GNUC__ || __MSVCRT_VERSION__ >= 0x0601)
        struct __stat64 fileStat;
        int err = _stat64(mFilename.c_str(), &fileStat);
        if (0 != err)
            return 0;
        mCachedSize = (size_t)(fileStat.st_size);
#elif defined __GNUC__ || defined __APPLE__ || defined __MACH__ || defined __FreeBSD__
        struct stat fileStat;
        int err = stat(mFilename.c_str(), &fileStat);
        if (0 != err)
            return 0;
        const unsigned long long cachedSize = fileStat.st_size;
        mCachedSize = static_cast<size_t>(cachedSize);
#else
#error "Unknown platform"
#endif
    }
    return mCachedSize;
}

// ----------------------------------------------------------------------------------
void DefaultIOStream::Flush()
{
    if (mFile)
    {
        ::fflush(mFile);
    }
}

// ----------------------------------------------------------------------------------
