#pragma once

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

#include "ParsingUtils.h"
#include <assimp/IOStream.hpp>
#include <assimp/types.h>

#include <vector>

namespace Assimp
{
// ---------------------------------------------------------------------------
/**
 *  Implementation of a cached stream buffer.
 */
template <class T>
class IOStreamBuffer
{
public:
    /// @brief  The class constructor.
    IOStreamBuffer(size_t cache = 4096 * 4096);

    /// @brief  The class destructor.
    ~IOStreamBuffer();

    /// @brief  Will open the cached access for a given stream.
    /// @param  stream      The stream to cache.
    /// @return true if successful.
    bool open(IOStream *stream);

    /// @brief  Will close the cached access.
    /// @return true if successful.
    bool close();

    /// @brief  Returns the file-size.
    /// @return The file-size.
    size_t size() const;

    /// @brief  Returns the cache size.
    /// @return The cache size.
    size_t cacheSize() const;

    /// @brief  Will read the next block.
    /// @return true if successful.
    bool readNextBlock();

    /// @brief  Returns the number of blocks to read.
    /// @return The number of blocks.
    size_t getNumBlocks() const;

    /// @brief  Returns the current block index.
    /// @return The current block index.
    size_t getCurrentBlockIndex() const;

    /// @brief  Returns the current file pos.
    /// @return The current file pos.
    size_t getFilePos() const;

    /// @brief  Will read the next line.
    /// @param  buffer      The buffer for the next line.
    /// @return true if successful.
    bool getNextDataLine(std::vector<T> &buffer, T continuationToken);

    /// @brief  Will read the next line ascii or binary end line char.
    /// @param  buffer      The buffer for the next line.
    /// @return true if successful.
    bool getNextLine(std::vector<T> &buffer);

    /// @brief  Will read the next block.
    /// @param  buffer      The buffer for the next block.
    /// @return true if successful.
    bool getNextBlock(std::vector<T> &buffer);

private:
    IOStream *m_stream;
    size_t m_filesize;
    size_t m_cacheSize;
    size_t m_numBlocks;
    size_t m_blockIdx;
    std::vector<T> m_cache;
    size_t m_cachePos;
    size_t m_filePos;
};

template <class T>
inline IOStreamBuffer<T>::IOStreamBuffer(size_t cache)
    : m_stream(nullptr)
    , m_filesize(0)
    , m_cacheSize(cache)
    , m_numBlocks(0)
    , m_blockIdx(0)
    , m_cachePos(0)
    , m_filePos(0)
{
    m_cache.resize(cache);
    std::fill(m_cache.begin(), m_cache.end(), '\n');
}

template <class T>
inline IOStreamBuffer<T>::~IOStreamBuffer()
{
    // empty
}

template <class T>
inline bool IOStreamBuffer<T>::open(IOStream *stream)
{
    //  file still opened!
    if (nullptr != m_stream)
    {
        return false;
    }

    //  Invalid stream pointer
    if (nullptr == stream)
    {
        return false;
    }

    m_stream = stream;
    m_filesize = m_stream->FileSize();
    if (m_filesize == 0)
    {
        return false;
    }
    if (m_filesize < m_cacheSize)
    {
        m_cacheSize = m_filesize;
    }

    m_numBlocks = m_filesize / m_cacheSize;
    if ((m_filesize % m_cacheSize) > 0)
    {
        m_numBlocks++;
    }

    return true;
}

template <class T>
inline bool IOStreamBuffer<T>::close()
{
    if (nullptr == m_stream)
    {
        return false;
    }

    // init counters and state vars
    m_stream = nullptr;
    m_filesize = 0;
    m_numBlocks = 0;
    m_blockIdx = 0;
    m_cachePos = 0;
    m_filePos = 0;

    return true;
}

template <class T>
inline size_t IOStreamBuffer<T>::size() const
{
    return m_filesize;
}

template <class T>
inline size_t IOStreamBuffer<T>::cacheSize() const
{
    return m_cacheSize;
}

template <class T>
inline bool IOStreamBuffer<T>::readNextBlock()
{
    m_stream->Seek(m_filePos, aiOrigin_SET);
    size_t readLen = m_stream->Read(&m_cache[0], sizeof(T), m_cacheSize);
    if (readLen == 0)
    {
        return false;
    }
    if (readLen < m_cacheSize)
    {
        m_cacheSize = readLen;
    }
    m_filePos += m_cacheSize;
    m_cachePos = 0;
    m_blockIdx++;

    return true;
}

template <class T>
inline size_t IOStreamBuffer<T>::getNumBlocks() const
{
    return m_numBlocks;
}

template <class T>
inline size_t IOStreamBuffer<T>::getCurrentBlockIndex() const
{
    return m_blockIdx;
}

template <class T>
inline size_t IOStreamBuffer<T>::getFilePos() const
{
    return m_filePos;
}

template <class T>
inline bool IOStreamBuffer<T>::getNextDataLine(std::vector<T> &buffer, T continuationToken)
{
    buffer.resize(m_cacheSize);
    if (m_cachePos == m_cacheSize || 0 == m_filePos)
    {
        if (!readNextBlock())
        {
            return false;
        }
    }

    bool continuationFound(false), endOfDataLine(false);
    size_t i = 0;
    while (!endOfDataLine)
    {
        if (continuationToken == m_cache[m_cachePos])
        {
            continuationFound = true;
            ++m_cachePos;
        }
        if (IsLineEnd(m_cache[m_cachePos]))
        {
            if (!continuationFound)
            {
                // the end of the data line
                break;
            }
            else
            {
                // skip line end
                while (m_cache[m_cachePos] != '\n')
                {
                    ++m_cachePos;
                }
                ++m_cachePos;
                continuationFound = false;
            }
        }

        buffer[i] = m_cache[m_cachePos];
        m_cachePos++;
        i++;
        if (m_cachePos >= m_cacheSize)
        {
            if (!readNextBlock())
            {
                return false;
            }
        }
    }

    buffer[i] = '\n';
    m_cachePos++;

    return true;
}

static inline bool isEndOfCache(size_t pos, size_t cacheSize)
{
    return (pos == cacheSize);
}

template <class T>
inline bool IOStreamBuffer<T>::getNextLine(std::vector<T> &buffer)
{
    buffer.resize(m_cacheSize);
    if (isEndOfCache(m_cachePos, m_cacheSize) || 0 == m_filePos)
    {
        if (!readNextBlock())
        {
            return false;
        }
    }

    if (IsLineEnd(m_cache[m_cachePos]))
    {
        // skip line end
        while (m_cache[m_cachePos] != '\n')
        {
            ++m_cachePos;
        }
        ++m_cachePos;
        if (isEndOfCache(m_cachePos, m_cacheSize))
        {
            if (!readNextBlock())
            {
                return false;
            }
        }
    }

    size_t i = 0;
    while (!IsLineEnd(m_cache[m_cachePos]))
    {
        buffer[i] = m_cache[m_cachePos];
        m_cachePos++;
        i++;
        if (m_cachePos >= m_cacheSize)
        {
            if (!readNextBlock())
            {
                return false;
            }
        }
    }
    buffer[i] = '\n';
    m_cachePos++;

    return true;
}

template <class T>
inline bool IOStreamBuffer<T>::getNextBlock(std::vector<T> &buffer)
{
    // just return the last blockvalue if getNextLine was used before
    if (m_cachePos != 0)
    {
        buffer = std::vector<T>(m_cache.begin() + m_cachePos, m_cache.end());
        m_cachePos = 0;
    }
    else
    {
        if (!readNextBlock())
            return false;

        buffer = std::vector<T>(m_cache.begin(), m_cache.end());
    }
    return true;
}

} // namespace Assimp
