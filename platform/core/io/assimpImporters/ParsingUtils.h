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

/** @file ParsingUtils.h
 *  @brief Defines helper functions for text parsing
 */
#ifndef AI_PARSING_UTILS_H_INC
#define AI_PARSING_UTILS_H_INC

#include "StringComparison.h"
#include "StringUtils.h"
#include <assimp/defs.h>

namespace Assimp
{
// NOTE: the functions below are mostly intended as replacement for
// std::upper, std::lower, std::isupper, std::islower, std::isspace.
// we don't bother of locales. We don't want them. We want reliable
// (i.e. identical) results across all locales.

// The functions below accept any character type, but know only
// about ASCII. However, UTF-32 is the only safe ASCII superset to
// use since it doesn't have multi-byte sequences.

static const unsigned int BufferSize = 4096;

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE char_t ToLower(char_t in)
{
    return (in >= (char_t)'A' && in <= (char_t)'Z') ? (char_t)(in + 0x20) : in;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE char_t ToUpper(char_t in)
{
    return (in >= (char_t)'a' && in <= (char_t)'z') ? (char_t)(in - 0x20) : in;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool IsUpper(char_t in)
{
    return (in >= (char_t)'A' && in <= (char_t)'Z');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool IsLower(char_t in)
{
    return (in >= (char_t)'a' && in <= (char_t)'z');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool IsSpace(char_t in)
{
    return (in == (char_t)' ' || in == (char_t)'\t');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool IsLineEnd(char_t in)
{
    return (in == (char_t)'\r' || in == (char_t)'\n' || in == (char_t)'\0' || in == (char_t)'\f');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool IsSpaceOrNewLine(char_t in)
{
    return IsSpace<char_t>(in) || IsLineEnd<char_t>(in);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool SkipSpaces(const char_t* in, const char_t** out)
{
    while (*in == (char_t)' ' || *in == (char_t)'\t')
    {
        ++in;
    }
    *out = in;
    return !IsLineEnd<char_t>(*in);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool SkipSpaces(const char_t** inout)
{
    return SkipSpaces<char_t>(*inout, inout);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool SkipLine(const char_t* in, const char_t** out)
{
    while (*in != (char_t)'\r' && *in != (char_t)'\n' && *in != (char_t)'\0')
    {
        ++in;
    }

    // files are opened in binary mode. Ergo there are both NL and CR
    while (*in == (char_t)'\r' || *in == (char_t)'\n')
    {
        ++in;
    }
    *out = in;
    return *in != (char_t)'\0';
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool SkipLine(const char_t** inout)
{
    return SkipLine<char_t>(*inout, inout);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool SkipSpacesAndLineEnd(const char_t* in, const char_t** out)
{
    while (*in == (char_t)' ' || *in == (char_t)'\t' || *in == (char_t)'\r' || *in == (char_t)'\n')
    {
        ++in;
    }
    *out = in;
    return *in != '\0';
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool SkipSpacesAndLineEnd(const char_t** inout)
{
    return SkipSpacesAndLineEnd<char_t>(*inout, inout);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool GetNextLine(const char_t*& buffer, char_t out[BufferSize])
{
    if ((char_t)'\0' == *buffer)
    {
        return false;
    }

    char* _out = out;
    char* const end = _out + BufferSize;
    while (!IsLineEnd(*buffer) && _out < end)
    {
        *_out++ = *buffer++;
    }
    *_out = (char_t)'\0';

    while (IsLineEnd(*buffer) && '\0' != *buffer)
    {
        ++buffer;
    }

    return true;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool IsNumeric(char_t in)
{
    return (in >= '0' && in <= '9') || '-' == in || '+' == in;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool TokenMatch(char_t*& in, const char* token, unsigned int len)
{
    if (!::strncmp(token, in, len) && IsSpaceOrNewLine(in[len]))
    {
        if (in[len] != '\0')
        {
            in += len + 1;
        }
        else
        {
            // If EOF after the token make sure we don't go past end of buffer
            in += len;
        }
        return true;
    }

    return false;
}
// ---------------------------------------------------------------------------------
/** @brief Case-ignoring version of TokenMatch
 *  @param in Input
 *  @param token Token to check for
 *  @param len Number of characters to check
 */
AI_FORCE_INLINE bool TokenMatchI(const char*& in, const char* token, unsigned int len)
{
    if (!ASSIMP_strincmp(token, in, len) && IsSpaceOrNewLine(in[len]))
    {
        in += len + 1;
        return true;
    }
    return false;
}
// ---------------------------------------------------------------------------------
AI_FORCE_INLINE void SkipToken(const char*& in)
{
    SkipSpaces(&in);
    while (!IsSpaceOrNewLine(*in))
        ++in;
}
// ---------------------------------------------------------------------------------
AI_FORCE_INLINE std::string GetNextToken(const char*& in)
{
    SkipSpacesAndLineEnd(&in);
    const char* cur = in;
    while (!IsSpaceOrNewLine(*in))
        ++in;
    return std::string(cur, (size_t)(in - cur));
}

// ---------------------------------------------------------------------------------

} // namespace Assimp

#endif // ! AI_PARSING_UTILS_H_INC
