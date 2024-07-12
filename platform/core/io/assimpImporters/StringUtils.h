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
#ifndef INCLUDED_AI_STRINGUTILS_H
#define INCLUDED_AI_STRINGUTILS_H

#include <cstdlib>
#include <sstream>
#include <stdarg.h>

///	@fn		ai_snprintf
///	@brief	The portable version of the function snprintf ( C99 standard ),
/// which works on visual studio compilers 2013 and earlier.
///	@param	outBuf		The buffer to write in
///	@param	size		The buffer size
///	@param	format		The format string
///	@param	ap			The additional arguments.
///	@return	The number of written characters if the buffer size was big enough.
/// If an encoding error occurs, a negative number is returned.
#if defined(_MSC_VER) && _MSC_VER < 1900

inline int c99_ai_vsnprintf(char *outBuf, size_t size, const char *format, va_list ap)
{
    int count(-1);
    if (0 != size)
    {
        count = _vsnprintf_s(outBuf, size, _TRUNCATE, format, ap);
    }
    if (count == -1)
    {
        count = _vscprintf(format, ap);
    }

    return count;
}

inline int ai_snprintf(char *outBuf, size_t size, const char *format, ...)
{
    int count;
    va_list ap;

    va_start(ap, format);
    count = c99_ai_vsnprintf(outBuf, size, format, ap);
    va_end(ap);

    return count;
}

#else
#define ai_snprintf snprintf
#endif

template <typename T>
inline std::string to_string(T value)
{
    std::ostringstream os;
    os << value;
    return os.str();
}

inline float ai_strtof(const char *begin, const char *end)
{
    if (nullptr == begin)
    {
        return 0.0f;
    }
    float val(0.0f);
    if (nullptr == end)
    {
        val = static_cast<float>(::atof(begin));
    }
    else
    {
        std::string::size_type len(end - begin);
        std::string token(begin, len);
        val = static_cast<float>(::atof(token.c_str()));
    }

    return val;
}

#endif // INCLUDED_AI_STRINGUTILS_H
