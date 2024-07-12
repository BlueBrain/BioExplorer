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

/** @file Definition of platform independent string workers:

   ASSIMP_itoa10
   ASSIMP_stricmp
   ASSIMP_strincmp

   These functions are not consistently available on all platforms,
   or the provided implementations behave too differently.
*/
#ifndef INCLUDED_AI_STRING_WORKERS_H
#define INCLUDED_AI_STRING_WORKERS_H

#include "StringComparison.h"
#include <assimp/ai_assert.h>

#include <stdint.h>
#include <string.h>
#include <string>

namespace Assimp
{
// -------------------------------------------------------------------------------
/** @brief itoa with a fixed base 10
 * 'itoa' is not consistently available on all platforms so it is quite useful
 * to have a small replacement function here. No need to use a full sprintf()
 * if we just want to print a number ...
 * @param out Output buffer
 * @param max Maximum number of characters to be written, including '\0'.
 *   This parameter may not be 0.
 * @param number Number to be written
 * @return Length of the output string, excluding the '\0'
 */
inline unsigned int ASSIMP_itoa10(char *out, unsigned int max, int32_t number)
{
    ai_assert(NULL != out);

    // write the unary minus to indicate we have a negative number
    unsigned int written = 1u;
    if (number < 0 && written < max)
    {
        *out++ = '-';
        ++written;
        number = -number;
    }

    // We begin with the largest number that is not zero.
    int32_t cur = 1000000000; // 2147483648
    bool mustPrint = false;
    while (written < max)
    {
        const unsigned int digit = number / cur;
        if (mustPrint || digit > 0 || 1 == cur)
        {
            // print all future zeroes from now
            mustPrint = true;

            *out++ = '0' + static_cast<char>(digit);

            ++written;
            number -= digit * cur;
            if (1 == cur)
            {
                break;
            }
        }
        cur /= 10;
    }

    // append a terminal zero
    *out++ = '\0';
    return written - 1;
}

// -------------------------------------------------------------------------------
/** @brief itoa with a fixed base 10 (Secure template overload)
 *  The compiler should choose this function if he or she is able to determine
 * the
 *  size of the array automatically.
 */
template <size_t length>
inline unsigned int ASSIMP_itoa10(char (&out)[length], int32_t number)
{
    return ASSIMP_itoa10(out, length, number);
}

// -------------------------------------------------------------------------------
/** @brief Helper function to do platform independent string comparison.
 *
 *  This is required since stricmp() is not consistently available on
 *  all platforms. Some platforms use the '_' prefix, others don't even
 *  have such a function.
 *
 *  @param s1 First input string
 *  @param s2 Second input string
 *  @return 0 if the given strings are identical
 */
inline int ASSIMP_stricmp(const char *s1, const char *s2)
{
    ai_assert(NULL != s1 && NULL != s2);

#if (defined _MSC_VER)

    return ::_stricmp(s1, s2);
#elif defined(__GNUC__)

    return ::strcasecmp(s1, s2);
#else

    char c1, c2;
    do
    {
        c1 = tolower(*s1++);
        c2 = tolower(*s2++);
    } while (c1 && (c1 == c2));
    return c1 - c2;
#endif
}

// -------------------------------------------------------------------------------
/** @brief Case independent comparison of two std::strings
 *
 *  @param a First  string
 *  @param b Second string
 *  @return 0 if a == b
 */
inline int ASSIMP_stricmp(const std::string &a, const std::string &b)
{
    int i = (int)b.length() - (int)a.length();
    return (i ? i : ASSIMP_stricmp(a.c_str(), b.c_str()));
}

// -------------------------------------------------------------------------------
/** @brief Helper function to do platform independent string comparison.
 *
 *  This is required since strincmp() is not consistently available on
 *  all platforms. Some platforms use the '_' prefix, others don't even
 *  have such a function.
 *
 *  @param s1 First input string
 *  @param s2 Second input string
 *  @param n Macimum number of characters to compare
 *  @return 0 if the given strings are identical
 */
inline int ASSIMP_strincmp(const char *s1, const char *s2, unsigned int n)
{
    ai_assert(NULL != s1 && NULL != s2);
    if (!n)
        return 0;

#if (defined _MSC_VER)

    return ::_strnicmp(s1, s2, n);

#elif defined(__GNUC__)

    return ::strncasecmp(s1, s2, n);

#else
    char c1, c2;
    unsigned int p = 0;
    do
    {
        if (p++ >= n)
            return 0;
        c1 = tolower(*s1++);
        c2 = tolower(*s2++);
    } while (c1 && (c1 == c2));

    return c1 - c2;
#endif
}

// -------------------------------------------------------------------------------
/** @brief Evaluates an integer power
 *
 * todo: move somewhere where it fits better in than here
 */
inline unsigned int integer_pow(unsigned int base, unsigned int power)
{
    unsigned int res = 1;
    for (unsigned int i = 0; i < power; ++i)
        res *= base;

    return res;
}
} // namespace Assimp

#endif // !  AI_STRINGCOMPARISON_H_INC
