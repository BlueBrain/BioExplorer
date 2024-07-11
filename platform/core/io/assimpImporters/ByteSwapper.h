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

/** @file Helper class tp perform various byte oder swappings
   (e.g. little to big endian) */
#ifndef AI_BYTESWAPPER_H_INC
#define AI_BYTESWAPPER_H_INC

#include <assimp/ai_assert.h>
#include <assimp/types.h>
#include <stdint.h>

#if _MSC_VER >= 1400
#include <stdlib.h>
#endif

namespace Assimp
{
// --------------------------------------------------------------------------------------
/** Defines some useful byte order swap routines.
 *
 * This is required to read big-endian model formats on little-endian machines,
 * and vice versa. Direct use of this class is DEPRECATED. Use #StreamReader
 * instead. */
// --------------------------------------------------------------------------------------
class ByteSwap
{
    ByteSwap() {}

public:
    // ----------------------------------------------------------------------
    /** Swap two bytes of data
     *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
    static inline void Swap2(void* _szOut)
    {
        ai_assert(_szOut);

#if _MSC_VER >= 1400
        uint16_t* const szOut = reinterpret_cast<uint16_t*>(_szOut);
        *szOut = _byteswap_ushort(*szOut);
#else
        uint8_t* const szOut = reinterpret_cast<uint8_t*>(_szOut);
        std::swap(szOut[0], szOut[1]);
#endif
    }

    // ----------------------------------------------------------------------
    /** Swap four bytes of data
     *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
    static inline void Swap4(void* _szOut)
    {
        ai_assert(_szOut);

#if _MSC_VER >= 1400
        uint32_t* const szOut = reinterpret_cast<uint32_t*>(_szOut);
        *szOut = _byteswap_ulong(*szOut);
#else
        uint8_t* const szOut = reinterpret_cast<uint8_t*>(_szOut);
        std::swap(szOut[0], szOut[3]);
        std::swap(szOut[1], szOut[2]);
#endif
    }

    // ----------------------------------------------------------------------
    /** Swap eight bytes of data
     *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
    static inline void Swap8(void* _szOut)
    {
        ai_assert(_szOut);

#if _MSC_VER >= 1400
        uint64_t* const szOut = reinterpret_cast<uint64_t*>(_szOut);
        *szOut = _byteswap_uint64(*szOut);
#else
        uint8_t* const szOut = reinterpret_cast<uint8_t*>(_szOut);
        std::swap(szOut[0], szOut[7]);
        std::swap(szOut[1], szOut[6]);
        std::swap(szOut[2], szOut[5]);
        std::swap(szOut[3], szOut[4]);
#endif
    }

    // ----------------------------------------------------------------------
    /** ByteSwap a float. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(float* fOut)
    {
        Swap4(fOut);
    }
    // ----------------------------------------------------------------------
    /** ByteSwap a double. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(double* fOut)
    {
        Swap8(fOut);
    }
    // ----------------------------------------------------------------------
    /** ByteSwap an int16t. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(int16_t* fOut)
    {
        Swap2(fOut);
    }
    static inline void Swap(uint16_t* fOut)
    {
        Swap2(fOut);
    }
    // ----------------------------------------------------------------------
    /** ByteSwap an int32t. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(int32_t* fOut)
    {
        Swap4(fOut);
    }
    static inline void Swap(uint32_t* fOut)
    {
        Swap4(fOut);
    }
    // ----------------------------------------------------------------------
    /** ByteSwap an int64t. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(int64_t* fOut)
    {
        Swap8(fOut);
    }
    static inline void Swap(uint64_t* fOut)
    {
        Swap8(fOut);
    }
    // ----------------------------------------------------------------------
    //! Templatized ByteSwap
    //! \returns param tOut as swapped
    template <typename Type>
    static inline Type Swapped(Type tOut)
    {
        return _swapper<Type, sizeof(Type)>()(tOut);
    }

private:
    template <typename T, size_t size>
    struct _swapper;
};

template <typename T>
struct ByteSwap::_swapper<T, 2>
{
    T operator()(T tOut)
    {
        Swap2(&tOut);
        return tOut;
    }
};

template <typename T>
struct ByteSwap::_swapper<T, 4>
{
    T operator()(T tOut)
    {
        Swap4(&tOut);
        return tOut;
    }
};

template <typename T>
struct ByteSwap::_swapper<T, 8>
{
    T operator()(T tOut)
    {
        Swap8(&tOut);
        return tOut;
    }
};

// --------------------------------------------------------------------------------------
// ByteSwap macros for BigEndian/LittleEndian support
// --------------------------------------------------------------------------------------
#if (defined AI_BUILD_BIG_ENDIAN)
#define AI_LE(t) (t)
#define AI_BE(t) ByteSwap::Swapped(t)
#define AI_LSWAP2(p)
#define AI_LSWAP4(p)
#define AI_LSWAP8(p)
#define AI_LSWAP2P(p)
#define AI_LSWAP4P(p)
#define AI_LSWAP8P(p)
#define LE_NCONST const
#define AI_SWAP2(p) ByteSwap::Swap2(&(p))
#define AI_SWAP4(p) ByteSwap::Swap4(&(p))
#define AI_SWAP8(p) ByteSwap::Swap8(&(p))
#define AI_SWAP2P(p) ByteSwap::Swap2((p))
#define AI_SWAP4P(p) ByteSwap::Swap4((p))
#define AI_SWAP8P(p) ByteSwap::Swap8((p))
#define BE_NCONST
#else
#define AI_BE(t) (t)
#define AI_LE(t) ByteSwap::Swapped(t)
#define AI_SWAP2(p)
#define AI_SWAP4(p)
#define AI_SWAP8(p)
#define AI_SWAP2P(p)
#define AI_SWAP4P(p)
#define AI_SWAP8P(p)
#define BE_NCONST const
#define AI_LSWAP2(p) ByteSwap::Swap2(&(p))
#define AI_LSWAP4(p) ByteSwap::Swap4(&(p))
#define AI_LSWAP8(p) ByteSwap::Swap8(&(p))
#define AI_LSWAP2P(p) ByteSwap::Swap2((p))
#define AI_LSWAP4P(p) ByteSwap::Swap4((p))
#define AI_LSWAP8P(p) ByteSwap::Swap8((p))
#define LE_NCONST
#endif

namespace Intern
{
// --------------------------------------------------------------------------------------------
template <typename T, bool doit>
struct ByteSwapper
{
    void operator()(T* inout) { ByteSwap::Swap(inout); }
};

template <typename T>
struct ByteSwapper<T, false>
{
    void operator()(T*) {}
};

// --------------------------------------------------------------------------------------------
template <bool SwapEndianess, typename T, bool RuntimeSwitch>
struct Getter
{
    void operator()(T* inout, bool le)
    {
#ifdef AI_BUILD_BIG_ENDIAN
        le = le;
#else
        le = !le;
#endif
        if (le)
        {
            ByteSwapper<T, (sizeof(T) > 1 ? true : false)>()(inout);
        }
        else
            ByteSwapper<T, false>()(inout);
    }
};

template <bool SwapEndianess, typename T>
struct Getter<SwapEndianess, T, false>
{
    void operator()(T* inout, bool /*le*/)
    {
        // static branch
        ByteSwapper<T, (SwapEndianess && sizeof(T) > 1)>()(inout);
    }
};
} // namespace Intern
} // namespace Assimp

#endif //!! AI_BYTESWAPPER_H_INC
