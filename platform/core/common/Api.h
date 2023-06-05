#ifndef API_H
#define API_H

#include <Defines.h>

#if defined(_MSC_VER) || defined(__declspec)
#define CORE_DLLEXPORT __declspec(dllexport)
#define CORE_DLLIMPORT __declspec(dllimport)
#else // _MSC_VER
#define CORE_DLLEXPORT
#define CORE_DLLIMPORT
#endif // _MSC_VER

#if defined(CORE_STATIC)
#define PLATFORM_API
#elif defined(CORE_SHARED)
#define PLATFORM_API CORE_DLLEXPORT
#else
#define PLATFORM_API CORE_DLLIMPORT
#endif

#endif
