#ifndef API_H
#define API_H

#include <Defines.h>

#if defined(_MSC_VER) || defined(__declspec)
#define BRAYNS_DLLEXPORT __declspec(dllexport)
#define BRAYNS_DLLIMPORT __declspec(dllimport)
#else // _MSC_VER
#define BRAYNS_DLLEXPORT
#define BRAYNS_DLLIMPORT
#endif // _MSC_VER

#if defined(BRAYNS_STATIC)
#define BRAYNS_API
#elif defined(BRAYNS_SHARED)
#define BRAYNS_API BRAYNS_DLLEXPORT
#else
#define BRAYNS_API BRAYNS_DLLIMPORT
#endif

#endif
