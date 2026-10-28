#include <features.h>

#if !__GLIBC_PREREQ(2, 38)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// Offline native-validation archives require C23 parsing symbols on older
// glibc. This shim belongs to that validator; production odometry neither
// links these archives nor loads vehicle-state JSON through PhysX.
extern "C" unsigned long __isoc23_strtoul(
  const char * text, char ** end, int base)
{
  return std::strtoul(text, end, base);
}

extern "C" unsigned long long __isoc23_strtoull(
  const char * text, char ** end, int base)
{
  return std::strtoull(text, end, base);
}

extern "C" int __isoc23_vsscanf(
  const char * text, const char * format, va_list args)
{
  return std::vsscanf(text, format, args);
}

extern "C" int __isoc23_fscanf(FILE * stream, const char * format, ...)
{
  va_list args;
  va_start(args, format);
  const int result = std::vfscanf(stream, format, args);
  va_end(args);
  return result;
}

#endif
