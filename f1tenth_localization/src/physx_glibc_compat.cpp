#include <features.h>

#if !__GLIBC_PREREQ(2, 38)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// PhysX 4.1's checked-in Linux archives were built against a newer glibc and
// reference the C23 parsing entry points. The Odom runtime loads decimal
// vehicle-state JSON only, so Ubuntu 22.04 can provide the legacy parser
// implementations at this ABI boundary without changing the PhysX equations.
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
