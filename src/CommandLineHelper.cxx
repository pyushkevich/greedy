#include "CommandLineHelper.h"
#include <cassert>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <wordexp.h>
#endif

/**
 * Code to split a string into argc/argv with some shell expansion
 * from http://stackoverflow.com/questions/1706551/parse-string-into-argv-argc
 */
char **split_commandline(const char *cmdline, int *argc)
{
  int i;
  char **argv = NULL;
  assert(argc);

  if (!cmdline)
  {
    return NULL;
  }

// Posix.
#ifndef _WIN32
  {
    wordexp_t p;

    // Note! This expands shell variables.
    if (wordexp(cmdline, &p, 0))
    {
      return NULL;
    }

    *argc = p.we_wordc;

    if (!(argv = (char **) calloc(*argc, sizeof(char *))))
    {
      goto fail;
    }

    for (i = 0; i < p.we_wordc; i++)
    {
      if (!(argv[i] = strdup(p.we_wordv[i])))
      {
        goto fail;
      }
    }

    wordfree(&p);

    return argv;
  fail:
    wordfree(&p);
  }
#else // WIN32
  {
    wchar_t **wargs = NULL;
    size_t needed = 0;
    wchar_t *cmdlinew = NULL;
    size_t len = strlen(cmdline) + 1;

    if (!(cmdlinew = (wchar_t *) calloc(len, sizeof(wchar_t))))
      goto fail;

    if (!MultiByteToWideChar(CP_ACP, 0, cmdline, -1, cmdlinew, len))
      goto fail;

    if (!(wargs = CommandLineToArgvW(cmdlinew, argc)))
      goto fail;

    if (!(argv = (char **) calloc(*argc, sizeof(char *))))
      goto fail;

    // Convert from wchar_t * to ANSI char *
    for (i = 0; i < *argc; i++)
    {
      // Get the size needed for the target buffer.
      // CP_ACP = Ansi Codepage.
      needed = WideCharToMultiByte(CP_ACP, 0, wargs[i], -1,
                                   NULL, 0, NULL, NULL);

      if (!(argv[i] = (char *) malloc(needed)))
        goto fail;

      // Do the conversion.
      needed = WideCharToMultiByte(CP_ACP, 0, wargs[i], -1,
                                   argv[i], needed, NULL, NULL);
    }

    if (wargs) LocalFree(wargs);
    if (cmdlinew) free(cmdlinew);
    return argv;

  fail:
    if (wargs) LocalFree(wargs);
    if (cmdlinew) free(cmdlinew);
  }
#endif // WIN32

  if (argv)
  {
    for (i = 0; i < *argc; i++)
    {
      if (argv[i])
      {
        free(argv[i]);
      }
    }

    free(argv);
  }

  return NULL;
}

