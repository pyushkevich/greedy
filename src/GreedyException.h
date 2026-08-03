/*=========================================================================

  Program:   GreedyReg fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  GreedyReg initial development was funded by the NIH grant R01 EB017255.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

=========================================================================*/
#ifndef __GreedyException_h_
#define __GreedyException_h_

#include <cstdio>
#include <cstdarg>
#include <exception>

/**
 * A simple exception class with string formatting
 */
class GreedyException : public std::exception
{
public:

  GreedyException(const char *format, ...)
  {
    buffer = new char[4096];
    va_list args;
    va_start (args, format);
    vsnprintf (buffer, 4096, format, args);
    va_end (args);
  }

  virtual const char* what() const throw() { return buffer; }

  virtual ~GreedyException() throw() { delete buffer; }

  static void check(bool condition, const char *format, ...)
  {
    if(!condition)
    {
      char buffer[4096];
      va_list args;
      va_start (args, format);
      vsnprintf (buffer, 4096, format, args);
      va_end (args);
      throw GreedyException(buffer);
    }
  }

private:

  char *buffer;

};


#endif
