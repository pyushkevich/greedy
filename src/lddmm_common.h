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
#ifndef _LDDMM_COMMON_H_
#define _LDDMM_COMMON_H_

typedef unsigned int uint;

typedef double myreal;

// A macro for defining named inputs and outputs to ITK filters
#define itkNamedInputMacro(name, type, key) \
  virtual void Set##name (type *_arg) \
    { \
    itk::ProcessObject::SetInput(key, _arg); \
    } \
  \
  virtual type * Get##name() \
    { \
    return dynamic_cast<type *>(itk::ProcessObject::GetInput(key)); \
    }

#define itkNamedInputGetMacro(name, type, key) \
  virtual type * Get##name() \
    { \
    return dynamic_cast<type *>(itk::ProcessObject::GetInput(key)); \
    }

// A macro for defining named inputs and outputs to ITK filters
#define itkNamedOutputMacro(name, type, key) \
  virtual void Set##name (type *_arg) \
    { \
    itk::ProcessObject::SetOutput(key, _arg); \
    } \
  \
  virtual type * Get##name() \
    { \
    return dynamic_cast<type *>(itk::ProcessObject::GetOutput(key)); \
    }

#define itkNamedOutputGetMacro(name, type, key) \
  virtual type * Get##name() \
    { \
    return dynamic_cast<type *>(itk::ProcessObject::GetOutput(key)); \
    }





#endif
