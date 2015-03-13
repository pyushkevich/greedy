#ifndef _LDDMM_COMMON_H_
#define _LDDMM_COMMON_H_

typedef unsigned int uint;

typedef double myreal;

// A macro for defining named inputs and outputs to ITK filters
#define itkNamedInputMacro(name, type, key) \
  virtual void Set##name (type *_arg) \
    { \
    this->itk::ProcessObject::SetInput(key, _arg); \
    } \
  \
  virtual type * Get##name() \
    { \
    return dynamic_cast<type *>(this->ProcessObject::GetInput(key)); \
    }

// A macro for defining named inputs and outputs to ITK filters
#define itkNamedOutputMacro(name, type, key) \
  virtual void Set##name (type *_arg) \
    { \
    this->itk::ProcessObject::SetOutput(key, _arg); \
    } \
  \
  virtual type * Get##name() \
    { \
    return dynamic_cast<type *>(this->ProcessObject::GetOutput(key)); \
    }

#endif
