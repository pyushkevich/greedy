#ifndef __SparseSolverException_h_
#define __SparseSolverException_h_

#include <exception>
#include <stdexcept>

// Exception classes
class SparseSolverModelException : public std::runtime_error 
{
public:
  SparseSolverModelException(const char *text) : std::runtime_error(text) {}
};

class ModelIOException : public SparseSolverModelException 
{
public:
  ModelIOException(const char *text) : SparseSolverModelException(text) {}
};

#endif // __SparseSolverException_h_
