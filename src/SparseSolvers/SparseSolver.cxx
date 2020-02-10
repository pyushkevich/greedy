#include "SparseSolver.h"
#include "SparseSolverException.h"

#ifdef HAVE_PARDISO

#include "PardisoInterface.h"

SparseSolver* 
SparseSolver
::MakeSolver(bool symmetric)
{
  if(symmetric)
    return new SymmetricPositiveDefiniteRealPARDISO();
  else 
    return new UnsymmetricRealPARDISO();
}

#elif HAVE_TAUCS

#include "TaucsInterface.h"

SparseSolver* 
SparseSolver
::MakeSolver(bool symmetric)
{
  return new TaucsSolverInterface(symmetric);
}

#elif HAVE_MKL

#include "MKLSolverInterface.h"

SparseSolver* 
SparseSolver
::MakeSolver(bool symmetric)
{
  return new MKLSolverInterface(symmetric ? 
    MKLSolverInterface::SPD : 
    MKLSolverInterface::UNSYMMETRIC);
}




#else

SparseSolver* 
SparseSolver
::MakeSolver(bool symmetric)
{
  throw MedialModelException("The sparse solver has not been configured. Use PARDISO or TAUCS");
}

#endif

