#ifndef __MKLSolverInterface_h_
#define __MKLSolverInterface_h_

#include <iostream>
#include "SparseSolver.h"
#include "SparseMatrix.h"

#ifdef WIN32
#include <windows.h>
#endif

// BLAS/PARDISO references
#ifdef HAVE_MKL

#include <mkl_pardiso.h>

#endif


class MKLSolverInterface : public SparseSolver
{
public:
  // Factor the system for arbitrary right hand sides and matrices of the same
  // non-zer element structure
  void SymbolicFactorization(size_t n, int *idxRows, int *idxCols, double *xMatrix);

  // Perform symbolic factorization given a matrix
  void SymbolicFactorization(const ImmutableSparseMatrix<double> &mat);

  // Factor the system for a specific matrix, but arbitrary right hand side
  void NumericFactorization(const double *xMatrix);

  // Numeric factorization using sparse matrix datatype
  void NumericFactorization(const ImmutableSparseMatrix<double> &mat)
    { NumericFactorization(mat.GetSparseData()); }

  // Solve the system for the given right hand side, solution in xSoln
  void Solve(double *xRhs, double *xSoln);

  // Solve the system for a number of right hand sides, if the second vector
  // is NULL, will solve in-place
  void Solve(size_t nRHS, double *xRhs, double *xSoln);

  // Outut dumping
  void SetVerbose(bool flag)
    { flagVerbose = flag; }

  enum ProblemType { SPD = 2, UNSYMMETRIC = 11 };

  // Constructor, takes the problem type
  MKLSolverInterface(ProblemType mtype);
  
  // Destructor
  virtual ~MKLSolverInterface();

protected:

  // Reset the index arrays()
  void ResetIndices();

  /** Internal data for PARDISO */
  int PT[64];
  int MTYPE;
  int IPARM[64];

  // Storage for data in intermediate steps
  int n, *idxRows, *idxCols;
  const double *xMatrix;
  bool flagPardisoCalled, flagOwnIndexArrays;
};

#endif //__MKLSolverInterface_h_
