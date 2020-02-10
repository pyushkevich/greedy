#ifndef __SparseSolver_h_
#define __SparseSolver_h_

#include <iostream>
#include "SparseMatrix.h"

class SparseSolver
{
public:
  virtual ~SparseSolver() {}

  // Factor the system for arbitrary right hand sides and matrices of the same
  // non-zer element structure. THIS METHOD USED 1-BASED INDEXING!!!
  virtual void SymbolicFactorization(size_t n, int *idxRows, int *idxCols, double *xMatrix) = 0;

  // Perform symbolic factorization given a matrix
  virtual void SymbolicFactorization(const ImmutableSparseMatrix<double> &mat) = 0;

  // Factor the system for a specific matrix, but arbitrary right hand side
  virtual void NumericFactorization(const double *xMatrix) = 0;

  // Numeric factorization using sparse matrix datatype
  virtual void NumericFactorization(const ImmutableSparseMatrix<double> &mat) = 0;

  // Solve the system for the given right hand side, solution in xSoln
  virtual void Solve(double *xRhs, double *xSoln) = 0;

  // Solve the system for a number of right hand sides, if the second vector
  // is NULL, will solve in-place
  virtual void Solve(size_t nRHS, double *xRhs, double *xSoln) = 0;

  // Outut dumping
  virtual void SetVerbose(bool flag)
    { flagVerbose = flag; }

  // Factory method to generate solver based on system settings
  static SparseSolver *MakeSolver(bool symmetric);

protected:

  bool flagVerbose;
};


#endif
