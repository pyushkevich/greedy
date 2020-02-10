#ifndef __TaucsInterface_h_
#define __TaucsInterface_h_

#include "SparseSolver.h"
#include "SparseMatrix.h"

extern "C" {
#include "taucs.h"
}

class TaucsSolverInterface : public SparseSolver
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
  
  // Constructor, takes the problem type
  TaucsSolverInterface(bool symmetric);

  // Destructor
  virtual ~TaucsSolverInterface();

protected:

  template <class TIndex> void DoSymbolicFactorization(
    size_t n, const TIndex *idxRows, const TIndex *idxCols, const double *xMatrix, int idxbase);

  void Deallocate();

  // Transpose of the input sparse matrix
  ImmutableSparseArray<int> At;

  // Sparse matrix in CCS format
  taucs_ccs_matrix ccs;

  // Factorization from taucs
  struct TaucsFactorization
    {
    int *perm, *invperm;
    void *lu;
    };
 
  // Factorization object
  TaucsFactorization fact;

  // Symmetric flag
  bool symmetric;
};

#endif //__TaucsInterface_h_
