#ifndef __PardisoInterface_h_
#define __PardisoInterface_h_

#include <iostream>
#include "SparseSolver.h"
#include "SparseMatrix.h"

#ifdef WIN32
#include <windows.h>
#endif

// BLAS/PARDISO references
#ifdef HAVE_PARDISO

#ifdef PARDISO_DYNLOAD

  // Declarations for dynamic loading (tested in Win32 only)
  typedef void (__cdecl *PardisoInitFunc)(size_t *, int *, int *);
  typedef void (__cdecl *PardisoMainFunc)(size_t *, int *, int *, int *, 
    int *, int *, double *, int *, int *, 
    int *, int *, int *, int *, double *, double *, int*);

#else

extern "C" {
  void pardisoinit_ (void * , int * , int * , int * , double * , int *);

  void pardiso_ ( void * , int * , int * , int * , int * , int * ,
    double * , int * , int * , int * , int * , int * ,
    int * , double * , double * , int * , double *);
}

#endif

#endif


class GenericRealPARDISO : public SparseSolver
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

protected:
  
  // Constructor, takes the problem type
  GenericRealPARDISO(int mtype);

  // Destructor
  virtual ~GenericRealPARDISO();

  // Reset the index arrays()
  void ResetIndices();

  /** Function pointers used to access PARDISO */
#ifdef PARDISO_DYNLOAD
#ifdef WIN32
  HINSTANCE hLib;
#endif

  PardisoInitFunc pardisoinit_;
  PardisoMainFunc pardiso_;
#endif

  /** Internal data for PARDISO */
  void *PT[64];
  int MTYPE;
  int SOLVER;
  int IPARM[64];
  double DPARM[64];

  // Storage for data in intermediate steps
  int n, *idxRows, *idxCols;
  const double *xMatrix;
  bool flagPardisoCalled, flagOwnIndexArrays;
};

class UnsymmetricRealPARDISO : public GenericRealPARDISO
{
public:
  // Initialize the solver 
  UnsymmetricRealPARDISO();
};

class SymmetricPositiveDefiniteRealPARDISO : public GenericRealPARDISO
{
public:
  // Initialize the solver 
  SymmetricPositiveDefiniteRealPARDISO() : GenericRealPARDISO(2) {};
};

#endif //__PardisoInterface_h_
