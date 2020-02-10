#include "TaucsInterface.h"
#include "MedialException.h"

void
TaucsSolverInterface
::Deallocate()
{
  // Delete data in ccs
  if(ccs.values.d)
    {
    delete[] ccs.colptr;
    delete[] ccs.rowind;
    delete[] ccs.values.d;
    ccs.values.d = NULL;
    }

  // Delete other resources
  if(fact.perm)
    {
    free(fact.perm);
    free(fact.invperm);
    fact.perm = NULL;
    }
}

template <class TIndex>
void
TaucsSolverInterface
::DoSymbolicFactorization(size_t n, const TIndex *idxRows, const TIndex *idxCols, const double *xMatrix, int idxbase)
{
  // Reset resources
  this->Deallocate();

  // Initialize the transpose matrix in STL format
  ImmutableSparseArray<int>::STLSourceType src;
  src.resize(n);
  for(size_t r = 0; r < n; r++)
    for(size_t ic = (size_t) idxRows[r] - idxbase; ic < (size_t) idxRows[r+1] - idxbase; ic++)
      src[idxCols[ic] - idxbase].push_back(
        ImmutableSparseArray<int>::STLEntryType(r, ic));

  // Generate sparse matrix from STL matrix
  At.SetFromSTL(src, n);

  // Initialize the TAUCS data structure
  ccs.n = n; 
  ccs.m = n;
  ccs.colptr = new int[At.GetNumberOfRows() + 1];
  ccs.rowind = new int[At.GetNumberOfSparseValues()];
  ccs.values.d = new double[At.GetNumberOfSparseValues()];
  ccs.flags = symmetric ? TAUCS_DOUBLE | TAUCS_SYMMETRIC : TAUCS_DOUBLE;

  for(size_t i = 0; i <= At.GetNumberOfRows(); i++)
    ccs.colptr[i] = At.GetRowIndex()[i];

  for(size_t i = 0; i < At.GetNumberOfSparseValues(); i++)
    ccs.rowind[i] = At.GetColIndex()[i];
}

void 
TaucsSolverInterface
::SymbolicFactorization(size_t n, int *idxRows, int *idxCols, double *xMatrix)
{
  this->DoSymbolicFactorization(n, idxRows, idxCols, xMatrix, 1);
}

void 
TaucsSolverInterface
::SymbolicFactorization(const ImmutableSparseMatrix<double> &mat)
{
  this->DoSymbolicFactorization(
    mat.GetNumberOfRows(),  mat.GetRowIndex(), mat.GetColIndex(), mat.GetSparseData(), 0); 
}

void 
TaucsSolverInterface
::NumericFactorization(const double *xMatrix)
{
  // taucs_logfile("taucs_log");

  // Copy the values into the transpose matrix
  for(size_t i = 0; i < At.GetNumberOfSparseValues(); i++)
    ccs.values.d[i] = xMatrix[At.GetSparseData()[i]];

  // Clear fact if it's used

  // Perform reordering
  taucs_ccs_order(&ccs, &fact.perm, &fact.invperm, "colamd"); 

  // Do factorization
  int rc = taucs_ooc_factor_lu(&ccs, fact.perm, (taucs_io_handle *) fact.lu, 
    taucs_available_memory_size() / 16);

  if(rc)
    throw MedialModelException("non-zero return code in taucs_ooc_factor_lu");
}

void 
TaucsSolverInterface
::Solve(double *xRhs, double *xSoln)
{
  // taucs_linsolve(&ccs, &fact, 1, xSoln, xRhs, options_factor, NULL);
  taucs_ooc_solve_lu((taucs_io_handle *) fact.lu, xSoln, xRhs);
}

void
TaucsSolverInterface
::Solve(size_t nRHS, double *xRhs, double *xSoln)
{
  for(size_t i = 0; i < nRHS; i++)
    {
    taucs_ooc_solve_lu((taucs_io_handle *) fact.lu, xSoln, xRhs);
    xSoln += ccs.n;
    xRhs += ccs.n;
    }
}

TaucsSolverInterface
::TaucsSolverInterface(bool symmetric)
{
  this->symmetric = symmetric;
  ccs.values.d = NULL;
  fact.perm = NULL;

  // Create tempfile for output
  char *matfile = tempnam(NULL, "taucs.L");
  fact.lu = taucs_io_create_multifile(matfile);
  free(matfile);
}

TaucsSolverInterface
::~TaucsSolverInterface()
{
  this->Deallocate();
  taucs_io_delete((taucs_io_handle *) fact.lu);
}
