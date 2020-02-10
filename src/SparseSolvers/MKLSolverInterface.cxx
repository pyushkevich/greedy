#include "MKLSolverInterface.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>


using namespace std;



MKLSolverInterface::MKLSolverInterface(ProblemType type)
{
  // Set the type of matrix to unsymmetric real
  MTYPE = (int) type; 

  // Clear the parameter array
  memset(IPARM, 0, sizeof(int) * 64);

  // Initialize PARDISO to default values
  memset(PT, 0, sizeof(int) * 64);

  // Whether we have called PARDISO yet
  flagPardisoCalled = false;

  // Whether the index arrays are owned
  flagOwnIndexArrays = false;

  flagVerbose = false;
}


void 
MKLSolverInterface
::ResetIndices()
{
  if(flagOwnIndexArrays)
    { delete idxRows; delete idxCols; }
  flagOwnIndexArrays = false;
}

void 
MKLSolverInterface
::SymbolicFactorization(size_t n, int *idxRows, int *idxCols, double *xMatrix)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 11, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Perform the symbolic factorization phase
  pardiso(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    xMatrix, idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR);

  // Record the parameter for next phase
  ResetIndices();
  this->idxCols = idxCols;
  this->idxRows = idxRows;
  this->n = n;

  // Set the flag so we know that pardiso was launched before
  flagPardisoCalled = true;
}

void 
MKLSolverInterface
::SymbolicFactorization(const ImmutableSparseMatrix<double> &mat)
{
  // Init the index arrays
  ResetIndices();

  // We are going to own the indices
  flagOwnIndexArrays = true;

  // Record the parameter for next phase
  this->n = mat.GetNumberOfRows();

  // The arrays have to be incremented by one before calling PARDISO
  idxRows = new int[mat.GetNumberOfRows() + 1];
  for(size_t i = 0; i <= mat.GetNumberOfRows(); i++)
    idxRows[i] = (int)(1 + mat.GetRowIndex()[i]);

  idxCols = new int[mat.GetNumberOfSparseValues()];
  for(size_t j = 0; j < mat.GetNumberOfSparseValues(); j++)
    idxCols[j] = (int)(1 + mat.GetColIndex()[j]);
  
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 11, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 

  // Perform the symbolic factorization phase
  pardiso(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(mat.GetSparseData()), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR);

  if(PERROR)
    printf("PARDISO phase %d, error %d\n", PHASE, PERROR);

  // Set the flag so we know that pardiso was launched before
  flagPardisoCalled = true;
}

void 
MKLSolverInterface
::NumericFactorization(const double *xMatrix)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 22, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Perform the symbolic factorization phase
  pardiso(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(xMatrix), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR);

  if(PERROR)
    printf("PARDISO phase %d, error %d\n", PHASE, PERROR);
  
  // Record the parameter for next phase
  this->xMatrix = xMatrix;
}

void 
MKLSolverInterface
::Solve(double *xRhs, double *xSoln)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 33, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 

  // Perform the symbolic factorization phase
  pardiso(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(xMatrix), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, xRhs, xSoln, &PERROR);

  if(PERROR)
    printf("PARDISO phase %d, error %d\n", PHASE, PERROR);
}

void                              
MKLSolverInterface
::Solve(size_t nRHS, double *xRhs, double *xSoln)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 33, N = n, NRHS = nRHS, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Perform the symbolic factorization phase
  pardiso(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(xMatrix), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, xRhs, xSoln, &PERROR);

  if(PERROR)
    printf("PARDISO phase %d, error %d\n", PHASE, PERROR);
}

MKLSolverInterface::
~MKLSolverInterface()
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = -1, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Deallocate data
  if(flagPardisoCalled)
    pardiso(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
      const_cast<double *>(xMatrix), idxRows, idxCols,
      NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR);

  // Reset the arrays
  ResetIndices();
}

