#include "PardisoInterface.h"
#include <iostream>
#include <cstdlib>
#include <cstring>


using namespace std;



GenericRealPARDISO::GenericRealPARDISO(int type)
{

#ifdef PARDISO_DYNLOAD

#ifdef WIN32

  // Load the DLL (hardcoding the name for now)
  hLib = LoadLibrary("libpardiso_GNU_MINGW32.dll");
  if(hLib==NULL) 
    throw std::exception("Unable to load PARDISO library libpardiso_GNU_MINGW32.dll!");

  // Print the library name
  // char mod[512];
  // GetModuleFileName((HMODULE)hLib, (LPTSTR)mod, 512);
  // cout << "Library loaded: " << mod << endl;

  // Get addresses of functions
  pardisoinit_ = (PardisoInitFunc) GetProcAddress((HMODULE)hLib, "pardisoinit_");
  pardiso_ = (PardisoMainFunc) GetProcAddress((HMODULE)hLib, "pardiso_");
  
  // Check that the functions are real
  if(pardisoinit_ == NULL || pardiso_ == NULL)
    throw std::exception("Unable to load functions from PARDISO DLL");  

#endif // WIN32

#endif // PARDISO_DYNLOAD


  // Set the type of matrix to unsymmetric real
  MTYPE = type; 
  SOLVER = 0;
  int ERROR = 0;

  // Clear the parameter array
  memset(IPARM, 0, sizeof(int) * 64);

  // Initialize PARDISO to default values
  pardisoinit_(PT,&MTYPE,&SOLVER,IPARM,DPARM,&ERROR);
  switch(ERROR)
    {
    case -10 : cerr << "No PARDISO license file found" << endl; exit(-1);
    case -11 : cerr << "PARDISO license is expired" << endl; exit(-1);
    case -12 : cerr << "PARDISO license username mismatch" << endl; exit(-1);
    }

  // Specify the number of processors on the system (1)
  IPARM[0] = 0;
  IPARM[2] = 1;

  flagPardisoCalled = false;

  // Whether the index arrays are owned
  flagOwnIndexArrays = false;

  flagVerbose = false;
}


void 
GenericRealPARDISO
::ResetIndices()
{
  if(flagOwnIndexArrays)
    { delete idxRows; delete idxCols; }
  flagOwnIndexArrays = false;
}

void 
GenericRealPARDISO
::SymbolicFactorization(size_t n, int *idxRows, int *idxCols, double *xMatrix)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 11, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Perform the symbolic factorization phase
  pardiso_(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    xMatrix, idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR, DPARM);

  // Record the parameter for next phase
  ResetIndices();
  this->idxCols = idxCols;
  this->idxRows = idxRows;
  this->n = n;

  // Set the flag so we know that pardiso was launched before
  flagPardisoCalled = true;
}

void 
GenericRealPARDISO
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
  pardiso_(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(mat.GetSparseData()), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR, DPARM);

  // Set the flag so we know that pardiso was launched before
  flagPardisoCalled = true;
}

void 
GenericRealPARDISO
::NumericFactorization(const double *xMatrix)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 22, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Perform the symbolic factorization phase
  pardiso_(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(xMatrix), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR, DPARM);

  // Record the parameter for next phase
  this->xMatrix = xMatrix;
}

void 
GenericRealPARDISO
::Solve(double *xRhs, double *xSoln)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 33, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 

  // Perform the symbolic factorization phase
  pardiso_(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(xMatrix), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, xRhs, xSoln, &PERROR, DPARM);
}

void                              
GenericRealPARDISO
::Solve(size_t nRHS, double *xRhs, double *xSoln)
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 33, N = n, NRHS = nRHS, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 

  IPARM[7] = 1;
  
  // Perform the symbolic factorization phase
  pardiso_(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    const_cast<double *>(xMatrix), idxRows, idxCols,
    NULL, &NRHS, IPARM, &MSGLVL, xRhs, xSoln, &PERROR, DPARM);
}

GenericRealPARDISO::
~GenericRealPARDISO()
{
  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = -1, N = n, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Deallocate data
  if(flagPardisoCalled)
    pardiso_(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
      const_cast<double *>(xMatrix), idxRows, idxCols,
      NULL, &NRHS, IPARM, &MSGLVL, NULL, NULL, &PERROR, DPARM);

  // Reset the arrays
  ResetIndices();

#if defined(PARDISO_DYNLOAD) && defined(WIN32)
  FreeLibrary((HMODULE)hLib);
#endif
}

  // Initialize the solver 
UnsymmetricRealPARDISO::UnsymmetricRealPARDISO()
: GenericRealPARDISO(11) 
{
  // Small test
  int ia[] = {1,5,8,10,12,13,16,18,21};
  int ja[] = {1,3,6,7,2,3,5,3,8,4,7,2,3,6,8,2,7,3,7,8};
  double val[] = {7.,1.,2.,7.,-4.,8.,2.,1.,5.,7.,9.,-4.,7.,3.,5.,17.,11.,-3.,2.,5.};
  double rhs[] = {6.,1.,0.,6.,4.,9.,1.,9.};
  double sol[] = {0.,0.,0.,0.,0.,0.,0.,0.};

  /* 

  // Set the various parameters
  int MAXFCT = 1, MNUM = 1, PHASE = 13, N = 8, NRHS = 1, PERROR = 0; 
  int MSGLVL = (flagVerbose) ? 1 : 0; 
  
  // Perform the symbolic factorization phase
  pardiso_(PT, &MAXFCT, &MNUM, &MTYPE, &PHASE, &N, 
    val, ia, ja,
    NULL, &NRHS, IPARM, &MSGLVL, rhs, sol, &PERROR);

  // Check result
  if(PERROR == 0 && fabs(sol[0] + 2.25) < 1.0e-6)
    cout << "PARDISO library initialized and tested" << endl;
  else
    throw exception("PARDISO library unable to solve test problem");

  */
};

