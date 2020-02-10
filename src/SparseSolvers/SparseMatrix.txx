#include <algorithm>
#include <cassert>
#include <cstdio>

template<class TVal>
ImmutableSparseArray<TVal>
::ImmutableSparseArray()
{
  xSparseValues = NULL;
  xRowIndex = NULL;
  xColIndex = NULL;
  nRows = nColumns = nSparseEntries = 0;
}

template<class TVal>
ImmutableSparseArray<TVal>
::~ImmutableSparseArray()
{
  Reset();
}

template<class TVal>
void
ImmutableSparseArray<TVal>
::Reset()
{
  nRows = nColumns = nSparseEntries = 0;
  if(xSparseValues)
    { 
    delete[] xSparseValues; 
    delete[] xRowIndex; 
    delete[] xColIndex; 
    xSparseValues = NULL;
    xRowIndex = xColIndex = NULL;
    }
}

template<class TVal>
void
ImmutableSparseArray<TVal>
::SetFromSTL(STLSourceType &src, size_t nColumns)
{
  size_t i;
  
  // Delete the sparse matrix storage if it exists
  Reset();

  // Set the number of rows and columns
  this->nRows = src.size();
  this->nColumns = nColumns;

  // Allocate the row index (number of rows + 1)
  xRowIndex = new size_t[nRows + 1];

  // Fill the row index with indices into the sparse data
  xRowIndex[0] = 0;
  for(i = 0; i < nRows; i++)
    xRowIndex[i+1] = xRowIndex[i] + src[i].size();

  // Set the number of non-zero elements
  nSparseEntries = xRowIndex[nRows];

  // Initialize the data and column index arrays
  xColIndex = new size_t[nSparseEntries];
  xSparseValues = new TVal[nSparseEntries];

  // Fill the arrays
  size_t k = 0;
  for(i = 0; i < nRows; i++)
    {
    typename STLRowType::iterator it = src[i].begin();
    for(; it != src[i].end(); ++it, k++)
      {
      xColIndex[k] = it->first;
      xSparseValues[k] = it->second;
      }
    }

}

template<class TVal>
void
ImmutableSparseArray<TVal>
::SetFromVNL(VNLSourceType &src)
{
  size_t i, j;
  
  // Delete the sparse matrix storage if it exists
  Reset();

  // Set the number of rows and columns
  nRows = src.rows();
  nColumns = src.columns();

  // Allocate the row index (number of rows + 1)
  xRowIndex = new size_t[src.rows() + 1];

  // Fill the row index with indices into the sparse data
  xRowIndex[0] = 0;
  for(i = 0; i < src.rows(); i++)
    xRowIndex[i+1] = xRowIndex[i] + src.get_row(i).size();

  // Set the number of non-zero elements
  nSparseEntries = xRowIndex[src.rows()];

  // Initialize the data and column index arrays
  xColIndex = new size_t[nSparseEntries];
  xSparseValues = new TVal[nSparseEntries];

  // Fill the arrays
  size_t k = 0;
  for(i = 0; i < src.rows(); i++)
    {
    typename VNLSourceType::row &r = src.get_row(i);
    for(j = 0; j < r.size(); j++, k++)
      {
      xColIndex[k] = r[j].first;
      xSparseValues[k] = r[j].second;
      }
    }
}

template<class TVal>
bool
ImmutableSparseMatrix<TVal>
::operator == (const Self &B)
{
  // Metastructure must match
  if(this->nColumns != B.nColumns || this->nRows != B.nRows 
    || this->nSparseEntries != B.nSparseEntries)
    return false;

  // Compare row indices, etc
  for(size_t i = 0; i < this->nRows; i++)
    {
    // Row size must match
    if(this->xRowIndex[i+1] != B.xRowIndex[i+1])
      return false;
    
    // Column entries and values must match
    for(size_t j = this->xRowIndex[i]; j < this->xRowIndex[i+1]; j++)
      if(this->xColIndex[j] != B.xColIndex[j] ||
        this->xSparseValues[j] != B.xSparseValues[j])
        return false;
    }

  return true;
}

template<class TVal>
ImmutableSparseArray<TVal>::ImmutableSparseArray(const ImmutableSparseArray<TVal> &src)
{
  // Make copies of all non-arrays
  nRows = src.nRows;
  nColumns = src.nColumns;
  nSparseEntries = src.nSparseEntries;

  // If the source object is NULL, there is nothing to do
  if(src.xSparseValues == NULL) 
  {
    xSparseValues = NULL;
    xRowIndex = NULL;
    xColIndex = NULL;
    return;
  }

  // Allocate the arrays
  xRowIndex = new size_t[nRows + 1];
  xColIndex = new size_t[nSparseEntries];
  xSparseValues = new TVal[nSparseEntries];

  // Copy the array contennts
  std::copy(src.xRowIndex, src.xRowIndex + nRows + 1, xRowIndex);
  std::copy(src.xColIndex, src.xColIndex + nSparseEntries, xColIndex);
  std::copy(src.xSparseValues, src.xSparseValues + nSparseEntries, xSparseValues);
}

template<class TVal>
ImmutableSparseArray<TVal> &
ImmutableSparseArray<TVal>::operator= (const ImmutableSparseArray<TVal> &src)
{
  // Check if this is the same object (or both are reset)
  if(xSparseValues == src.xSparseValues)
    return *this;

  // Clear all the data
  Reset();

  // If the source object is NULL, there is nothing to do
  if(src.xSparseValues == NULL) return *this;

  // Make copies of all non-arrays
  nRows = src.nRows;
  nColumns = src.nColumns;
  nSparseEntries = src.nSparseEntries;

  // Allocate the arrays
  xRowIndex = new size_t[nRows + 1];
  xColIndex = new size_t[nSparseEntries];
  xSparseValues = new TVal[nSparseEntries];

  // Copy the array contennts
  std::copy(src.xRowIndex, src.xRowIndex + nRows + 1, xRowIndex);
  std::copy(src.xColIndex, src.xColIndex + nSparseEntries, xColIndex);
  std::copy(src.xSparseValues, src.xSparseValues + nSparseEntries, xSparseValues);
  
  return *this;
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::MultiplyTransposeByVector(const Vec &b, Vec &result) const
{
  // Make sure the dimensions match
  assert(b.size() == this->nRows);
  assert(result.size() == this->nColumns);

  // Initialize the vector
  result.fill(0.0);

  // Iterate over rows and columns of the matrix
  for(size_t i = 0; i < this->nRows; i++)
    for(size_t j = this->xRowIndex[i]; j < this->xRowIndex[i+1]; j++)
      result[this->xColIndex[j]] += this->xSparseValues[j] * b[i];
}


template<class TVal>
typename ImmutableSparseMatrix<TVal>::Vec 
ImmutableSparseMatrix<TVal>
::MultiplyTransposeByVector(const Vec &b) const
{
  Vec c(this->nColumns, 0);
  MultiplyTransposeByVector(b, c);
  return c;
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::MultiplyByVector(const Vec &b, Vec &result) const
{
  // Make sure the dimensions match
  assert(b.size() == this->nColumns);
  assert(result.size() == this->nColumns);

  // Initialize the vector
  result.fill(0.0);

  // Iterate over rows and columns of the matrix
  for(size_t i = 0; i < this->nRows; i++)
    for(size_t j = this->xRowIndex[i]; j < this->xRowIndex[i+1]; j++)
      result[i] += this->xSparseValues[j] * b[this->xColIndex[j]];
}

template<class TVal>
typename ImmutableSparseMatrix<TVal>::Vec 
ImmutableSparseMatrix<TVal>
::MultiplyByVector(const Vec &b) const
{
  Vec c(this->nRows, 0);
  MultiplyByVector(b, c);
  return c;
}

template<class TVal>
void 
ImmutableSparseMatrix<TVal>
::Multiply(Self &C, const Self &A, const Self &B)
{
  size_t i, j, k, l, q;

  // Of course, check compatibility
  assert(A.nColumns == B.nRows);

  // This is a horrible cheat, but we will create a vnl_sparse_matrix
  // into which we will stick in intermediate products
  vnl_sparse_matrix<TVal> T(A.nRows, B.nColumns);

  for(i = 0; i < A.nRows; i++) 
    for(j = A.xRowIndex[i]; j < A.xRowIndex[i+1]; j++)
      {
      // Here we are looking at the element A(ik). Its product
      // with B(kq) contributes to C(iq). So that's what we do
      k = A.xColIndex[j];

      // Loop over data in B(k*)
      for(l = B.xRowIndex[k]; l < B.xRowIndex[k+1]; l++)
        {
        // Here we found a non-zero element B(kq).
        q = B.xColIndex[l];

        // Add the product to C(iq)
        T(i, q) += A.xSparseValues[j] * B.xSparseValues[l];
        }
      }
 
  // Now, just use the assignment operator to compact the sparse matrix
  C.SetFromVNL(T);
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::PrintSelf(std::ostream &out) const 
{
  size_t i, j;
  out << "ImmutableSparseArray: [ ";
  for(i = 0; i < this->nRows; i++) 
    for(j = this->xRowIndex[i]; j < this->xRowIndex[i+1]; j++)
      out << "(" << i << "," << this->xColIndex[j] << "," << this->xSparseValues[j] << ") ";
  out << "]";
}


template<class TVal>
void
ImmutableSparseMatrix<TVal>
::PrintSelfMathematica(std::ostream &out) const 
{
  size_t i, j;
  out << "SparseArray[{";
  for(i = 0; i < this->nRows; i++) 
    {
    for(j = this->xRowIndex[i]; j < this->xRowIndex[i+1]; j++)
      {
      out << "{" << i+1 << "," << this->xColIndex[j]+1 << "} -> " << this->xSparseValues[j];
      if(j < this->nSparseEntries - 1)
        out << ", ";
      else
        out << "} ";
      }
    out << std::endl;
    }
    
  out << ", " << this->nRows << ", " << this->nColumns << "]; " << std::endl;
}

template<class TVal>
void
ImmutableSparseArray<TVal>
::SetArrays(size_t rows, size_t cols, size_t *xRowIndex, size_t *xColIndex, TVal *data)
{
  Reset();
  this->nRows = rows; 
  this->nColumns = cols;
  this->nSparseEntries = xRowIndex[rows];
  this->xRowIndex = xRowIndex; 
  this->xColIndex = xColIndex;
  this->xSparseValues = data;
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::InitializeATA(Self &ATA, const Self &A)
{
  // Here we are going to cheat and use vnl matrices
  vnl_sparse_matrix<TVal> T(A.nColumns, A.nColumns);
  printf("Mat [%lu, %lu]\n", (unsigned long) A.nColumns, (unsigned long) A.nColumns);

  // Set the values of the sparse matrix. We only set the values in the upper triangle
  for(size_t i = 0; i < A.nRows; i++)
    {
    // Set all the diagonal entries to 1
    T(i, i) = 1;

    // printf("i = %d of %d\n",i,A.nRows);
    for(size_t j1 = A.xRowIndex[i]; j1 < A.xRowIndex[i+1]; j1++)
      {
      for(size_t j2 = j1; j2 < A.xRowIndex[i+1]; j2++)
        {
        // printf("j1 = %d, j2 = %d\n", j1, j2);
        size_t p = A.xColIndex[j1];
        size_t q = A.xColIndex[j2];
        // printf("T(%d,%d) = 1\n", p, q);
        T(p, q) = 1;
        }
      }
    }

  // Create an immutable sparse matrix from this
  ATA.SetFromVNL(T);
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::ComputeATA(Self &ATA, const Self &A)
{
  // Clear the values in the output matrix. 
  ATA.Fill(0);

  // Set the values of the sparse matrix. We only set the values in the upper triangle
  for(size_t i = 0; i < A.nRows; i++)
    {
    // First-level loop
    for(size_t j1 = A.xRowIndex[i]; j1 < A.xRowIndex[i+1]; j1++)
      {
      // The first column index
      size_t p = A.xColIndex[j1];
      size_t j2 = j1;

      // Second-level loop
      for(size_t z = ATA.xRowIndex[p]; z < ATA.xRowIndex[p+1]; z++)
        {
        if(ATA.xColIndex[z] == A.xColIndex[j2])
          {
          ATA.xSparseValues[z] += A.xSparseValues[j1] * A.xSparseValues[j2];
          j2++;
          }
        }
      }
    }
}

template<class TVal>
vnl_matrix<TVal>
ImmutableSparseMatrix<TVal>
::GetDenseMatrix() const
{
  // Create matrix
  vnl_matrix<TVal> A(this->nRows, this->nColumns);

  // Copy values
  for(size_t i = 0; i < this->nRows; i++) 
    for(size_t j = this->xRowIndex[i]; j < this->xRowIndex[i+1]; j++)
      A(i, this->xColIndex[j]) = this->xSparseValues[j];
    
  return A;
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::SetIdentity(size_t n)
{
  // Reset the array
  this->Reset();

  if(n > 0)
    {
    // Set the row and column numbers
    this->nRows = this->nColumns = this->nSparseEntries = n;

    // Set the row index
    this->xRowIndex = new size_t[n+1];
    for(size_t i = 0; i <= n; i++)
      this->xRowIndex[i] = i;

    // Set the column and sparse array indices
    this->xColIndex = new size_t[n];
    this->xSparseValues = new TVal[n];
    for(size_t j = 0; j < n; j++)
      {
      this->xColIndex[j] = j;
      this->xSparseValues[j] = 1;
      }
    }
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::AddScaledMatrix(const Self &B, TVal scale)
{
  // The structure must match
  assert(this->nRows == B.nRows && this->nColumns == B.nColumns);

  // Matrix B may have a subset of entries in this matrix
  for(int i = 0; i < this->nRows; i++)
    {
    // Iterate over the columns of B
    for(size_t j0 = this->xRowIndex[i], j = B.xRowIndex[i]; j < B.xRowIndex[i+1]; j0++)
      {
      if(this->xColIndex[j0] == B.xColIndex[j])
        this->xSparseValues[j0] += B.xSparseValues[j++] * scale;
      }
    }
}

template <class TVal>
void
ImmutableSparseMatrix<TVal>
::AddScaledOuterProduct(const Vec &A, const Vec &B, double scale)
{
  assert(this->nRows == this->nColumns && this->nRows == A.size() && this->nRows == B.size());

  // Iterate over all the entries in the matrix
  for(int i = 0; i < this->nRows; i++)
    {
    for(size_t j = this->xRowIndex[i]; j < this->xRowIndex[i+1]; j++)
      {
      this->xSparseValues[j] += A[i] * B[this->xColIndex[j]] * scale;
      }
    }
}

template<class TVal>
void
ImmutableSparseMatrix<TVal>
::Scale(TVal c)
{
  for(int k = 0; k < this->nSparseEntries; k++)
    this->xSparseValues[k] *= c;
}
