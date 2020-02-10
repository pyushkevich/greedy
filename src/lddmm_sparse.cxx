/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as publishGed by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#include "lddmm_data.h"

#ifdef _LDDMM_SPARSE_SOLVERS_

// Use VNL sparse matrix as a temporary data structure
#include <vnl/vnl_sparse_matrix.h>
#include <SparseMatrix.h>
#include <SparseSolver.h>

/**
 * A specialized implementation for double images
 */
template <class TFloat, uint VDim>
class PoissonPDEZeroBoundary
{
public:
  typedef LDDMMData<TFloat, VDim> Data;
  typedef typename Data::ImageBaseType ImageBaseType;
  typedef typename Data::ImageType ImageType;
  typedef ImmutableSparseMatrix<double> SMat;

  typedef itk::Image<long, VDim> IndexImage;
  typedef itk::ImageRegionIteratorWithIndex<IndexImage> IndexIterator;

  PoissonPDEZeroBoundary(ImageBaseType *ref, ImageType *mask = NULL)
    {
    // Number of pixels in the image (dimensions of the matrix)
    unsigned long np = ref->GetBufferedRegion().GetNumberOfPixels();

    // Allocate the mask index image
    m_MaskIndex = IndexImage::New();
    m_MaskIndex->SetRegions(ref->GetBufferedRegion());
    m_MaskIndex->CopyInformation(ref);
    m_MaskIndex->Allocate();
    m_MaskIndex->FillBuffer(-1l);


    // If a mask is supplied, figure out the number of variables and map pixels
    // to variables in the Laplacian problem
    unsigned long nv = 0;
    if(mask)
      {
      typename Data::ImageIterator it_mask(mask, mask->GetBufferedRegion());
      IndexIterator it_mindex(m_MaskIndex, mask->GetBufferedRegion());
      for(; !it_mask.IsAtEnd(); ++it_mask, ++it_mindex)
        {
        if(it_mask.Get() >= 1.0)
          it_mindex.Set(nv++);
        }
      }
    else
      {
      IndexIterator it_mindex(m_MaskIndex, m_MaskIndex->GetBufferedRegion());
      for(; !it_mindex.IsAtEnd(); ++it_mindex)
        it_mindex.Set(nv++);
      }

    // Create a sparse matrix representing the image Laplacian
    vnl_sparse_matrix<double> S(nv, nv);

    // Get the image offsets and size
    const typename ImageBaseType::OffsetValueType *offsets = ref->GetOffsetTable();
    typename ImageBaseType::SpacingType spacing = ref->GetSpacing();
    typename ImageBaseType::IndexType i_start = ref->GetBufferedRegion().GetIndex();
    typename ImageBaseType::IndexType i_end = i_start + ref->GetBufferedRegion().GetSize();

    // Weights of the neighbor voxels and center voxel
    vnl_vector_fixed<double, VDim> w_nbr;
    double w_ctr = 0.0;
    for(unsigned int d = 0; d < VDim; d++)
      {
      double s = ref->GetSpacing()[d];
      w_nbr[d] = 1.0 / (s * s);
      w_ctr -= 2.0 / (s * s); 
      }

    // Visit each pixel in order
    long *mindex_ptr = m_MaskIndex->GetBufferPointer();
    IndexIterator it(m_MaskIndex, m_MaskIndex->GetBufferedRegion());
    for(unsigned int i=0; !it.IsAtEnd(); ++it, ++i)
      {
      // We must be inside the mask to have a row in the matrix
      long j_center = mindex_ptr[i];
      if(j_center >= 0)
        {
        // Add the central value
        S.put(j_center, j_center, w_ctr);

        // Add the neighbor values
        for(unsigned int d = 0; d < VDim; d++)
          {
          unsigned int k = it.GetIndex()[d];
          if(k > i_start[d])
            {
            long j_nbr = mindex_ptr[i-offsets[d]];
            if(j_nbr >= 0)
              S.put(j_center, j_nbr, w_nbr[d]);
            }
          if(k < i_end[d]-1)
            {
            long j_nbr = mindex_ptr[i+offsets[d]];
            if(j_nbr >= 0)
              S.put(j_center, j_nbr, w_nbr[d]);
            }
          }
        }
      }
     
    // Create a sparse matrix
    m_LaplacianMatrix.SetFromVNL(S); 

    // Create a sparse solver (not symmetric)
    m_Solver = SparseSolver::MakeSolver(false);
    m_Solver->SymbolicFactorization(m_LaplacianMatrix);
    m_Solver->NumericFactorization(m_LaplacianMatrix.GetSparseData());

    // Allocate the work vectors
    m_WorkU.set_size(nv);
    m_WorkV.set_size(nv);
    }

  ~PoissonPDEZeroBoundary()
    {
    delete m_Solver;
    }

  // Use mask to load data into a nv vector
  void PutImageIntoVector(ImageType *img, SMat::Vec &vec)
    {
    TFloat *pix = img->GetBufferPointer();
    long *mindex_ptr = m_MaskIndex->GetBufferPointer();
    for(unsigned long i = 0; i < img->GetBufferedRegion().GetNumberOfPixels(); i++)
      if(mindex_ptr[i] >= 0)
        vec[mindex_ptr[i]] = pix[i];
    }

  // Load from the data from an nv vector
  void GetImageFromVector(SMat::Vec &vec, ImageType *img)
    {
    TFloat *pix = img->GetBufferPointer();
    long *mindex_ptr = m_MaskIndex->GetBufferPointer();
    for(unsigned long i = 0; i < img->GetBufferedRegion().GetNumberOfPixels(); i++)
      if(mindex_ptr[i] >= 0)
        pix[i] = vec[mindex_ptr[i]];
    }

  void Solve(ImageType *rhs, ImageType *soln)
    {
    PutImageIntoVector(rhs, m_WorkU);
    m_Solver->Solve(m_WorkU.data_block(), m_WorkV.data_block());
    soln->FillBuffer(0.0);
    GetImageFromVector(m_WorkV, soln);
    }

  void ComputeLaplacian(ImageType *u, ImageType *res)
    {
    PutImageIntoVector(u, m_WorkU);
    m_LaplacianMatrix.MultiplyByVector(m_WorkU, m_WorkV);
    res->FillBuffer(0.0);
    GetImageFromVector(m_WorkV, res);
    }

private:
  SparseSolver *m_Solver;
  SMat m_LaplacianMatrix;
  typename IndexImage::Pointer m_MaskIndex;
  SMat::Vec m_WorkU, m_WorkV;
};

#else

/*
 * Default implementation of the PDE solver is a dummy that does not work
 */
template <class TFloat, uint VDim>
class PoissonPDEZeroBoundary
{
public:
  typedef LDDMMData<TFloat, VDim> Data;
  typedef typename Data::ImageBaseType ImageBaseType;
  typedef typename Data::ImageType ImageType;

  PoissonPDEZeroBoundary(ImageBaseType *ref, ImageType *mask = NULL)
    {
    std::cerr << "PDE solver not available" << std::endl;
    throw std::exception();
    }

  void Solve(ImageType *rhs, ImageType *soln) {}
  void ComputeLaplacian(ImageType *u, ImageType *res) {}
};


#endif 


/*
 * Initialize the sparse solver solving Poisson equation involving
 * the provided image, with zero Dirichlet boundary conditions
 *
 * This is meant to be called only once, then iteratively call solve
 *
 * It's the caller's job to delete the void *
 */
template <class TFloat, uint VDim>
void *
LDDMMData<TFloat, VDim>
::poisson_pde_zero_boundary_initialize(ImageBaseType *ref, ImageType *mask)
{
  typedef PoissonPDEZeroBoundary<TFloat, VDim> PDEType;
  PDEType *pde = new PDEType(ref, mask);
  return pde;
}

template <typename TFloat, uint VDim>
void
LDDMMData<TFloat, VDim>
::poisson_pde_zero_boundary_solve(void *solver_data, ImageType *rhs, ImageType *soln)
{
  typedef PoissonPDEZeroBoundary<TFloat, VDim> PDEType;
  PDEType *pde = static_cast<PDEType *>(solver_data);
  pde->Solve(rhs, soln);
}

template <typename TFloat, uint VDim>
void
LDDMMData<TFloat, VDim>
::poisson_pde_zero_boundary_laplacian(void *solver_data, ImageType *u, ImageType *result)
{
  typedef PoissonPDEZeroBoundary<TFloat, VDim> PDEType;
  PDEType *pde = static_cast<PDEType *>(solver_data);
  pde->ComputeLaplacian(u, result);
}

template class LDDMMData<float, 2>;
template class LDDMMData<float, 3>;
template class LDDMMData<float, 4>;

template class LDDMMData<double, 2>;
template class LDDMMData<double, 3>;
template class LDDMMData<double, 4>;

