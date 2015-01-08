/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: SimpleWarpImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2009-10-29 11:19:10 $
  Version:   $Revision: 1.34 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __MultiImageAffineMSDMetricFilter_txx
#define __MultiImageAffineMSDMetricFilter_txx
#include "MultiImageAffineMSDMetricFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkContinuousIndex.h"
#include "ImageRegionConstIteratorWithIndexOverride.h"
#include "vnl/vnl_math.h"
#include "FastLinearInterpolator.h"

/**
 * Setup state of filter before multi-threading.
 * InterpolatorType::SetInputImage is not thread-safe and hence
 * has to be setup before ThreadedGenerateData
 */
template <class TInputImage>
void
MultiImageAffineMSDMetricFilter<TInputImage>
::BeforeThreadedGenerateData()
{
  // Initialize the per thread data
  m_ThreadData.resize(this->GetNumberOfThreads(), ThreadData());
}

/**
 * Setup state of filter after multi-threading.
 */
template <class TInputImage>
void
MultiImageAffineMSDMetricFilter<TInputImage>
::AfterThreadedGenerateData()
{
  // Add up all the thread data
  ThreadData summary;
  for(int i = 0; i < m_ThreadData.size(); i++)
    {
    summary.metric += m_ThreadData[i].metric;
    summary.mask += m_ThreadData[i].mask;
    summary.gradient += m_ThreadData[i].gradient;
    summary.grad_mask += m_ThreadData[i].grad_mask;
    }

  // Compute the objective value
  m_MetricValue = summary.metric / summary.mask;

  // Compute the gradient
  vnl_vector<double> grad_metric(summary.gradient.size());
  for(int j = 0; j < summary.gradient.size(); j++)
    {
    grad_metric[j] =
        (-2.0 * summary.gradient[j] - m_MetricValue * summary.grad_mask[j]) / summary.mask;
    }

  // Pack into the output
  m_MetricGradient = TransformType::New();
  unflatten_affine_transform(grad_metric.data_block(), m_MetricGradient.GetPointer());
}

template <class TInputImage>
void
MultiImageAffineMSDMetricFilter<TInputImage>
::GenerateInputRequestedRegion()
{
  // Call the superclass's implementation
  Superclass::GenerateInputRequestedRegion();

  // Set regions to max
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));

  if(moving)
    moving->SetRequestedRegionToLargestPossibleRegion();

  if(fixed)
    fixed->SetRequestedRegionToLargestPossibleRegion();
}

template <class TInputImage>
void
MultiImageAffineMSDMetricFilter<TInputImage>
::EnlargeOutputRequestedRegion(itk::DataObject *data)
{
  Superclass::EnlargeOutputRequestedRegion(data);
  data->SetRequestedRegionToLargestPossibleRegion();
}


template <class TInputImage>
void
MultiImageAffineMSDMetricFilter<TInputImage>
::AllocateOutputs()
{
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  this->GraftOutput(fixed);
}


template <class TInputImage>
void
MultiImageAffineMSDMetricFilter<TInputImage>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  itk::ThreadIdType threadId )
{
  // Get the pointers to the input and output images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));

  // Get the pointer to the start of the fixed data
  const InputComponentType *fix_buffer = fixed->GetBufferPointer();

  // Get the number of components
  int kFixed = fixed->GetNumberOfComponentsPerPixel();
  int kMoving = moving->GetNumberOfComponentsPerPixel();

  // Create an interpolator for the moving image
  typedef FastLinearInterpolator<InputComponentType, ImageDimension> FastInterpolator;
  FastInterpolator flint(moving);

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef ImageRegionConstIteratorWithIndexOverride<InputImageType> FixedIter;

  // Location of the lookup
  vnl_vector_fixed<float, ImageDimension> cix;

  // Pointer to store interpolated moving data
  vnl_vector<typename InputImageType::InternalPixelType> interp_mov(kFixed);

  // Pointer to store the gradient of the moving images
  vnl_vector<typename InputImageType::InternalPixelType> interp_mov_grad(kFixed * ImageDimension);

  // The thread data to accumulate
  ThreadData &td = m_ThreadData[threadId];

  // Affine transform matrix and vector
  vnl_matrix_fixed<double, ImageDimension, ImageDimension> M =
      m_Transform->GetMatrix().GetVnlMatrix();
  vnl_vector_fixed<double, ImageDimension> off =
      m_Transform->GetOffset().GetVnlVector();

  // Gradient accumulator
  vnl_vector_fixed<double, ImageDimension> grad, gradM;

  // Iterate over the fixed space region
  for(FixedIter it(fixed, outputRegionForThread); !it.IsAtEnd(); ++it)
    {
    // Get the index at the current location
    const IndexType &idx = it.GetIndex();

    // Get the pointer to the fixed pixel
    // TODO: WHY IS THIS RETURNING NONSENSE?
    const InputComponentType *fix_ptr =
        fix_buffer + (it.GetPosition() - fix_buffer) * kFixed;

    // Map to a position at which to interpolate
    // TODO: all this can be done more efficiently!
    for(int i = 0; i < ImageDimension; i++)
      {
      cix[i] = off[i];
      for(int j = 0; j < ImageDimension; j++)
        cix[i] += M(i,j) * idx[j];
      }

    // Do we need the gradient?
    if(m_ComputeGradient)
      {
      // Interpolate moving image with gradient
      typename FastInterpolator::InOut status =
          flint.InterpolateWithGradient(cix.data_block(),
                                        interp_mov.data_block(),
                                        interp_mov_grad.data_block());

      // Stop if the sample is outside
      if(status == FastInterpolator::OUTSIDE)
        continue;

      // Initialize the gradient to zeros
      grad.fill(0.0);

      // Iterate over the components
      const InputComponentType *mov_ptr = interp_mov.data_block(), *mov_ptr_end = mov_ptr + kFixed;
      const InputComponentType *mov_grad_ptr = interp_mov_grad.data_block();
      float *wgt_ptr = m_Weights.data_block();
      double w_sq_diff = 0.0;

      // Compute the gradient of the term contribution for this voxel
      for( ;mov_ptr < mov_ptr_end; ++mov_ptr, ++fix_ptr, ++wgt_ptr)
        {
        // Intensity difference for k-th component
        double del = (*fix_ptr) - *(mov_ptr);

        // Weighted intensity difference for k-th component
        double delw = (*wgt_ptr) * del;

        // Accumulate the weighted sum of squared differences
        w_sq_diff += delw * del;

        // Accumulate the weighted gradient term
        for(int i = 0; i < ImageDimension; i++)
          grad[i] += delw * *(mov_grad_ptr++);
        }

      // Accumulators for the gradients
      double *out_grad = td.gradient.data_block();
      double *out_grad_mask = td.grad_mask.data_block();

      // For border regions, we need to explicitly deal with the mask
      if(status == FastInterpolator::BORDER)
        {
        // Border - compute the mask and its gradient
        double mask = flint.GetMaskAndGradient(gradM.data_block());

        // Compute the mask and metric gradient contributions
        for(int i = 0; i < ImageDimension; i++)
          {
          double v = grad[i] * mask - 0.5 * gradM[i] * w_sq_diff;
          *(out_grad++) += v;
          *(out_grad_mask++) += gradM[i];
          for(int j = 0; j < ImageDimension; j++)
            {
            *(out_grad++) += v * idx[j];
            *(out_grad_mask++) += gradM[i] * idx[j];
            }
          }

        td.metric += w_sq_diff * mask;
        td.mask += mask;
        }
      else
        {
        // No border - means no dealing with the mask!
        for(int i = 0; i < ImageDimension; i++)
          {
          *(out_grad++) += grad[i];
          for(int j = 0; j < ImageDimension; j++)
            {
            *(out_grad++) += grad[i] * idx[j];
            }
          }

        td.metric += w_sq_diff;
        td.mask += 1.0;
        }
      }

    // No gradient requested
    else
      {
      // Interpolate moving image with gradient
      typename FastInterpolator::InOut status =
          flint.Interpolate(cix.data_block(), interp_mov.data_block());

      // Stop if the sample is outside
      if(status == FastInterpolator::OUTSIDE)
        continue;

      // Iterate over the components
      const InputComponentType *mov_ptr = interp_mov.data_block(), *mov_ptr_end = mov_ptr + kFixed;
      float *wgt_ptr = m_Weights.data_block();
      double w_sq_diff = 0.0;

      // Compute the gradient of the term contribution for this voxel
      for( ;mov_ptr < mov_ptr_end; ++mov_ptr, ++fix_ptr, ++wgt_ptr)
        {
        // Intensity difference for k-th component
        double del = (*fix_ptr) - *(mov_ptr);

        // Weighted intensity difference for k-th component
        double delw = (*wgt_ptr) * del;

        // Accumulate the weighted sum of squared differences
        w_sq_diff += delw * del;
        }

      // For border regions, we need to explicitly deal with the mask
      if(status == FastInterpolator::BORDER)
        {
        // Border - compute the mask and its gradient
        double mask = flint.GetMaskAndGradient(gradM.data_block());
        td.metric += w_sq_diff * mask;
        td.mask += mask;
        }
      else
        {
        td.metric += w_sq_diff;
        td.mask += 1.0;
        }
      }
    }
}


#endif
