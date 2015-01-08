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
#ifndef __MultiImageOpticalFlowImageFilter_txx
#define __MultiImageOpticalFlowImageFilter_txx
#include "MultiImageOpticalFlowImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkContinuousIndex.h"
#include "vnl/vnl_math.h"
#include "FastLinearInterpolator.h"
#include "ImageRegionConstIteratorWithIndexOverride.h"

/**
 * Default constructor.
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::MultiImageOpticalFlowImageFilter()
{
  // Setup default values
  // m_DeformationScaling = 1.0;
}

/**
 * Standard PrintSelf method.
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

/**
 * Setup state of filter before multi-threading.
 * InterpolatorType::SetInputImage is not thread-safe and hence
 * has to be setup before ThreadedGenerateData
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::BeforeThreadedGenerateData()
{
  // Create the prototype results vector
  m_MetricPerThread.resize(this->GetNumberOfThreads(), 0.0);
}

/**
 * Setup state of filter after multi-threading.
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::AfterThreadedGenerateData()
{
  m_MetricValue = 0.0;
  for(int i = 0; i < m_MetricPerThread.size(); i++)
    m_MetricValue += m_MetricPerThread[i];
}


/**
 * Compute the output for the region specified by outputRegionForThread.
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  itk::ThreadIdType threadId )
{
  // Get the pointers to the input and output images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));
  DeformationFieldType *phi = dynamic_cast<DeformationFieldType *>(this->ProcessObject::GetInput("phi"));
  OutputImageType *out = this->GetOutput();

  // Get the number of components
  int kFixed = fixed->GetNumberOfComponentsPerPixel();
  int kMoving = moving->GetNumberOfComponentsPerPixel();

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef ImageRegionConstIteratorWithIndexOverride<OutputImageType> OutputIter;

  // Location of the lookup
  vnl_vector_fixed<float, ImageDimension> cix;

  // Pointer to the fixed image data
  const InputComponentType *bFix = fixed->GetBufferPointer();
  const InputComponentType *bMov = moving->GetBufferPointer();
  const DeformationVectorType *bPhi = phi->GetBufferPointer();
  OutputPixelType *bOut = out->GetBufferPointer();

  // Pointer to store interpolated moving data
  vnl_vector<InputComponentType> interp_mov(kFixed);

  // Pointer to store the gradient of the moving images
  vnl_vector<InputComponentType> interp_mov_grad(kFixed * ImageDimension);

  // Our component of the metric computation
  double &metric = m_MetricPerThread[threadId];

  // Get the stride for interpolation (how many moving pixels to skip)
  int stride = 1 + ImageDimension;

  // Create an interpolator for the moving image
  typedef FastLinearInterpolator<InputComponentType, ImageDimension> FastInterpolator;
  FastInterpolator flint(moving);

  // Iterate over the fixed space region
  for(OutputIter it(out, outputRegionForThread); !it.IsAtEnd(); ++it)
    {
    // Get the index at the current location
    const IndexType &idx = it.GetIndex();

    // Get the output pointer at this location
    OutputPixelType *ptrOut = const_cast<OutputPixelType *>(it.GetPosition());

    // Get the offset into the fixed pointer
    long offset = ptrOut - bOut;

    // Map to a position at which to interpolate
    const DeformationVectorType &def = bPhi[offset];
    for(int i = 0; i < ImageDimension; i++)
      cix[i] = idx[i] + def[i];

    // Perform the interpolation, put the results into interp_mov
    flint.InterpolateWithGradient(cix.data_block(), stride,
                                  interp_mov.data_block(),
                                  interp_mov_grad.data_block());

    // Compute the optical flow gradient field
    OutputPixelType &vOut = *ptrOut;
    for(int i = 0; i < ImageDimension; i++)
      vOut[i] = 0;

    // Iterate over the components
    const InputComponentType *mov_ptr = interp_mov.data_block();
    const InputComponentType *mov_ptr_end = mov_ptr + kFixed;
    const InputComponentType *mov_grad_ptr = interp_mov_grad.data_block();
    const InputComponentType *fix_ptr = bFix + offset * kFixed;
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
        vOut[i] += delw * *(mov_grad_ptr++);
      }

    metric += w_sq_diff;
    }
}


template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::GenerateInputRequestedRegion()
{
  // call the superclass's implementation
  Superclass::GenerateInputRequestedRegion();

  // Different behavior for fixed and moving images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));
  DeformationFieldType *phi = dynamic_cast<DeformationFieldType *>(this->ProcessObject::GetInput("phi"));

  if(moving)
    moving->SetRequestedRegionToLargestPossibleRegion();

  if(fixed)
    {
    fixed->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
    if(!fixed->VerifyRequestedRegion())
      fixed->SetRequestedRegionToLargestPossibleRegion();
    }

  if(phi)
    {
    phi->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
    if(!phi->VerifyRequestedRegion())
      phi->SetRequestedRegionToLargestPossibleRegion();
    }
}


template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::GenerateOutputInformation()
{
  // call the superclass's implementation of this method
  Superclass::GenerateOutputInformation();

  OutputImageType *outputPtr = this->GetOutput();
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  outputPtr->SetSpacing( fixed->GetSpacing() );
  outputPtr->SetOrigin( fixed->GetOrigin() );
  outputPtr->SetDirection( fixed->GetDirection() );
  outputPtr->SetLargestPossibleRegion( fixed->GetLargestPossibleRegion() );
}





#endif
