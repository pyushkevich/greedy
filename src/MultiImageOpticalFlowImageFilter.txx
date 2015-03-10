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



template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilterBase<TInputImage,TOutputImage,TDeformationField>
::GenerateInputRequestedRegion()
{
  // call the superclass's implementation
  Superclass::GenerateInputRequestedRegion();

  // Different behavior for fixed and moving images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("moving"));
  DeformationFieldType *phi = dynamic_cast<DeformationFieldType *>(this->itk::ProcessObject::GetInput("phi"));

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
MultiImageOpticalFlowImageFilterBase<TInputImage,TOutputImage,TDeformationField>
::GenerateOutputInformation()
{
  // call the superclass's implementation of this method
  Superclass::GenerateOutputInformation();

  OutputImageType *outputPtr = this->GetOutput();
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("Primary"));
  outputPtr->SetSpacing( fixed->GetSpacing() );
  outputPtr->SetOrigin( fixed->GetOrigin() );
  outputPtr->SetDirection( fixed->GetDirection() );
  outputPtr->SetLargestPossibleRegion( fixed->GetLargestPossibleRegion() );
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
  const InputImageType *fixed = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("Primary"));

  // Create the prototype results vector
  m_MetricStatsPerThread.clear();
  for(int i = 0; i < this->GetNumberOfThreads(); i++)
    {
    MetricStats ms;
    ms.metric_values.set_size(fixed->GetNumberOfComponentsPerPixel());
    ms.metric_values.fill(0.0);
    ms.num_voxels = 0;
    m_MetricStatsPerThread.push_back(ms);
    }

}

/**
 * Setup state of filter after multi-threading.
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::AfterThreadedGenerateData()
{
  const InputImageType *fixed = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("Primary"));

  // Allocate the final result vector
  m_MetricStats.metric_values.set_size(fixed->GetNumberOfComponentsPerPixel());
  m_MetricStats.metric_values.fill(0.0);
  m_MetricStats.num_voxels = 0.0;

  for(int i = 0; i < m_MetricStatsPerThread.size(); i++)
    {
    m_MetricStats.metric_values += m_MetricStatsPerThread[i].metric_values;
    m_MetricStats.num_voxels += m_MetricStatsPerThread[i].num_voxels;
    }

  // Compute overall metric value
  m_MetricValue = 0.0;
  for(int j = 0; j < fixed->GetNumberOfComponentsPerPixel(); j++)
    m_MetricValue += m_MetricStats.metric_values[j];

  // Metric value normalized by the number of voxels - to make comparisons simpler
  m_MetricValue /= m_MetricStats.num_voxels;
}

template <class TInputImage, class TOutputImage, class TDeformationField>
vnl_vector<double>
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::GetAllMetricValues() const
{
  vnl_vector<double> result;
  result = m_MetricStats.metric_values / m_MetricStats.num_voxels;
  return result;
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
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("moving"));
  DeformationFieldType *phi = dynamic_cast<DeformationFieldType *>(this->itk::ProcessObject::GetInput("phi"));
  OutputImageType *out = this->GetOutput();

  // Get the number of components
  int kFixed = fixed->GetNumberOfComponentsPerPixel();
  int kMoving = moving->GetNumberOfComponentsPerPixel();

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef itk::ImageRegionIteratorWithIndex<OutputImageType> OutputIterBase;
  typedef IteratorExtender<OutputIterBase> OutputIter;

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
  MetricStats &metric_stats = m_MetricStatsPerThread[threadId];
  double *metric_accum_ptr = metric_stats.metric_values.data_block();

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
    typename FastInterpolator::InOut status =
        flint.InterpolateWithGradient(cix.data_block(),
                                      interp_mov.data_block(),
                                      interp_mov_grad.data_block());

    // Handle outside values
    if(status == FastInterpolator::OUTSIDE)
      {
      interp_mov.fill(0.0);
      interp_mov_grad.fill(0.0);
      }

    // Compute the optical flow gradient field
    OutputPixelType &vOut = *ptrOut;
    for(int i = 0; i < ImageDimension; i++)
      vOut[i] = 0;

    // Iterate over the components
    const InputComponentType *mov_ptr = interp_mov.data_block();
    const InputComponentType *mov_ptr_end = mov_ptr + kFixed;
    const InputComponentType *mov_grad_ptr = interp_mov_grad.data_block();
    const InputComponentType *fix_ptr = bFix + offset * kFixed;
    float *wgt_ptr = this->m_Weights.data_block();
    double *met_ptr = metric_accum_ptr;

    // Compute the gradient of the term contribution for this voxel
    for( ;mov_ptr < mov_ptr_end; ++mov_ptr, ++fix_ptr, ++wgt_ptr, ++met_ptr)
      {
      // Intensity difference for k-th component
      double del = (*fix_ptr) - *(mov_ptr);

      // Weighted intensity difference for k-th component
      double delw = (*wgt_ptr) * del;

      // Accumulate the weighted sum of squared differences
      *met_ptr += delw * del;

      // Accumulate the weighted gradient term
      for(int i = 0; i < ImageDimension; i++)
        vOut[i] += delw * *(mov_grad_ptr++);
      }

    metric_stats.num_voxels++;
    }
}


/**
 * Generate output information, which will be different from the default
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageNCCPrecomputeFilter<TInputImage,TOutputImage,TDeformationField>
::GenerateOutputInformation()
{
  this->GetOutput()->CopyInformation(this->GetInput());

  int ncomp = (m_ComputeGradient)
              ? 1 + this->GetInput()->GetNumberOfComponentsPerPixel() * (5 + ImageDimension * 3)
              : 1 + this->GetInput()->GetNumberOfComponentsPerPixel() * 5;

  this->GetOutput()->SetNumberOfComponentsPerPixel(ncomp);
}


/**
 * Compute the output for the region specified by outputRegionForThread.
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
void
MultiImageNCCPrecomputeFilter<TInputImage,TOutputImage,TDeformationField>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  itk::ThreadIdType threadId )
{
  // Get the pointers to the input and output images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("moving"));
  DeformationFieldType *phi = dynamic_cast<DeformationFieldType *>(this->itk::ProcessObject::GetInput("phi"));
  OutputImageType *out = this->GetOutput();

  // Get the number of components
  int kFixed = fixed->GetNumberOfComponentsPerPixel();
  int kOutput = out->GetNumberOfComponentsPerPixel();

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef itk::ImageLinearConstIteratorWithIndex<OutputImageType> OutputIterBase;
  typedef IteratorExtender<OutputIterBase> OutputIter;
  // typedef ImageRegionConstIteratorWithIndexOverride<OutputImageType> OutputIter;

  // Location of the lookup
  vnl_vector_fixed<float, ImageDimension> cix;

  // Pointer to the fixed image data
  const InputComponentType *bFix = fixed->GetBufferPointer();
  const DeformationVectorType *bPhi = phi->GetBufferPointer();
  OutputComponentType *bOut = out->GetBufferPointer();

  // Pointer to store interpolated moving data
  vnl_vector<InputComponentType> interp_mov(kFixed);

  // Pointer to store the gradient of the moving images
  vnl_vector<InputComponentType> interp_mov_grad(kFixed * ImageDimension);

  // Create an interpolator for the moving image
  typedef FastLinearInterpolator<InputComponentType, ImageDimension> FastInterpolator;
  FastInterpolator flint(moving);

  // Iterate over the fixed space region
  for(OutputIter it(out, outputRegionForThread); !it.IsAtEnd(); it.NextLine())
    {
    // Process the whole line using pointer arithmetic. We have to deal with messy behavior
    // of iterators on vector images. Trying to avoid using accessors and Set/Get
    long offset_in_pixels = it.GetPosition() - bOut;
    OutputComponentType *ptrOut = bOut + offset_in_pixels * kOutput;
    OutputComponentType *ptrEnd = ptrOut + outputRegionForThread.GetSize(0) * kOutput;

    // Get the beginning of the same line in the deformation image
    const DeformationVectorType *def_ptr = bPhi + offset_in_pixels;

    // Get the beginning of the same line in the fixed image
    const InputComponentType *fix_ptr = bFix + offset_in_pixels * kFixed;

    // Get the index at the current location
    IndexType idx = it.GetIndex();

    // Loop over the line
    for(; ptrOut < ptrEnd; idx[0]++, def_ptr++)
      {
      // Map to a position at which to interpolate
      for(int i = 0; i < ImageDimension; i++)
        cix[i] = idx[i] + (*def_ptr)[i];

      // Perform the interpolation, put the results into interp_mov
      typename FastInterpolator::InOut status =
          flint.InterpolateWithGradient(cix.data_block(),
                                        interp_mov.data_block(),
                                        interp_mov_grad.data_block());

      double x = cix[0], y = cix[1], z = cix[2];

      // Handle outside values
      if(status == FastInterpolator::OUTSIDE)
        {
        interp_mov.fill(0.0);
        interp_mov_grad.fill(0.0);
        }

      // Fake a function
      /*
      double a = 0.01, b = 0.005, c = 0.008, d = 0.004;
      interp_mov[0] = sin(a * x * y + b * z) + cos(c * x + d * y * z);
      interp_mov_grad[0] = cos(a * x * y + b * z) * a * y - sin(c * x + d * y * z) * c;
      interp_mov_grad[1] = cos(a * x * y + b * z) * a * x - sin(c * x + d * y * z) * d * z;
      interp_mov_grad[2] = cos(a * x * y + b * z) * b     - sin(c * x + d * y * z) * d * y;
      */

      // Fixed image pointer
      const InputComponentType *mov_ptr = interp_mov.data_block();
      const InputComponentType *mov_ptr_end = mov_ptr + kFixed;
      const InputComponentType *mov_grad_ptr = interp_mov_grad.data_block();

      // Fill out the output vector
      // i, j, i^2, i*j, j^2, gradJ[0], i*gradJ[0], j*gradJ[0], ...

      // Write out 1!
      *ptrOut++ = 1.0;

      for( ;mov_ptr < mov_ptr_end; ++mov_ptr, ++fix_ptr)
        {
        InputComponentType x_fix = *fix_ptr, x_mov = *mov_ptr;
        *ptrOut++ = x_fix;
        *ptrOut++ = x_mov;
        *ptrOut++ = x_fix * x_fix;
        *ptrOut++ = x_mov * x_mov;
        *ptrOut++ = x_fix * x_mov;

        if(m_ComputeGradient)
          {
          for(int i = 0; i < ImageDimension; i++, mov_grad_ptr++)
            {
            InputComponentType x_grad_mov_i = *mov_grad_ptr;
            *ptrOut++ = x_grad_mov_i;
            *ptrOut++ = x_fix * x_grad_mov_i;
            *ptrOut++ = x_mov * x_grad_mov_i;
            }
          }
        }
      }
    }
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::MultiImageNCCPostcomputeFilter()
{
  // Set the number of outputs
  this->SetNumberOfRequiredOutputs(2);
  this->SetNthOutput(0, this->MakeOutput(0));
  this->SetNthOutput(1, this->MakeOutput(1));

  // We are not computing the gradient by default
  m_ComputeGradient = false;
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
typename itk::DataObject::Pointer
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::MakeOutput(DataObjectPointerArraySizeType idx)
{
  if(idx == 0)
    return (MetricImageType::New()).GetPointer();
  else if(idx == 1)
    return (GradientImageType::New()).GetPointer();
  else
    return NULL;
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
typename MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>::MetricImageType *
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::GetMetricOutput()
{
  return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput(0));
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
typename MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>::GradientImageType *
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::GetGradientOutput()
{
  return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput(1));
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
void
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::BeforeThreadedGenerateData()
{
  // Create the prototype results vector
  m_MetricPerThread.resize(this->GetNumberOfThreads(), 0.0);
}

/**
 * Setup state of filter after multi-threading.
 */
template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
void
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::AfterThreadedGenerateData()
{
  m_MetricValue = 0.0;
  for(int i = 0; i < m_MetricPerThread.size(); i++)
    m_MetricValue += m_MetricPerThread[i];
}

// Compute sigma_I, sigma_J, sigma_IJ
/*
 * COMPUTATION WITH EPSILON IN DENOM
 *
InputComponentType x_fix = *ptr++;
InputComponentType x_mov = *ptr++;
InputComponentType x_fix_sq = *ptr++;
InputComponentType x_mov_sq = *ptr++;
InputComponentType x_fix_mov = *ptr++;

InputComponentType x_fix_over_n = x_fix * one_over_n;
InputComponentType x_mov_over_n = x_mov * one_over_n;

InputComponentType var_fix = x_fix_sq - x_fix * x_fix_over_n;
InputComponentType var_mov = x_mov_sq - x_mov * x_mov_over_n;
InputComponentType cov_fix_mov = x_fix_mov - x_fix * x_mov_over_n;

InputComponentType one_over_denom = 1.0 / (var_fix * var_mov + eps);
InputComponentType cov_fix_mov_over_denom = cov_fix_mov * one_over_denom;
InputComponentType ncc_fix_mov = cov_fix_mov * cov_fix_mov_over_denom;

for(int i = 0; i < ImageDimension; i++)
  {
  InputComponentType x_grad_mov_i = *ptr++;
  InputComponentType x_fix_grad_mov_i = *ptr++;
  InputComponentType x_mov_grad_mov_i = *ptr++;

  // Derivative of cov_fix_mov
  InputComponentType grad_cov_fix_mov_i = x_fix_grad_mov_i - x_fix_over_n * x_grad_mov_i;

  // One half derivative of var_mov
  InputComponentType half_grad_var_mov_i = x_mov_grad_mov_i - x_mov_over_n * x_grad_mov_i;

  InputComponentType grad_ncc_fix_mov_i =
      2 * cov_fix_mov_over_denom * (grad_cov_fix_mov_i - var_fix * half_grad_var_mov_i * cov_fix_mov_over_denom);

  (*ptr_gradient)[i] += m_Weights[i_wgt] * grad_ncc_fix_mov_i;
  // (*ptr_gradient)[i] = grad_ncc_fix_mov_i;


  // (*ptr_gradient)[i] = x_grad_mov_i; // grad_cov_fix_mov_i;
  }

// *ptr_metric = ncc_fix_mov;
*ptr_metric += m_Weights[i_wgt] * ncc_fix_mov;
// *ptr_metric = x_mov; // cov_fix_mov;

++i_wgt;
*/


/*
 * ADD epsilon to numerator and denominator
 *

InputComponentType x_fix = *ptr++;
InputComponentType x_mov = *ptr++;
InputComponentType x_fix_sq = *ptr++;
InputComponentType x_mov_sq = *ptr++;
InputComponentType x_fix_mov = *ptr++;

InputComponentType x_fix_over_n = x_fix * one_over_n;
InputComponentType x_mov_over_n = x_mov * one_over_n;

// Epsilon is used to stabilize numerical computation
double eps = 1.0e-4;

InputComponentType var_fix = x_fix_sq - x_fix * x_fix_over_n + eps;
InputComponentType var_mov = x_mov_sq - x_mov * x_mov_over_n + eps;

InputComponentType cov_fix_mov = x_fix_mov - x_fix * x_mov_over_n + eps;

InputComponentType one_over_denom = 1.0 / (var_fix * var_mov);
InputComponentType cov_fix_mov_over_denom = cov_fix_mov * one_over_denom;
InputComponentType ncc_fix_mov = cov_fix_mov * cov_fix_mov_over_denom;

float w = m_Weights[i_wgt];
if(cov_fix_mov < 0)
  w = -w;

for(int i = 0; i < ImageDimension; i++)
  {
  InputComponentType x_grad_mov_i = *ptr++;
  InputComponentType x_fix_grad_mov_i = *ptr++;
  InputComponentType x_mov_grad_mov_i = *ptr++;

  // Derivative of cov_fix_mov
  InputComponentType grad_cov_fix_mov_i = x_fix_grad_mov_i - x_fix_over_n * x_grad_mov_i;

  // One half derivative of var_mov
  InputComponentType half_grad_var_mov_i = x_mov_grad_mov_i - x_mov_over_n * x_grad_mov_i;

  InputComponentType grad_ncc_fix_mov_i =
      2 * cov_fix_mov_over_denom * (grad_cov_fix_mov_i - var_fix * half_grad_var_mov_i * cov_fix_mov_over_denom);

  (*ptr_gradient)[i] += w * grad_ncc_fix_mov_i;
  // (*ptr_gradient)[i] = grad_ncc_fix_mov_i;


  // (*ptr_gradient)[i] = x_grad_mov_i; // grad_cov_fix_mov_i;
  }

// *ptr_metric = ncc_fix_mov;

*ptr_metric += w * ncc_fix_mov;
// *ptr_metric = x_mov; // cov_fix_mov;
*/


template <class TPixel, class TWeight, class TMetric, class TGradient>
TPixel *
MultiImageNNCPostComputeFunction(
    TPixel *ptr, TPixel *ptr_end, TWeight *weights, TMetric *ptr_metric, TGradient *ptr_gradient, int ImageDimension)
{
  // Get the size of the mean filter kernel
  TPixel n = *ptr++, one_over_n = 1.0 / n;

  // Loop over components
  int i_wgt = 0;
  const TPixel eps = 1e-8;

  // Initialize metric to zero
  *ptr_metric = 0;

  for(; ptr < ptr_end; ++i_wgt)
    {
    TPixel x_fix = *ptr++;
    TPixel x_mov = *ptr++;
    TPixel x_fix_sq = *ptr++;
    TPixel x_mov_sq = *ptr++;
    TPixel x_fix_mov = *ptr++;

    TPixel x_fix_over_n = x_fix * one_over_n;
    TPixel x_mov_over_n = x_mov * one_over_n;

    TPixel var_fix = x_fix_sq - x_fix * x_fix_over_n;
    TPixel var_mov = x_mov_sq - x_mov * x_mov_over_n;

    if(var_fix < eps || var_mov < eps)
      {
      if(ptr_gradient)
        ptr += 3 * ImageDimension;
      continue;
      }

    TPixel cov_fix_mov = x_fix_mov - x_fix * x_mov_over_n;
    TPixel one_over_denom = 1.0 / (var_fix * var_mov);
    TPixel cov_fix_mov_over_denom = cov_fix_mov * one_over_denom;
    TPixel ncc_fix_mov = cov_fix_mov * cov_fix_mov_over_denom;

    // Weight - includes scaling of squared covariance by direction
    TWeight w = (cov_fix_mov < 0) ? -weights[i_wgt] : weights[i_wgt];

    if(ptr_gradient)
      {
      for(int i = 0; i < ImageDimension; i++)
        {
        TPixel x_grad_mov_i = *ptr++;
        TPixel x_fix_grad_mov_i = *ptr++;
        TPixel x_mov_grad_mov_i = *ptr++;

        // Derivative of cov_fix_mov
        TPixel grad_cov_fix_mov_i = x_fix_grad_mov_i - x_fix_over_n * x_grad_mov_i;

        // One half derivative of var_mov
        TPixel half_grad_var_mov_i = x_mov_grad_mov_i - x_mov_over_n * x_grad_mov_i;

        TPixel grad_ncc_fix_mov_i =
            2 * cov_fix_mov_over_denom * (grad_cov_fix_mov_i - var_fix * half_grad_var_mov_i * cov_fix_mov_over_denom);

        (*ptr_gradient)[i] += w * grad_ncc_fix_mov_i;
        }
      }

    // Accumulate the metric
    *ptr_metric += w * ncc_fix_mov;
    }

  return ptr;
}


template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
void
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  itk::ThreadIdType threadId )
{  
  // Set up the iterators for the three images. In the future, check if the
  // iteration contributes in any way to the filter cost, and consider more
  // direct, faster approaches
  typedef itk::ImageLinearConstIteratorWithIndex<InputImageType> InputIteratorTypeBase;
  typedef IteratorExtender<InputIteratorTypeBase> InputIteratorType;
  typedef itk::ImageRegionIterator<TMetricImage> MetricIteratorType;
  typedef itk::ImageRegionIterator<TGradientImage> GradientIteratorType;

  InputImageType *image = const_cast<InputImageType *>(this->GetInput());
  InputIteratorType it_input(image, outputRegionForThread);

  // Get the mask image (optional)
  const MaskImageType *mask = dynamic_cast<MaskImageType *>(this->itk::ProcessObject::GetInput("mask"));

  // Number of input components
  int nc = this->GetInput()->GetNumberOfComponentsPerPixel();

  // Accumulated metric
  m_MetricPerThread[threadId] = 0.0;

  // Iterate over lines for greater efficiency
  for(; !it_input.IsAtEnd(); it_input.NextLine())
    {
    // Get the pointer to the start of the current line
    long offset_in_pixels = it_input.GetPosition() - image->GetBufferPointer();
    const InputComponentType *line_begin = image->GetBufferPointer() + offset_in_pixels * nc,
        *ptr = line_begin;
    const InputComponentType *line_end = ptr + outputRegionForThread.GetSize(0) * nc;

    // Get the offset into the metric and gradient images
    MetricPixelType *ptr_metric = this->GetMetricOutput()->GetBufferPointer() + offset_in_pixels;

    // The gradient output is optional
    GradientPixelType *ptr_gradient = (this->m_ComputeGradient)
                                      ? this->GetGradientOutput()->GetBufferPointer() + offset_in_pixels
                                      : NULL;

    // Four versions of the code - depending on the mask and gradient
    if(mask)
      {
      if(ptr_gradient)
        {
        // Get the offset into the mask, if a mask exists
        const MaskPixelType *ptr_mask = mask->GetBufferPointer() + offset_in_pixels;

        for(; ptr < line_end; ++ptr_metric, ++ptr_gradient, ++ptr_mask)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;
          *ptr_gradient = itk::NumericTraits<GradientPixelType>::Zero;

          // Should we skip this pixel?
          MaskPixelType mask_val = *ptr_mask;
          if(mask_val == 0)
            {
            ptr += nc;
            }
          else
            {
            // End of the chunk for this pixel
            const InputComponentType *ptr_end = ptr + nc;

            // Apply the post computation
            ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

            // Scale metric and gradient by the mask
            *ptr_metric *= mask_val;
            *ptr_gradient *= mask_val;

            // Accumulate the summary metric
            m_MetricPerThread[threadId] += *ptr_metric;
            }
          }
        }
      else
        {
        // Get the offset into the mask, if a mask exists
        const MaskPixelType *ptr_mask = mask->GetBufferPointer() + offset_in_pixels;

        for(; ptr < line_end; ++ptr_metric, ++ptr_mask)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;

          // Should we skip this pixel?
          MaskPixelType mask_val = *ptr_mask;
          if(mask_val == 0)
            {
            ptr += nc;
            }
          else
            {
            // End of the chunk for this pixel
            const InputComponentType *ptr_end = ptr + nc;

            // Apply the post computation
            ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

            // Scale metric and gradient by the mask
            *ptr_metric *= mask_val;

            // Accumulate the summary metric
            m_MetricPerThread[threadId] += *ptr_metric;
            }
          }
        }
      }
    else
      {
      if(ptr_gradient)
        {
        for(; ptr < line_end; ++ptr_metric, ++ptr_gradient)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;
          *ptr_gradient = itk::NumericTraits<GradientPixelType>::Zero;

          // End of the chunk for this pixel
          const InputComponentType *ptr_end = ptr + nc;

          // Apply the post computation
          ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

          // Accumulate the summary metric
          m_MetricPerThread[threadId] += *ptr_metric;
          }
        }
      else
        {
        for(; ptr < line_end; ++ptr_metric)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;

          // End of the chunk for this pixel
          const InputComponentType *ptr_end = ptr + nc;

          // Apply the post computation
          ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

          // Accumulate the summary metric
          m_MetricPerThread[threadId] += *ptr_metric;
          }
        }
      }
    }
}




#endif
