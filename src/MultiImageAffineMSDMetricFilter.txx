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
template <class TMetricTraits>
void
MultiImageAffineMetricFilter<TMetricTraits>
::BeforeThreadedGenerateData()
{
  // Initialize the per thread data
  m_ThreadData.resize(this->GetNumberOfThreads(), ThreadData());
}

/**
 * Setup state of filter after multi-threading.
 */
template <class TMetricTraits>
void
MultiImageAffineMetricFilter<TMetricTraits>
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

  // Initialize the output metric gradient
  m_MetricGradient = TransformType::New();

  // Compute the objective value
  // m_MetricValue = summary.metric / summary.mask;
  m_MetricValue = summary.metric;

  // Compute the gradient
  vnl_vector<double> grad_metric(summary.gradient.size());
  for(int j = 0; j < summary.gradient.size(); j++)
    {
    // grad_metric[j] =
       // (-2.0 * summary.gradient[j] - m_MetricValue * summary.grad_mask[j]) / summary.mask;
    grad_metric[j] = m_GradientScalingFactor * summary.gradient[j];
    }

  // Pack into the output
  unflatten_affine_transform(grad_metric.data_block(), m_MetricGradient.GetPointer());
}

template <class TMetricTraits>
void
MultiImageAffineMetricFilter<TMetricTraits>
::GenerateInputRequestedRegion()
{
  // Call the superclass's implementation
  Superclass::GenerateInputRequestedRegion();

  // Set all regions to max
  this->GetMetricImage()->SetRequestedRegionToLargestPossibleRegion();
  this->GetMovingDomainMaskImage()->SetRequestedRegionToLargestPossibleRegion();

  if(m_ComputeGradient)
    {
    this->GetGradientImage()->SetRequestedRegionToLargestPossibleRegion();
    this->GetMovingDomainMaskGradientImage()->SetRequestedRegionToLargestPossibleRegion();
    }
}

template <class TMetricTraits>
void
MultiImageAffineMetricFilter<TMetricTraits>
::EnlargeOutputRequestedRegion(itk::DataObject *data)
{
  Superclass::EnlargeOutputRequestedRegion(data);
  data->SetRequestedRegionToLargestPossibleRegion();
}


template <class TMetricTraits>
void
MultiImageAffineMetricFilter<TMetricTraits>
::AllocateOutputs()
{
  // Propagate input
  this->GraftOutput(this->GetMetricImage());
}


template <class TMetricTraits>
void
MultiImageAffineMetricFilter<TMetricTraits>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  itk::ThreadIdType threadId )
{
  // Get the pointers to the input and output images
  InputImageType *metric = this->GetMetricImage();
  InputImageType *mask = this->GetMovingDomainMaskImage();

  // Get the pointers to the buffers of these four images
  const InputPixelType *b_metric = metric->GetBufferPointer();
  const InputPixelType *b_mask = mask->GetBufferPointer();

  const GradientPixelType *b_gradient =
      m_ComputeGradient ? this->GetGradientImage()->GetBufferPointer() : NULL;

  const GradientPixelType *b_mask_gradient =
      m_ComputeGradient ? this->GetMovingDomainMaskGradientImage()->GetBufferPointer() : NULL;

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef itk::ImageLinearConstIteratorWithIndex<InputImageType> IterBase;
  typedef IteratorExtender<IterBase> Iter;

  // The thread data to accumulate
  ThreadData &td = m_ThreadData[threadId];

  // Gradient accumulator
  vnl_vector_fixed<double, ImageDimension> grad, gradM;

  // Iterate over the fixed space region
  for(Iter it(metric, outputRegionForThread); !it.IsAtEnd(); it.NextLine())
    {
    // Process the whole line using pointer arithmetic. We have to deal with messy behavior
    // of iterators on vector images. Trying to avoid using accessors and Set/Get
    long offset_in_pixels = it.GetPosition() - b_metric;

    // Get pointers to the start of the line
    const InputPixelType *p_metric = b_metric + offset_in_pixels;
    const InputPixelType *p_mask = b_mask + offset_in_pixels;

    // Loop over the line
    const InputPixelType *p_metric_end = p_metric + outputRegionForThread.GetSize(0);

    // Loop if we are computing gradient
    if(m_ComputeGradient)
      {
      const GradientPixelType *p_gradient = b_gradient + offset_in_pixels;
      const GradientPixelType *p_mask_gradient = b_mask_gradient + offset_in_pixels;

      // Get the index at the current location
      IndexType idx = it.GetIndex();

      // Do we need the gradient?
      for(; p_metric < p_metric_end; ++p_metric, ++p_mask, ++p_gradient, ++p_mask_gradient, ++idx[0])
        {
        // Accumulators for the gradients
        double *out_grad = td.gradient.data_block();
        double *out_grad_mask = td.grad_mask.data_block();

        // For border regions, we need to explicitly deal with the mask
        const InputPixelType &metric = *p_metric;
        const InputPixelType &mask = *p_mask;
        const GradientPixelType &grad = *p_gradient;
        const GradientPixelType &gradM = *p_mask_gradient;

        // Compute the mask and metric gradient contributions
        /*
        for(int i = 0; i < ImageDimension; i++)
          {
          double v = grad[i] * mask - 0.5 * gradM[i] * metric;
          *(out_grad++) += v;
          *(out_grad_mask++) += gradM[i];
          for(int j = 0; j < ImageDimension; j++)
            {
            *(out_grad++) += v * idx[j];
            *(out_grad_mask++) += gradM[i] * idx[j];
            }
          }

        td.metric += metric * mask;
        td.mask += mask;*/


        for(int i = 0; i < ImageDimension; i++)
          {
          double v = grad[i];
          *(out_grad++) += v;
          for(int j = 0; j < ImageDimension; j++)
            {
            *(out_grad++) += v * idx[j];
            }
          }

        td.metric += metric;
        }
      }
    else
      {
      // Do we need the gradient?
      for(; p_metric < p_metric_end; ++p_metric, ++p_mask)
        {
        // For border regions, we need to explicitly deal with the mask
        const InputPixelType &metric = *p_metric;
        const InputPixelType &mask = *p_mask;
        td.metric += metric * mask;
        td.mask += mask;
        }
      }
    }
}


#endif
