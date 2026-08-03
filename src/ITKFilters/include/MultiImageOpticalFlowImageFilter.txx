/*=========================================================================

  Program:   GreedyReg fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  GreedyReg initial development was funded by the NIH grant R01 EB017255.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

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
 * Compute the output for the region specified by outputRegionForThread.
 */
template <class TMetricTraits>
void
MultiImageOpticalFlowImageFilter<TMetricTraits>
::DynamicThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread)
{
  // Get the number of components
  int ncomp = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Create an iterator specialized for going through metrics
  typedef MultiComponentMetricWorker<TMetricTraits, MetricImageType> InterpType;
  InterpType iter(this, this->GetMetricOutput(), outputRegionForThread);

  // This is the accumulator of metric and gradient values for this region
  typename Superclass::ThreadAccumulatedData td(ncomp);

  // Iterate over the lines
  for(; !iter.IsAtEnd(); iter.NextLine())
    {
    // If we are computing deforamble gradient, get a pointer to the gradient line
    GradientPixelType *grad_line =
        (this->m_ComputeGradient && !this->m_ComputeAffine)
        ? this->GetDeformationGradientOutput()->GetBufferPointer() + iter.GetOffsetInPixels()
        : NULL;

    // Temporary gradient pixel
    GradientPixelType grad_metric;

    // Iterate over the pixels in the line
    for(; !iter.IsAtEndOfLine(); ++iter)
      {
      // Metric value at this pixel
      double metric = 0.0;

      // Gradient vector at this pixel.
      if(this->m_ComputeGradient)
        grad_metric.Fill(0.0);

      // Check the fixed mask at this location
      if(iter.CheckFixedMask())
        {
        // Interpolate the moving image at the current position. The worker knows
        // whether to interpolate the gradient or not
        typedef typename InterpType::InterpType FastInterpolator;
        typename FastInterpolator::InOut status = iter.Interpolate();

        // Logic splits on whether we want weighted or unweighted computation. For weighted computation,
        // hits outside of the moving mask / moving domain do not contribute to the metric at all. For
        // unweighted computation, locations outside of the moving mask / moving domain are assugmed to
        // have a fixed background intensity value (default being 0)
        if(this->m_Weighted)
          {
          itkAssertOrThrowMacro(false, "Weighted SSD metric not yet implemented");
          }
        else
          {
          // For unweighted computation, every visited pixel contributes 1.0 to the mask
          td.mask += 1.0;

          // There is some code duplication here, but it looks neater than having multiple if statements
          // and should run faster as well
          if(status == FastInterpolator::INSIDE)
            {
            for(int k = 0; k < ncomp; k++)
              {
              // Compute the metric
              double del = iter.GetFixedLine()[k] - iter.GetMovingSample()[k];
              double delw = this->m_Weights[k] * del;
              double del2w = delw * del;

              // Add the value to the metric
              metric += del2w;
              td.comp_metric[k] += del2w;

              if(this->m_ComputeGradient)
                {
                const RealType *grad_mov_k = iter.GetMovingSampleGradient(k);
                for(int i = 0; i < ImageDimension; i++)
                  grad_metric[i] += delw * grad_mov_k[i];
                }
              }
            }
          else if(status == FastInterpolator::OUTSIDE)
            {
            // Compute the metric
            for(int k = 0; k < ncomp; k++)
              {
              // Compute the metric
              double del = iter.GetFixedLine()[k] - this->m_BackgroundValue;
              double delw = this->m_Weights[k] * del;
              double del2w = delw * del;

              // Add the value to the metric
              metric += del2w;
              td.comp_metric[k] += del2w;
              }
            }
          else
            {
            // Compute the metric
            for(int k = 0; k < ncomp; k++)
              {
              // Compute the metric
              double mov_sample = iter.GetMovingSample()[k] + (1.0 - iter.GetMask()) * this->m_BackgroundValue;
              double del = iter.GetFixedLine()[k] - mov_sample;
              double delw = this->m_Weights[k] * del;
              double del2w = delw * del;

              // Add the value to the metric
              metric += del2w;
              td.comp_metric[k] += del2w;

              // Compute the gradient
              if(this->m_ComputeGradient)
                {
                const RealType *grad_mov_k = iter.GetMovingSampleGradient(k);
                for(int i = 0; i < ImageDimension; i++)
                  {
                  double d_mov_sample = grad_mov_k[i] - iter.GetMaskGradient()[i] * this->m_BackgroundValue;
                  grad_metric[i] += delw * d_mov_sample;
                  }
                }
              }
            }
          }

        // Add metric to the total
        td.metric += metric;

        // Complete gradient computation for affine registration
        if(this->m_ComputeGradient && this->m_ComputeAffine)
          {
          for(int i = 0, q = 0; i < ImageDimension; i++)
            {
            td.gradient[q++] += grad_metric[i];
            for(int j = 0; j < ImageDimension; j++)
              td.gradient[q++] += grad_metric[i] * iter.GetIndex()[j];
            }
          }

/*
          // If on the border, apply the mask
          if(status == FastInterpolator::BORDER && this->m_ComputeMovingDomainMask)
            {
            double mask = iter.GetMask();

            // Update the metric and the gradient vector to reflect multiplication with mask
            for(int i = 0; i < ImageDimension; i++)
              grad_metric[i] = grad_metric[i] * mask - 0.5 * iter.GetMaskGradient()[i] * metric;
            metric = metric * mask;

            td.mask += mask;
            }
          else
            {
            td.mask += 1.0;
            }

          // Accumulate the metric
          td.metric += metric;

          // Do the gradient computation
          if(this->m_ComputeGradient)
            {
            // If affine, use the gradient to accumulte the affine partial derivs
            if(this->m_ComputeAffine)
              {
              for(int i = 0, q = 0; i < ImageDimension; i++)
                {
                td.gradient[q++] += grad_metric[i];
                for(int j = 0; j < ImageDimension; j++)
                  td.gradient[q++] += grad_metric[i] * iter.GetIndex()[j];
                }

              if(status == FastInterpolator::BORDER)
                {
                for(int i = 0, q = 0; i < ImageDimension; i++)
                  {
                  td.grad_mask[q++] += iter.GetMaskGradient()[i];
                  for(int j = 0; j < ImageDimension; j++)
                    td.grad_mask[q++] += iter.GetMaskGradient()[i] * iter.GetIndex()[j];
                  }
                }
              }
            }
          } // not outside */
        } // check fixed mask

      // Last thing - update the output voxels
      *iter.GetOutputLine() += metric;
      if(grad_line)
        grad_line[iter.GetLinePos()] += grad_metric;
      }
    }

  // Update the accumulated values in a thread-safe way
  this->m_AccumulatedData.Accumulate(td);
}


#endif
