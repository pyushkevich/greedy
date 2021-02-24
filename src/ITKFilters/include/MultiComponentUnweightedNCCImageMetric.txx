/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#ifndef __MultiComponentUnweightedNCCImageMetric_txx
#define __MultiComponentUnweightedNCCImageMetric_txx

#include "MultiComponentUnweightedNCCImageMetric.h"
#include "OneDimensionalInPlaceAccumulateFilter.h"
#include "itkImageFileWriter.h"


template <class TMetricTraits>
void
MultiComponentUnweightedNCCImageMetric<TMetricTraits>
::PrecomputeAccumulatedComponents(const OutputImageRegionType &outputRegionForThread)
{
  // Create an iterator specialized for going through metrics
  typedef MultiComponentMetricWorker<TMetricTraits, InputImageType> InterpType;
  typedef typename InterpType::InterpType FastInterpolator;
  InterpType iter(this, this->m_WorkingImage, outputRegionForThread);

  // Iterate over the lines
  for(; !iter.IsAtEnd(); iter.NextLine())
    {
    // Iterate over the pixels in the line
    for(; !iter.IsAtEndOfLine(); ++iter)
      {
      // Get the output pointer for this voxel
      InputComponentType *accum = iter.GetOutputLine();
      InputComponentType *saved = accum + m_FirstPassAccumComponents;
      typename FastInterpolator::InOut status;

      // TODO: do we need to go over points with mask of 0.5 or mask of 1.0?
      if(!iter.CheckFixedMask(0.0))
        {
        for(int j = 0; j < m_TotalWorkingImageComponents; j++)
          *accum++ = 0.0;
        }
      else
        {
        // Interpolate
        status = iter.Interpolate();

        // Simpler, internal case, weight is 1
        *accum++ = 1;

        // Iterate over the components
        for(int k = 0; k < m_InputComponents; k++)
          {
          // The fixed value is just the moving component
          InputComponentType x_fix = iter.GetFixedLine()[k];

          // What we sample from the moving image is actually moving image intensity times the
          // moving image mask
          InputComponentType x_mov = iter.GetMovingSample()[k];

          // Write the five components that are averaged in the cross-correlation computation
          *accum++ = x_fix;                         // f
          *accum++ = x_mov;                         // m
          *accum++ = x_fix * x_fix;                 // f^2
          *accum++ = x_mov * x_mov;                 // m^2
          *accum++ = x_fix * x_mov;                 // f * m

          // Store elements needed for gradient computation
          *saved++ = x_fix;
          *saved++ = x_mov;
          const InputComponentType *x_mov_grad = iter.GetMovingSampleGradient(k);
          for(unsigned int d = 0; d < ImageDimension; d++)
            *saved++ = x_mov_grad[d];
          }
        }
      }
    }
}

template <class TMetricTraits>
void
MultiComponentUnweightedNCCImageMetric<TMetricTraits>
::ComputeNCCAndGradientAccumulatedComponents(const OutputImageRegionType &outputRegionForThread)
{
  // This is the second pass, where we compute NCC and also place into the working image
  // additional terms to accumulate for gradient computation
  const double eps = 1.0e-2;

  // Data to accumulate in our thread
  typename Superclass::ThreadAccumulatedData td(m_InputComponents);

  // Where to store the accumulated metric (gets copied to td, but should have TPixel type)
  vnl_vector<InputComponentType> comp_metric(m_InputComponents, 0.0);

  // Radius size (i.e., N for variance computations)
  double patch_size = 1;
  for(unsigned int d = 0; d < ImageDimension; d++)
    patch_size *= (1 + 2 * m_Radius[d]);

  /*
  // Added in Feb 2020, part of NaN masking. We now scale the patch NCC by the
  // number of non-background voxels in the mask, to prevent oversize contribution
  // of border pixels. To account for this, the weights are scaled by the patch size
  double one_over_patch_size = 1.0;
  for(unsigned int k = 0; k < ImageDimension; k++)
    one_over_patch_size /= (1 + 2.0 * this->m_Radius[k]);
  typename Superclass::WeightVectorType wgt_scaled = this->m_Weights * one_over_patch_size;
  */

  // Set up an iterator for the working image (which contains accumulation results)
  InputIteratorType it(m_WorkingImage, outputRegionForThread);

  // Loop over the lines
  for (; !it.IsAtEnd(); it.NextLine())
    {
    // Get the pointer to the input line
    long offset_in_pixels = it.GetPosition() - m_WorkingImage->GetBufferPointer();

    // Pointer to the input pixel data for this line
    InputComponentType *p_work =
        m_WorkingImage->GetBufferPointer() + m_TotalWorkingImageComponents * offset_in_pixels;

    // Pointer to the output metric image line
    MetricPixelType *p_metric =
        this->GetMetricOutput()->GetBufferPointer() + offset_in_pixels;

    // Loop over the pixels in the line
    for(int i = 0; i < outputRegionForThread.GetSize()[0]; ++i)
      {
      // Clear the metric output
      *p_metric = itk::NumericTraits<MetricPixelType>::ZeroValue();

      // Read the number of in-mask pixels considered
      double n = p_work[0];

      // If zero, there is no data for this pixel
      if(n == 0)
        {
        // Clear the accumulation data for this pixel and continue to the next
        if(m_NeedGradient)
          for(unsigned int k = 0; k < m_SecondPassAccumComponents; k++)
            p_work[k+1] = 0.0;
        }
      else
        {
        // Iterate over the components
        InputComponentType *p_accum = p_work + 1;
        InputComponentType *p_accum_out = p_work + 1;
        for(unsigned int k = 0; k < m_InputComponents; k++)
          {
          // We are reusing the working image, replacing elements 3-5 with new items that
          // must be accumulated
          double sum_f =  *p_accum++;
          double sum_m =  *p_accum++;
          double sum_ff = *p_accum++;
          double sum_mm = *p_accum++;
          double sum_fm = *p_accum++;

          // Compute the weighted normalized correlation, it has a nice symmetrical formula;
          // However, to avoid division by zero issues, we should add epsilon to the components
          double var_f  = patch_size * sum_ff - sum_f * sum_f + eps;
          double var_m  = patch_size * sum_mm - sum_m * sum_m + eps;
          double cov_fm = patch_size * sum_fm - sum_f * sum_m;

          // This is to preserve the direction of the metric
          double abs_cov_fm = cov_fm < 0 ? -cov_fm : cov_fm;

          // Compute the signed square correlation coefficient.
          double one_over_denom = 1.0 / (var_f * var_m);
          double ncc_fm = abs_cov_fm * cov_fm * one_over_denom;

          // Scale by the weight
          // TODO: work out sign later
          // double w_comp = (cov_fm < 0) ? -comp_weights[i] : comp_weights[i];
          double w_comp = this->m_Weights[k];

          // We use sum_w as an additional scaling factor so that the contribution of
          // border pixels is reduced (seems like a fair way to do things)
          // MetricPixelType weighted_metric = (MetricPixelType) (w_comp * n * ncc_fm);
          MetricPixelType weighted_metric = (MetricPixelType) w_comp * ncc_fm;

          // Compute the derivatives
          if(m_NeedGradient)
            {
            // Fill out the quantities accumulated on the second (gradient) pass
            double q1 = abs_cov_fm * one_over_denom;
            double q2 = ncc_fm / var_m;
            *p_accum_out++ = patch_size * q1;
            *p_accum_out++ = - patch_size * q2;
            *p_accum_out++ = sum_m * q2 - sum_f * q1;
            }

          // Store the componentwise metric
          comp_metric[k] += weighted_metric;

          // Write the metric value to the metric image
          *p_metric += weighted_metric;
          } // Loop over components
        }

      p_work += m_TotalWorkingImageComponents;
      p_metric++;
      } // Iterate over pixels in line
    } // Iterate over lines

  // Typecast the per-component metrics
  for(unsigned int a = 0; a < m_InputComponents; a++)
    td.comp_metric[a] = comp_metric[a];

  // Accumulate this region's data in a thread-safe way
  this->m_AccumulatedData.Accumulate(td);
}

template <class TMetricTraits>
void
MultiComponentUnweightedNCCImageMetric<TMetricTraits>
::ComputeNCCGradient(const OutputImageRegionType &outputRegionForThread)
{
  // Data to accumulate in our thread
  typename Superclass::ThreadAccumulatedData td(m_InputComponents);

  // Set up an iterator for the working image (which contains accumulation results)
  InputIteratorType it(m_WorkingImage, outputRegionForThread);

  // Loop over the lines
  for (; !it.IsAtEnd(); it.NextLine())
    {
    // Get the pointer to the input line
    long offset_in_pixels = it.GetPosition() - m_WorkingImage->GetBufferPointer();

    // Pointer to the input pixel data for this line
    const InputComponentType *p_work =
        m_WorkingImage->GetBufferPointer() + m_TotalWorkingImageComponents * offset_in_pixels;

    // Pointer to the output metric gradient image line
    GradientPixelType *p_grad_metric =
        this->GetDeformationGradientOutput()->GetBufferPointer() + offset_in_pixels;

    // Loop over the pixels in the line
    for(int i = 0; i < outputRegionForThread.GetSize()[0]; ++i)
      {
      // Clear the metric output
      *p_grad_metric = itk::NumericTraits<GradientPixelType>::ZeroValue();

      // Read the number of in-mask pixels considered
      double n = p_work[0];

      // If zero, there is no data for this pixel
      if(n > 0)
        {
        const InputComponentType *p_accum = p_work + 1;
        const InputComponentType *p_saved = p_work + m_FirstPassAccumComponents;
        for(unsigned int k = 0; k < m_InputComponents; k++)
          {
          double x_fix = *p_saved++;
          double x_mov = *p_saved++;
          double z1 = (*p_accum++) * x_fix;
          double z2 = (*p_accum++) * x_mov;
          double z3 = (*p_accum++);

          // This is the partial derivative of the accumulated metric with respect
          // to the moving pixel intensity. It should be multiplied by the gradient
          // of moving intensity with respect to phi
          double d_metric_d_x_mov = 2 * (z1 + z2 + z3);

          double w_comp = this->m_Weights[k];
          d_metric_d_x_mov *= w_comp; // * n;

          for(unsigned int d = 0; d < ImageDimension; d++)
            {
            double dMk_dPhi_d = *p_saved++;
            (*p_grad_metric)[d] += dMk_dPhi_d * d_metric_d_x_mov;
            }
          }
        }

      p_work += m_TotalWorkingImageComponents;
      p_grad_metric++;
      }
    }
}

template <class TMetricTraits>
void
MultiComponentUnweightedNCCImageMetric<TMetricTraits>
::AccumulateWorkingImageComponents(unsigned int comp_begin, unsigned int comp_end)
{
  // Accumulate
  unsigned int skip_end = m_WorkingImage->GetNumberOfComponentsPerPixel() - comp_end;
  typename InputImageType::Pointer img_accum =
      AccumulateNeighborhoodSumsInPlace(m_WorkingImage.GetPointer(), m_Radius, comp_begin, skip_end);

  // Reassign to the working image
  img_accum->DisconnectPipeline();
  m_WorkingImage = img_accum;
}

template <class TMetricTraits>
void
MultiComponentUnweightedNCCImageMetric<TMetricTraits>
::GenerateData()
{
  // Working image is required
  itkAssertOrThrowMacro(m_WorkingImage, "Working image missing in MultiComponentUnweightedNCCImageMetric");

  // Call the requisite methods
  Superclass::AllocateOutputs();
  Superclass::BeforeThreadedGenerateData();

  // Sort out how many components will be involved in accumulation
  m_InputComponents = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Gradient needed?
  m_NeedGradient = this->GetComputeGradient();

  // The working memory is organized as follows. At the front are components that
  // are accumulated for computing NCC. There are 1 + 5 *nc of these components. At
  // the back are components that are retained for gradient computation and not
  // accumulated. These are f, m, and grad_M.
  m_FirstPassAccumComponents = 1 + 5 * m_InputComponents;
  m_FirstPassSavedComponents = m_NeedGradient ? (2 + ImageDimension) * m_InputComponents : 0;
  m_SecondPassAccumComponents = 3 * m_InputComponents;
  m_TotalWorkingImageComponents = m_FirstPassAccumComponents + m_FirstPassSavedComponents;

  // Main buffered region
  // TODO: make it possible to split the region into pieces to conserve memory
  auto working_region = this->GetFixedImage()->GetBufferedRegion();

  // If the working image is supplied, make sure that it has sufficient size
  // Check if the working image needs to be allocated
  if(m_WorkingImage->GetBufferedRegion() != working_region
     || m_WorkingImage->GetNumberOfComponentsPerPixel() < m_TotalWorkingImageComponents)
    {
    // Configure the working image
    m_WorkingImage->CopyInformation(this->GetFixedImage());
    m_WorkingImage->SetNumberOfComponentsPerPixel(m_TotalWorkingImageComponents);
    m_WorkingImage->SetRegions(working_region);
    m_WorkingImage->Allocate();
    }

  // Fill the working image with quanitities that need to be accumulated
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  mt->ParallelizeImageRegion<Self::ImageDimension>(
        working_region,
        [this](const OutputImageRegionType &outputRegionForThread)
    {
    this->PrecomputeAccumulatedComponents(outputRegionForThread);
    }, nullptr);

#ifdef DUMP_NCC
  typename itk::ImageFileWriter<InputImageType>::Pointer pwriter = itk::ImageFileWriter<InputImageType>::New();
  pwriter->SetInput(m_WorkingImage);
  pwriter->SetFileName("nccpre.nii.gz");
  pwriter->Update();
#endif

  // It is feasible that the working image has been allocated for more components that
  // are currently used. In this case, we skip those components at the end
  this->AccumulateWorkingImageComponents(0, m_FirstPassAccumComponents);

#ifdef DUMP_NCC
  typename itk::ImageFileWriter<InputImageType>::Pointer pwriter2 = itk::ImageFileWriter<InputImageType>::New();
  pwriter2->SetInput(img_accum);
  pwriter2->SetFileName("nccaccum.nii.gz");
  pwriter2->Update();
#endif

  // At this point, the working image will hold the proper neighborhood sums (I, J, I^2, J^2, IJ, etc).
  // The next step is to use this information to compute the NCC and the quantities that must be
  // accumulated for the second pass through the filter
  mt->ParallelizeImageRegion<Self::ImageDimension>(
        working_region,
        [this](const OutputImageRegionType &outputRegionForThread)
    {
    this->ComputeNCCAndGradientAccumulatedComponents(outputRegionForThread);
    }, nullptr);

  // The next step is only done if computing gradients, and involves another layer of accumulation,
  // this time only using 3 values per component
  if(m_NeedGradient)
    {
    // Accumulate for the second pass
    this->AccumulateWorkingImageComponents(1, m_SecondPassAccumComponents+1);

    // Last parallel operation, in which the gradients are computed
    mt->ParallelizeImageRegion<Self::ImageDimension>(
          working_region,
          [this](const OutputImageRegionType &outputRegionForThread)
      {
      this->ComputeNCCGradient(outputRegionForThread);
      }, nullptr);

    #ifdef DUMP_NCC
    typename itk::ImageFileWriter<GradientImageType>::Pointer pwriter3 = itk::ImageFileWriter<GradientImageType>::New();
    pwriter3->SetInput(this->GetDeformationGradientOutput());
    pwriter3->SetFileName("nccgrad.mhd");
    pwriter3->Update();
    #endif
    }

  Superclass::AfterThreadedGenerateData();
}


#endif // __MultiComponentUnweightedNCCImageMetric_txx

