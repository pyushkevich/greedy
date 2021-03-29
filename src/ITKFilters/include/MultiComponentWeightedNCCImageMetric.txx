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
#ifndef __MultiComponentWeightedNCCImageMetric_txx
#define __MultiComponentWeightedNCCImageMetric_txx

#include "MultiComponentWeightedNCCImageMetric.h"
#include "OneDimensionalInPlaceAccumulateFilter.h"
#include "itkImageFileWriter.h"


template <class TMetricTraits>
void
MultiComponentWeightedNCCImageMetric<TMetricTraits>
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
      InputComponentType *saved = accum + m_SavedComponentsOffset;

      // The fixed mask does not change through the iterations and can be safely ignored.
      // We don't even need to clear the accumulated values because they are zeroed out
      // when the working image is allocated
      if(iter.CheckFixedMask(0.0))
        {
        // Interpolate at this location
        typename FastInterpolator::InOut status = iter.Interpolate();

        // Split on weighted vs. unweighted mode, since the accumulated quantities
        // are going to be different
        if(m_Weighted)
          {
          // If status is outside, we must zero out all accumulated elements
          if(status == FastInterpolator::OUTSIDE ||
             (status == FastInterpolator::BORDER && iter.GetMask() == 0))
            {
            for(unsigned int q = 0; q < m_TotalWorkingImageComponents; q++)
              *accum++ = 0.0;
            }
          else
            {
            // TODO: what about the fixed mask contribution?
            double w;

            // Get and save the weight and save the weight gradient. The weight is one and
            // weight gradient is zero for inside voxels
            if(status == FastInterpolator::INSIDE)
              {
              w = 1.0;
              if(m_NeedGradient)
                {
                *saved++ = w;
                for(unsigned int d = 0; d < ImageDimension; d++)
                  *saved++ = 0.0;
                }
              }
            else
              {
              w = iter.GetMask();
              if(m_NeedGradient)
                {
                *saved++ = w;
                for(unsigned int d = 0; d < ImageDimension; d++)
                  *saved++ = iter.GetMaskGradient()[d];
                }
              }

            // Accumulate the weight
            *accum++ = w;

            // Iterate over the components
            for(int k = 0; k < m_InputComponents; k++)
              {
              // The fixed value is just the moving component
              InputComponentType x_fix = iter.GetFixedLine()[k];

              // What we sample from the moving image is actually moving image intensity times the
              // moving image mask
              InputComponentType x_wmov = iter.GetMovingSample()[k];

              // Compute the weighted fixed and unweighted moving values
              double x_wfix = w * x_fix, x_mov = x_wmov / w;

              // Write the five components that are averaged in the cross-correlation computation
              *accum++ = x_wfix;                         // w * f
              *accum++ = x_wmov;                         // w * m
              *accum++ = x_wfix * x_fix;                 // w * f^2
              *accum++ = x_wmov * x_mov;                 // w * m^2
              *accum++ = x_fix * x_wmov;                 // w * f * m

              // Store elements needed for gradient computation
              if(m_NeedGradient)
                {
                *saved++ = x_fix;
                *saved++ = x_mov;
                const InputComponentType *x_mov_grad = iter.GetMovingSampleGradient(k);
                for(unsigned int d = 0; d < ImageDimension; d++)
                  *saved++ = x_mov_grad[d];
                }
              }
            }
          }
        else // i.e., m_Weighted is false
          {
          // The pixel is counted no matter what the interpolation status
          *accum++ = 1.0;

          // Handle the special case of outside status. Here the moving image is assumed
          // to be zero, but the fixed image should still be recorded in the accumulation
          if(status == FastInterpolator::OUTSIDE)
            {
            // Iterate over the components
            for(int k = 0; k < m_InputComponents; k++)
              {
              // The fixed value is just the moving component
              InputComponentType x_fix = iter.GetFixedLine()[k];

              // Write the five components that are averaged in the cross-correlation computation
              *accum++ = x_fix;                         // f
              *accum++ = 0.0;                           // m
              *accum++ = x_fix * x_fix;                 // f^2
              *accum++ = 0.0;                           // m^2
              *accum++ = 0.0;                           // f * m

              // Store elements needed for gradient computation
              if(m_NeedGradient)
                {
                *saved++ = x_fix;
                *saved++ = 0.0;
                for(unsigned int d = 0; d < ImageDimension; d++)
                  *saved++ = 0.0;
                }
              }
            }
          else // status is inside or border
            {
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
              if(m_NeedGradient)
                {
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
      }
    }
}

template <class TMetricTraits>
void
MultiComponentWeightedNCCImageMetric<TMetricTraits>
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

  // Added in Feb 2020, part of NaN masking. We now scale the patch NCC by the
  // number of non-background voxels in the mask, to prevent oversize contribution
  // of border pixels. To account for this, the weights are scaled by the patch size
  double one_over_patch_size = 1.0;
  for(unsigned int k = 0; k < ImageDimension; k++)
    one_over_patch_size /= (1 + 2.0 * this->m_Radius[k]);

  // This array holds the components accumulated on the second pass. This is only needed
  // for WNCC with multiple components
  bool need_accum_scratch_space = m_Weighted && m_InputComponents > 1;
  InputComponentType *out_accum_line =
      need_accum_scratch_space ? new InputComponentType[m_SecondPassAccumComponents] : nullptr;

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

    // Pointer to the fixed mask - we only compute NCC at these locations
    typename MaskImageType::PixelType *fixed_mask_line =
        this->GetFixedMaskImage()
        ? this->GetFixedMaskImage()->GetBufferPointer() + offset_in_pixels
        : nullptr;

    // Pointer to the output metric image line
    MetricPixelType *p_metric =
        this->GetMetricOutput()->GetBufferPointer() + offset_in_pixels;

    // Loop over the pixels in the line
    for(int i = 0; i < outputRegionForThread.GetSize()[0]; ++i)
      {
      if(i == 51 && it.GetIndex()[1] == 32)
        std::cout << "here we are " << std::endl;

      // Clear the metric output
      *p_metric = itk::NumericTraits<MetricPixelType>::ZeroValue();

      // Fixed mask must be 1.0 indicating that NCC is being measured at this point
      double f_mask = fixed_mask_line? *fixed_mask_line++ : 1.0;

      // Add the fixed mask to the total
      if(f_mask >= 1.0)
        td.mask++;

      // Read the number of in-mask pixels considered
      double sum_w = p_work[0];

      // If mask is below one or weighted mask adds up to zero, there is no data for this pixel
      if(f_mask < 1.0 || sum_w == 0)
        {
        // Clear the accumulation data for this pixel and continue to the next
        if(m_NeedGradient)
          for(unsigned int k = 0; k < m_SecondPassAccumComponents; k++)
            p_work[k+1] = 0.0;
        }
      else
        {
        // How we calculate variance and covariance depends on weighting
        double N = patch_size, w_scale = 1.0;
        if(m_Weighted)
          {
          N = sum_w;
          w_scale = std::pow(sum_w * one_over_patch_size, m_WeightScalingExponent);
          }

        // Iterate over the components
        InputComponentType *p_accum = p_work + 1;
        InputComponentType *p_accum_out = need_accum_scratch_space ? out_accum_line : p_accum;
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
          double var_f  = N * sum_ff - sum_f * sum_f + eps;
          double var_m  = N * sum_mm - sum_m * sum_m + eps;
          double cov_fm = N * sum_fm - sum_f * sum_m;

          // This is to preserve the direction of the metric
          double abs_cov_fm = cov_fm < 0 ? -cov_fm : cov_fm;

          // Compute the signed square correlation coefficient.
          double one_over_denom = 1.0 / (var_f * var_m);
          double ncc_fm = abs_cov_fm * cov_fm * one_over_denom;

          // Scale ncc by weight
          double ncc_fm_w_scaled = w_scale * ncc_fm;

          // Scale by the weight
          double w_comp = this->m_Weights[k];

          // We use sum_w as an additional scaling factor so that the contribution of
          // border pixels is reduced (seems like a fair way to do things)
          // MetricPixelType weighted_metric = (MetricPixelType) (w_comp * n * ncc_fm);
          MetricPixelType weighted_metric = (MetricPixelType) w_comp * w_scale * ncc_fm;

          // Store the componentwise metric
          comp_metric[k] += weighted_metric;

          // Write the metric value to the metric image
          *p_metric += weighted_metric;

          // Fill out the quantities accumulated on the second (gradient) pass
          if(m_NeedGradient)
            {
            // These common quantities are just the NCC^2 metric divided by either
            // covariance, or one of the variances
            double q_fm = w_scale * abs_cov_fm * one_over_denom;
            double q_m  = ncc_fm_w_scaled / var_m;

            if(m_Weighted)
              {
              double q_f  = ncc_fm_w_scaled / var_f;
              *p_accum_out++ = sum_w * q_fm;
              *p_accum_out++ = sum_w * q_f;
              *p_accum_out++ = sum_w * q_m;
              *p_accum_out++ = sum_m * q_m - sum_f * q_fm;
              *p_accum_out++ = sum_f * q_f - sum_m * q_fm;
              *p_accum_out++ = 2 * sum_fm * q_fm - sum_ff * q_f - sum_mm * q_m
                               + m_WeightScalingExponent * ncc_fm_w_scaled / sum_w;
              }
            else
              {
              // Fill out the quantities accumulated on the second (gradient) pass
              *p_accum_out++ = patch_size * q_fm;
              *p_accum_out++ = patch_size * q_m;
              *p_accum_out++ = sum_m * q_m - sum_f * q_fm;
              }
            }
          } // Loop over components

        // Copy the out_accum_line into the working image
        if(need_accum_scratch_space)
          {
          InputComponentType *p_src = out_accum_line, *p_trg = p_work+1, *p_trg_end = p_trg+m_SecondPassAccumComponents;
          while(p_trg < p_trg_end)
              *p_trg++ = *p_src++;
          }

        }

      p_work += m_TotalWorkingImageComponents;
      p_metric++;
      } // Iterate over pixels in line
    } // Iterate over lines

  // Delete scratch space
  if(need_accum_scratch_space)
    delete[] out_accum_line;

  // Typecast the per-component metrics
  for(unsigned int a = 0; a < m_InputComponents; a++)
    {
    td.comp_metric[a] = comp_metric[a];
    td.metric += comp_metric[a];
    }

  // Accumulate this region's data in a thread-safe way
  this->m_AccumulatedData.Accumulate(td);
}

template <class TMetricTraits>
void
MultiComponentWeightedNCCImageMetric<TMetricTraits>
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

    // Pointer to the fixed mask - gradient should only be computed in the vincinity of the mask
    typename MaskImageType::PixelType *fixed_mask_line =
        this->GetFixedMaskImage()
        ? this->GetFixedMaskImage()->GetBufferPointer() + offset_in_pixels
        : nullptr;

    // Pointer to the output metric gradient image line
    GradientPixelType *p_grad_metric =
        this->GetDeformationGradientOutput() ?
        this->GetDeformationGradientOutput()->GetBufferPointer() + offset_in_pixels
        : nullptr;

    // Get the index (for affine gradient computation)
    itk::ContinuousIndex<double, ImageDimension> cix(it.GetIndex());

    // Loop over the pixels in the line
    for(int i = 0; i < outputRegionForThread.GetSize()[0]; ++i)
      {
      // Clear the metric output
      if(p_grad_metric)
        *p_grad_metric = itk::NumericTraits<GradientPixelType>::ZeroValue();

      // Fixed mask must be 1.0 indicating that NCC is being measured at this point
      double f_mask = fixed_mask_line? *fixed_mask_line++ : 1.0;

      // Read the number of in-mask pixels considered
      double sum_w = p_work[0];

      // If zero, there is no data for this pixel
      if(f_mask > 0 && sum_w > 0)
        {
        const InputComponentType *p_accum = p_work + 1;
        const InputComponentType *p_saved = p_work + m_SavedComponentsOffset;

        if(m_Weighted)
          {
          // Read the saved weight and gradient that are at the start of saved data
          double w = *p_saved++;
          auto* grad_w = p_saved; p_saved += ImageDimension;

          // Iterate over components
          for(unsigned int k = 0; k < m_InputComponents; k++)
            {
            double x_fix = *p_saved++;
            double x_mov = *p_saved++;

            // Read the six accumulated quantities
            double z1 = *p_accum++, z2 = *p_accum++, z3 = *p_accum++;
            double z4 = *p_accum++, z5 = *p_accum++, z6 = *p_accum++;

            // This is the partial derivative of the accumulated metric with respect
            // to the moving pixel intensity.
            double d_metric_d_x_mov_w =
                2. * (x_fix * z1 - x_mov * z3 + z4);

            // This is the partial derivative of the accumulated metric for this
            // component with respect to the weight vector
            double d_metric_d_w =
                2. * (x_fix * z5 + x_mov * z4 + x_fix * x_mov * z1)
                - x_fix * x_fix * z2
                - x_mov * x_mov * z3
                + z6;

            // Deal with component weights
            double w_comp = this->m_Weights[k];

            // Compute multipliers for the spatial gradients of WM and W
            double mult_grad_wm = w_comp * d_metric_d_x_mov_w;
            double mult_grad_w = w_comp * (d_metric_d_w - x_mov * d_metric_d_x_mov_w);

            // TODO: Scaling by n?

            // Compute the gradient
            double *p_affine_grad = this->m_ComputeAffine ? td.gradient.data_block() : nullptr;
            for(unsigned int d = 0; d < ImageDimension; d++)
              {
              double grad_wm_d = *p_saved++;
              double d_metric_d_phi_k = grad_wm_d * mult_grad_wm + grad_w[d] * mult_grad_w;

              // Store the metric
              if(p_grad_metric)
                (*p_grad_metric)[d] += d_metric_d_phi_k;

              // Compute affine transform terms
              if(p_affine_grad)
                {
                *p_affine_grad++ += d_metric_d_phi_k;
                *p_affine_grad++ += d_metric_d_phi_k * i;
                for(unsigned int q = 1; q < ImageDimension; q++)
                  *p_affine_grad++ += d_metric_d_phi_k * cix[q];
                }
              }
            }

          }
        else
          {
          for(unsigned int k = 0; k < m_InputComponents; k++)
            {
            double x_fix = *p_saved++;
            double x_mov = *p_saved++;
            double z1 = *p_accum++, z2 = *p_accum++, z3 = *p_accum++;

            // This is the partial derivative of the accumulated metric with respect
            // to the moving pixel intensity. It should be multiplied by the gradient
            // of moving intensity with respect to phi
            double d_metric_d_x_mov = 2 * (z1 * x_fix - z2 * x_mov + z3);

            double w_comp = this->m_Weights[k];
            d_metric_d_x_mov *= w_comp; // * n;

            double *p_affine_grad = this->m_ComputeAffine ? td.gradient.data_block() : nullptr;
            for(unsigned int d = 0; d < ImageDimension; d++)
              {
              double dMk_dPhi_d = *p_saved++;
              double d_metric_d_phi_k = dMk_dPhi_d * d_metric_d_x_mov;

              // Store the metric
              if(p_grad_metric)
                (*p_grad_metric)[d] += d_metric_d_phi_k;

              // Compute affine transform terms
              if(p_affine_grad)
                {
                *p_affine_grad++ += d_metric_d_phi_k;
                *p_affine_grad++ += d_metric_d_phi_k * i;
                for(unsigned int q = 1; q < ImageDimension; q++)
                  *p_affine_grad++ += d_metric_d_phi_k * cix[q];
                }
              }
            }
          }
        }

      p_work += m_TotalWorkingImageComponents;
      p_grad_metric++;
      }
    }

  // Accumulate this region's data in a thread-safe way
  if(this->m_ComputeAffine)
    this->m_AccumulatedData.Accumulate(td);
}

template <class TMetricTraits>
void
MultiComponentWeightedNCCImageMetric<TMetricTraits>
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
MultiComponentWeightedNCCImageMetric<TMetricTraits>
::GenerateData()
{
  // Working image is required
  itkAssertOrThrowMacro(m_WorkingImage, "Working image missing in MultiComponentWeightedNCCImageMetric");

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
  // accumulated. These are f, m, and grad_M in the unweighted case; and also w and
  // grad_W in the weighted case.

  // Same number of components is needed on the first accumulation pass
  m_FirstPassAccumComponents = 1 + 5 * m_InputComponents;

  // On the second pass, we have 3 accumulations per component for unweighted
  // and six per component for weighted
  m_SecondPassAccumComponents = m_NeedGradient ? (m_Weighted ? 6 : 3) * m_InputComponents : 0;

  // The offset in the working image where saved components start
  m_SavedComponentsOffset = std::max(m_FirstPassAccumComponents, m_SecondPassAccumComponents+1);

  // The number of saved components - those needed at the end to compute the gradient
  // without having to perform costly interpolation again
  m_FirstPassSavedComponents = m_NeedGradient
                               ? (2 + ImageDimension) * m_InputComponents +
                                 (m_Weighted ? 1 + ImageDimension : 0)
                               : 0;

  // Total size that the working image needs to be
  m_TotalWorkingImageComponents = m_SavedComponentsOffset + m_FirstPassSavedComponents;

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

    // It is very important to zero out the buffer
    auto *p = m_WorkingImage->GetBufferPointer();
    auto *p_end = m_WorkingImage->GetBufferPointer() + m_WorkingImage->GetPixelContainer()->Size();
    while(p < p_end)
      *p++ = 0.0;
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


#endif // __MultiComponentWeightedNCCImageMetric_txx

