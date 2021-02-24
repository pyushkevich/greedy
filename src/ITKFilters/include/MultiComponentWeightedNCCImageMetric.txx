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
  // The interpolator
  typedef FastLinearInterpolator<InputImageType, typename Superclass::RealType,
      ImageDimension, typename TMetricTraits::MaskImageType> FastInterpolator;

  // Get the number of input and output components
  int ncomp_in = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Is the gradient needed
  bool need_grad = this->GetComputeGradient();

  // Create an iterator specialized for going through metrics
  typedef MultiComponentMetricWorker<TMetricTraits, InputImageType> InterpType;
  InterpType iter(this, this->m_WorkingImage, outputRegionForThread);

  // Iterate over the lines
  for(; !iter.IsAtEnd(); iter.NextLine())
    {
    // Iterate over the pixels in the line
    for(; !iter.IsAtEndOfLine(); ++iter)
      {
      // Get the output pointer for this voxel
      InputComponentType *out = iter.GetOutputLine();
      typename FastInterpolator::InOut status;

      // TODO: do we need to go over points with mask of 0.5 or mask of 1.0?
      if(!iter.CheckFixedMask(0.0) ||
         (status = iter.Interpolate()) == FastInterpolator::OUTSIDE)
        {
        for(int j = 0; j < this->m_AccumulatedComponents; j++)
          *out++ = 0.0;
        continue;
        }

      // Even though there is code duplication, it is more efficient to branch based on
      // whether we sample inside or outside of the mask
      // TODO: if(status == FastInterpolator::BORDER)
      if(false)
        {
        // More complex case, requires scaling of everything by mask value
        double w = iter.GetMask();

        // Record the weight for accumulation, shared by all components
        *out++ = w;

        // Record the gradient of the weights if needed
        if(this->GetComputeGradient())
          for(unsigned int i = 0; i < ImageDimension; i++)
            *out++ = iter.GetMaskGradient()[i];

        // Iterate over the components
        for(int k = 0; k < ncomp_in; k++)
          {
          // The fixed value is just the moving component
          InputComponentType x_fix = iter.GetFixedLine()[k];

          // What we sample from the moving image is actually moving image intensity times the
          // moving image mask
          InputComponentType x_wmov = iter.GetMovingSample()[k];

          // Compute the weighted fixed and unweighted moving values
          double x_wfix = w * x_fix, x_mov = x_wmov / w;

          // Write the five components that are averaged in the cross-correlation computation
          *out++ = x_wfix;                         // w * f
          *out++ = x_wmov;                         // w * m
          *out++ = x_wfix * x_fix;                 // w * f^2
          *out++ = x_wmov * x_mov;                 // w * m^2
          *out++ = x_fix * x_wmov;                 // w * f * m

          if(this->GetComputeGradient())
            {
            // Write the corresponding five gradient components
            const InputComponentType *grad_wmov = iter.GetMovingSampleGradient(k);
            const InputComponentType *grad_w = iter.GetMaskGradient();

            // Record x_mov^2 for reuse
            double x_mov_sq = x_mov * x_mov, x_mov_x2 = 2 * x_mov;

            // Iterate over the gradient component
            for(int i = 0; i < ImageDimension; i++)
              {
              // (w * f)' = w' * f
              double w_grad_times_xfix = x_fix * grad_w[i];
              *out++ = w_grad_times_xfix;

              // (w * m)' = w' * m + w * m'
              *out++ = grad_wmov[i];

              // (w * f^2)' = w' * f^2
              *out++ = w_grad_times_xfix * x_fix;

              // (w * m^2)' = w' * m^2 + 2 w * m * m' = 2 * m * (w * m)' - w' * m^2
              *out++ = x_mov_x2 * grad_wmov[i] - x_mov_sq * grad_w[i];

              // (w * m * f)' = f * (w * m)'
              *out++ = x_fix * grad_wmov[i];
              }
            }
          }
        } // Border case
      else
        {
        // Simpler, internal case, weight is 1
        *out++ = 1;

        // Record the gradient of the weights if needed
        if(this->GetComputeGradient())
          for(unsigned int i = 0; i < ImageDimension; i++)
            *out++ = 0.0;

        // Iterate over the components
        for(int k = 0; k < ncomp_in; k++)
          {
          // The fixed value is just the moving component
          InputComponentType x_fix = iter.GetFixedLine()[k];

          // What we sample from the moving image is actually moving image intensity times the
          // moving image mask
          InputComponentType x_mov = iter.GetMovingSample()[k];

          // Write the five components that are averaged in the cross-correlation computation
          *out++ = x_fix;                         // f
          *out++ = x_mov;                         // m
          *out++ = x_fix * x_fix;                 // f^2
          *out++ = x_mov * x_mov;                 // m^2
          *out++ = x_fix * x_mov;                 // f * m

          if(this->GetComputeGradient())
            {
            // Write the corresponding five gradient components
            const InputComponentType *grad_mov = iter.GetMovingSampleGradient(k);
            double x_mov_x2 = 2 * x_mov;

            // Iterate over the gradient component
            for(int i = 0; i < ImageDimension; i++)
              {
              // (f)' = 0
              *out++ = 0.0;

              // (m)' = m'
              *out++ = grad_mov[i];

              // (f^2)' = 0
              *out++ = 0.0;

              // (m^2)' = 2 m * m'
              *out++ = x_mov_x2 * grad_mov[i];

              // (m * f)' = f * m'
              *out++ = x_fix * grad_mov[i];
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
::PostAccumulationComputeMetricAndGradient(
    const InputComponentType *ptr, const WeightVectorType &comp_weights,
    MetricPixelType *ptr_metric, MetricPixelType *ptr_comp_metrics,
    GradientPixelType *ptr_gradient)
{
  // IMPORTANT: this code uses double precision because single precision float seems
  // to mess up and lead to unstable computations
  unsigned int nc = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Loop over components
  const double eps = 1e-2;

  // Initialize metric to zero
  *ptr_metric = 0;

  // Read the accumulated weight of this pixel
  double sum_w = *ptr++;

  // If the weight is zero, we skip this pixel and all of its components
  if (sum_w == 0.0)
    return;

  // Retain and skip the weight gradient
  const InputComponentType *D_sum_w = nullptr;
  if(ptr_gradient)
    {
    D_sum_w = ptr;
    ptr += ImageDimension;
    }

  // Iterate over the components
  for(unsigned int i = 0; i < nc; i++)
    {
    // Read the accumulated terms
    double sum_wf = *ptr++;
    double sum_wm = *ptr++;
    double sum_wff = *ptr++;
    double sum_wmm = *ptr++;
    double sum_wfm = *ptr++;

    // Compute the weighted normalized correlation, it has a nice symmetrical formula;
    // However, to avoid division by zero issues, we should add epsilon to the components
    double var_f  = sum_w * sum_wff - sum_wf * sum_wf + eps;
    double var_m  = sum_w * sum_wmm - sum_wm * sum_wm + eps;
    double cov_fm = sum_w * sum_wfm - sum_wf * sum_wm;

    // Compute the signed square correlation coefficient.
    double one_over_denom = 1.0 / (var_f * var_m);
    double ncc_fm = cov_fm * cov_fm * one_over_denom;

    // Scale by the weight
    double w_comp = (cov_fm < 0) ? -comp_weights[i] : comp_weights[i];

    // We use sum_w as an additional scaling factor so that the contribution of
    // border pixels is reduced (seems like a fair way to do things)
    // MetricPixelType weighted_metric = (MetricPixelType) (w_comp * sum_w * ncc_fm);
    MetricPixelType weighted_metric = (MetricPixelType) sum_wm;

    // Compute the derivatives
    if(ptr_gradient)
      {
      for(unsigned int k = 0; k < ImageDimension; k++)
        {
        // Load the partial derivatives of the accumulated quantities w.r.t. phi_k
        double Dk_sum_wf  = *ptr++;
        double Dk_sum_wm  = *ptr++;
        double Dk_sum_wff = *ptr++;
        double Dk_sum_wmm = *ptr++;
        double Dk_sum_wfm = *ptr++;
        double Dk_sum_w   = D_sum_w[k];

        // Compute the partial derivatives of the variances
        double Dk_var_f  = Dk_sum_w * sum_wff + sum_w * Dk_sum_wff - 2. * sum_wf * Dk_sum_wf;
        double Dk_var_m  = Dk_sum_w * sum_wmm + sum_w * Dk_sum_wmm - 2. * sum_wm * Dk_sum_wm;
        double Dk_cov_fm = Dk_sum_w * sum_wfm + sum_w * Dk_sum_wfm - Dk_sum_wf * sum_wm - sum_wf * Dk_sum_wm;

        // Compute the partial derivative of ncc_fm. We are using the relation
        // h = f/g; h' = (f' - g' h) / g;
        double Dk_ncc_fm = one_over_denom * (2. * cov_fm * Dk_cov_fm -
                                             (Dk_var_f * var_m - var_f * Dk_var_m) * ncc_fm);

        // Apply the weight
        // (*ptr_gradient)[k] += w_comp * (Dk_sum_w * ncc_fm + sum_w * Dk_ncc_fm);
        (*ptr_gradient)[k] += Dk_sum_wm;

        // Very large value check
        if(fabs((*ptr_gradient)[k]) > 1000)
          {
          // printf("Large value: %f\n", (*ptr_gradient)[k]);
          }
        }
      }

    // Store the componentwise metric
    (*ptr_comp_metrics++) += weighted_metric;

    // Accumulate the metric
    *ptr_metric += weighted_metric;
    }
}

template <class TMetricTraits>
void
MultiComponentWeightedNCCImageMetric<TMetricTraits>
::GenerateData()
{
  // Call the requisite methods
  Superclass::AllocateOutputs();
  Superclass::BeforeThreadedGenerateData();

  // Sort out how many components will be involved in accumulation
  unsigned int nc = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Compute number of accumulated components
  this->m_AccumulatedComponents = (1 + 5 * nc) *
                                  (this->GetComputeGradient() ? 1 + ImageDimension : 1);

  // Main buffered region
  // TODO: make it possible to split the region into pieces to conserve memory
  auto working_region = this->GetFixedImage()->GetBufferedRegion();

  // If the working image is supplied, make sure that it has sufficient size
  if(m_WorkingImage)
    {
    // Check if the working image needs to be allocated
    if(m_WorkingImage->GetBufferedRegion() != working_region
       || m_WorkingImage->GetNumberOfComponentsPerPixel() < this->m_AccumulatedComponents)
      {
      // Configure the working image
      m_WorkingImage->CopyInformation(this->GetFixedImage());
      m_WorkingImage->SetNumberOfComponentsPerPixel(this->m_AccumulatedComponents);
      m_WorkingImage->SetRegions(working_region);
      m_WorkingImage->Allocate();
      }
    }

  // Fill the working image with quanitities that need to be accumulated
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  mt->ParallelizeImageRegion<Self::ImageDimension>(
        working_region,
        [this](const OutputImageRegionType &region)
    {
    this->PrecomputeAccumulatedComponents(region);
    }, nullptr);

#ifdef DUMP_NCC
  typename itk::ImageFileWriter<InputImageType>::Pointer pwriter = itk::ImageFileWriter<InputImageType>::New();
  pwriter->SetInput(m_WorkingImage);
  pwriter->SetFileName("nccpre.nii.gz");
  pwriter->Update();
#endif

  // It is feasible that the working image has been allocated for more components that
  // are currently used. In this case, we skip those components at the end
  int n_overalloc_comp = m_WorkingImage->GetNumberOfComponentsPerPixel() - this->m_AccumulatedComponents;

  // Currently, we have all the stuff we need to compute the metric in the working
  // image. Next, we run the fast sum computation to give us the local average of
  // intensities, products, gradients in the working image
  typename InputImageType::Pointer img_accum =
      AccumulateNeighborhoodSumsInPlace(m_WorkingImage.GetPointer(), m_Radius, 0, n_overalloc_comp);

#ifdef DUMP_NCC
  typename itk::ImageFileWriter<InputImageType>::Pointer pwriter2 = itk::ImageFileWriter<InputImageType>::New();
  pwriter2->SetInput(img_accum);
  pwriter2->SetFileName("nccaccum.nii.gz");
  pwriter2->Update();
#endif

  // At this point, the working image will hold the proper neighborhood sums (I, J, I^2, J^2, IJ, etc).
  // The last step is to use this information to compute the gradients. This is done by the threaded
  // portion of the filter.
  mt->ParallelizeImageRegion<Self::ImageDimension>(
        working_region,
        [this, nc, img_accum](const OutputImageRegionType &outputRegionForThread)
    {
    // Length of the line
    int line_len = outputRegionForThread.GetSize()[0];

    // Data to accumulate in our thread
    typename Superclass::ThreadAccumulatedData td(nc);

    // Added in Feb 2020, part of NaN masking. We now scale the patch NCC by the
    // number of non-background voxels in the mask, to prevent oversize contribution
    // of border pixels. To account for this, the weights are scaled by the patch size
    double one_over_patch_size = 1.0;
    for(unsigned int k = 0; k < ImageDimension; k++)
      one_over_patch_size /= (1 + 2.0 * this->m_Radius[k]);
    typename Superclass::WeightVectorType wgt_scaled = this->m_Weights * one_over_patch_size;

    // Where to store the accumulated metric (gets copied to td, but should have TPixel type)
    vnl_vector<InputComponentType> comp_metric(nc, 0.0);

    // Set up an iterator for the working image
    typedef itk::ImageLinearConstIteratorWithIndex<InputImageType> InputIteratorTypeBase;
    typedef IteratorExtender<InputIteratorTypeBase> InputIteratorType;
    InputIteratorType it(img_accum, outputRegionForThread);

    // The gradient of the metric with respect to affine coeffs, if defined
    double *p_affine_grad = this->m_ComputeAffine && this->m_ComputeGradient
                            ? td.gradient.data_block() : nullptr;

    // Loop over the lines
    for (; !it.IsAtEnd(); it.NextLine())
      {
      // Get the pointer to the input line
      long offset_in_pixels = it.GetPosition() - img_accum->GetBufferPointer();

      // Pointer to the input pixel data for this line
      const InputComponentType *p_input =
          img_accum->GetBufferPointer() + this->m_AccumulatedComponents * offset_in_pixels;

      // Pointer to the metric data for this line
      MetricPixelType *p_metric =
          this->GetMetricOutput()->GetBufferPointer() + offset_in_pixels;

      // The gradient output is optional
      GradientPixelType *p_grad_metric =
          (this->m_ComputeGradient)
          ? this->GetDeformationGradientOutput()->GetBufferPointer() + offset_in_pixels
          : nullptr;

      // Get the fixed mask like
      typename MaskImageType::PixelType *fixed_mask_line =
          this->GetFixedMaskImage()
          ? this->GetFixedMaskImage()->GetBufferPointer() + offset_in_pixels
          : nullptr;

      // Get the current index
      auto line_idx = it.GetIndex();

      // Loop over the pixels in the line
      for(int i = 0; i < line_len; ++i)
        {
        // Clear the metric and the gradient
        *p_metric = itk::NumericTraits<MetricPixelType>::ZeroValue();
        *p_grad_metric = itk::NumericTraits<GradientPixelType>::ZeroValue();

        // Check the mask (TODO: shouldn't we be checking 0.5 here?)
        if(!fixed_mask_line || fixed_mask_line[i] > 0.0)
          {
          // Compute metric for this pixel
          PostAccumulationComputeMetricAndGradient(
                p_input, wgt_scaled, p_metric,
                comp_metric.data_block(), p_grad_metric);

          // Accumulate the total metric and mask (here the mask is just the fixed mask)
          td.metric += *p_metric;
          td.mask += 1.0;

          // Compute the partial derivatives of the affine transform, if required
          if(this->m_ComputeAffine && this->m_ComputeGradient)
            {
            // Get the coordinates of the point
            auto *p = p_affine_grad;
            for(unsigned int j = 0; j < ImageDimension; j++)
              {
              // First iteration is over the components of the gradient
              double grad_phi_j = (*p_grad_metric)[j];

              // The x coordinate is treated differently
              *p++ = (line_idx[0] + i) * grad_phi_j;

              // The remaining coordinates (y,z)
              for(unsigned int k = 1; k < ImageDimension; k++)
                *p++ = line_idx[k] * grad_phi_j;

              // The b derivatives
              *p++ = grad_phi_j;
              }
            }
          }

        // Increment the pointers
        p_input += this->m_AccumulatedComponents;
        p_metric++;
        if(this->m_ComputeGradient)
          p_grad_metric++;
        }
      }

    // Typecast the per-component metrics
    for(unsigned int a = 0; a < nc; a++)
      td.comp_metric[a] = comp_metric[a];

    // Accumulate this region's data in a thread-safe way
    this->m_AccumulatedData.Accumulate(td);

    }, nullptr);

// #ifdef DUMP_NCC
  /*
  typename itk::ImageFileWriter<GradientImageType>::Pointer pwriter3 = itk::ImageFileWriter<GradientImageType>::New();
  pwriter3->SetInput(this->GetDeformationGradientOutput());
  pwriter3->SetFileName("nccgrad.mhd");
  pwriter3->Update();
  */
// #endif

  Superclass::AfterThreadedGenerateData();
}


#endif // __MultiComponentWeightedNCCImageMetric_txx

