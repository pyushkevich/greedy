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
#include "MultiImageRegistrationHelper.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkContinuousIndex.h"
#include "vnl/vnl_math.h"
#include "lddmm_data.h"
#include "MultiImageOpticalFlowImageFilter.h"
#include "MultiComponentNCCImageMetric.h"
#include "MultiComponentApproximateNCCImageMetric.h"
#include "MultiComponentMutualInfoImageMetric.h"
#include "MahalanobisDistanceToTargetWarpMetric.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
#include "OneDimensionalInPlaceAccumulateFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkImageFileWriter.h"
#include "GreedyException.h"
#include "WarpFunctors.h"

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetDefaultPyramidFactors(int n_levels)
{
  for(int i = n_levels-1; i>=0; --i)
    m_PyramidFactors.push_back(1 << i);
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetPyramidFactors(const PyramidFactorsType &factors)
{
  m_PyramidFactors = factors;
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight)
{
  // Collect the weights
  for(unsigned i = 0; i < fixed->GetNumberOfComponentsPerPixel(); i++)
    m_Weights.push_back(weight);

  // Store the images
  m_Fixed.push_back(fixed);
  m_Moving.push_back(moving);
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetJitterSigma(double sigma)
{
  m_JitterSigma = sigma;
}

template<class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetGradientMaskTrimRadius(const std::vector<int> &radius)
{
  if(radius.size() != VDim)
    throw GreedyException("Gradien mask trim radius parameter has incorrect dimension");

  m_GradientMaskTrimRadius = radius;
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::PlaceIntoComposite(FloatImageType *source, MultiComponentImageType *target, int offset)
{
  // We do this using a loop - no threading
  TFloat *src_ptr = source->GetPixelContainer()->GetBufferPointer();
  TFloat *trg_ptr = target->GetPixelContainer()->GetBufferPointer() + offset;

  int trg_comp = target->GetNumberOfComponentsPerPixel();

  int n_voxels = source->GetPixelContainer()->Size();
  TFloat *trg_end = trg_ptr + n_voxels * target->GetNumberOfComponentsPerPixel();

  while(trg_ptr < trg_end)
    {
    *trg_ptr = *src_ptr++;
    trg_ptr += trg_comp;
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::PlaceIntoComposite(VectorImageType *source, MultiComponentImageType *target, int offset)
{
  // We do this using a loop - no threading
  VectorType *src_ptr = source->GetPixelContainer()->GetBufferPointer();
  TFloat *trg_ptr = target->GetPixelContainer()->GetBufferPointer() + offset;

  int trg_skip = target->GetNumberOfComponentsPerPixel() - VDim;

  int n_voxels = source->GetPixelContainer()->Size();
  TFloat *trg_end = trg_ptr + n_voxels * target->GetNumberOfComponentsPerPixel();

  while(trg_ptr < trg_end)
    {
    const VectorType &vsrc = *src_ptr++;
    for(int k = 0; k < VDim; k++)
      *trg_ptr++ = vsrc[k];
    trg_ptr += trg_skip;
    }
}

#include <vnl/vnl_random.h>

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::DilateCompositeGradientMasksForNCC(SizeType radius)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  for(int level = 0; level < m_PyramidFactors.size(); level++)
    {
    if(m_GradientMaskComposite[level])
      {
      // Threshold the mask itself
      LDDMMType::img_threshold_in_place(m_GradientMaskComposite[level], 0.5, 1e100, 0.5, 0);

      // Make a copy of the mask
      typename FloatImageType::Pointer mask_copy = LDDMMType::new_img(m_GradientMaskComposite[level]);
      LDDMMType::img_copy(m_GradientMaskComposite[level], mask_copy);

      // Run the accumulation filter on the mask
      typename FloatImageType::Pointer mask_accum =
          AccumulateNeighborhoodSumsInPlace(mask_copy.GetPointer(), radius);

      // Threshold the mask copy
      LDDMMType::img_threshold_in_place(mask_accum, 0.25, 1e100, 0.5, 0);

      // Add the two images - the result has 1 for the initial mask, 0.5 for the 'outer' mask
      LDDMMType::img_add_in_place(m_GradientMaskComposite[level], mask_accum);
      }
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::BuildCompositeImages(double noise_sigma_relative)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Offsets into the composite images
  int off_fixed = 0, off_moving = 0;

  // Set up the composite images
  m_FixedComposite.resize(m_PyramidFactors.size());
  m_MovingComposite.resize(m_PyramidFactors.size());

  // The fixed mask is binarized
  if(m_FixedMaskImage)
    LDDMMType::img_threshold_in_place(m_FixedMaskImage, 0.5, 1e100, 0.0, 1.0);

  // Repeat for each of the input images
  for(int j = 0; j < m_Fixed.size(); j++)
    {
    // Repeat for each component
    for(unsigned k = 0; k < m_Fixed[j]->GetNumberOfComponentsPerPixel(); k++)
      {
      // Extract the k-th image component from fixed and moving images
      typedef itk::VectorIndexSelectionCastImageFilter<MultiComponentImageType, FloatImageType> ExtractType;
      typename ExtractType::Pointer fltExtractFixed, fltExtractMoving;

      fltExtractFixed = ExtractType::New();
      fltExtractFixed->SetInput(m_Fixed[j]);
      fltExtractFixed->SetIndex(k);
      fltExtractFixed->Update();

      fltExtractMoving = ExtractType::New();
      fltExtractMoving->SetInput(m_Moving[j]);
      fltExtractMoving->SetIndex(k);
      fltExtractMoving->Update();

      // Deal with additive noise
      double noise_sigma_fixed = 0.0, noise_sigma_moving = 0.0;

      // Do the fixed and moving images have NaNs?
      bool nans_fixed, nans_moving;

      // If the fixed mask is present, we use it to set nans in the fixed image
      if(m_FixedMaskImage)
        {
        LDDMMType::img_reconstitute_nans_in_place(fltExtractFixed->GetOutput(), m_FixedMaskImage);
        }

      if(noise_sigma_relative > 0.0)
        {
        // Figure out the quartiles of the fixed image
        typedef MutualInformationPreprocessingFilter<FloatImageType, FloatImageType> QuantileFilter;
        typename QuantileFilter::Pointer fltQuantileFixed = QuantileFilter::New();
        fltQuantileFixed->SetLowerQuantile(0.01);
        fltQuantileFixed->SetUpperQuantile(0.99);
        fltQuantileFixed->SetInput(fltExtractFixed->GetOutput());
        fltQuantileFixed->Update();
        double range_fixed = fltQuantileFixed->GetUpperQuantileValue(0) - fltQuantileFixed->GetLowerQuantileValue(0);
        noise_sigma_fixed = noise_sigma_relative * range_fixed;

        // Figure out the quartiles of the moving image
        typename QuantileFilter::Pointer fltQuantileMoving = QuantileFilter::New();
        fltQuantileMoving->SetLowerQuantile(0.01);
        fltQuantileMoving->SetUpperQuantile(0.99);
        fltQuantileMoving->SetInput(fltExtractMoving->GetOutput());
        fltQuantileMoving->Update();
        double range_moving = fltQuantileMoving->GetUpperQuantileValue(0) - fltQuantileMoving->GetLowerQuantileValue(0);
        noise_sigma_moving = noise_sigma_relative * range_moving;

        // Report noise levels
        printf("Noise on image %d component %d: fixed = %g, moving = %g\n", j, k, noise_sigma_fixed, noise_sigma_moving);

        // Record the number of NaNs
        nans_fixed = fltQuantileFixed->GetNumberOfNaNs(0);
        nans_moving = fltQuantileMoving->GetNumberOfNaNs(0);
        }
      else
        {
        nans_fixed = isnan(LDDMMType::img_voxel_sum(fltExtractFixed->GetOutput()));
        nans_moving = isnan(LDDMMType::img_voxel_sum(fltExtractMoving->GetOutput()));
        }

      // Report number of NaNs in fixed and moving images
      if(j==0 && k==0)
        {
        printf("Number of NaNs: fixed: %d, moving %d\n", nans_fixed, nans_moving);
        }
      
      // Split the extracted images into a NaN mask and a non-NaN component
      FloatImagePointer nanMaskFixed, nanMaskMoving;
      if(nans_fixed)
        {
        nanMaskFixed = LDDMMType::new_img(fltExtractFixed->GetOutput());
        LDDMMType::img_filter_nans_in_place(fltExtractFixed->GetOutput(), nanMaskFixed);
        }
      
      if(nans_moving)
        {
        nanMaskMoving = LDDMMType::new_img(fltExtractMoving->GetOutput());
        LDDMMType::img_filter_nans_in_place(fltExtractMoving->GetOutput(), nanMaskMoving);
        }

      // Compute the pyramid for this component
      for(int i = 0; i < m_PyramidFactors.size(); i++)
        {
        // Downsample the image to the right pyramid level
        typename FloatImageType::Pointer lFixed, lMoving;
        if (m_PyramidFactors[i] == 1)
          {
          lFixed = fltExtractFixed->GetOutput();
          if(nans_fixed)
            LDDMMType::img_reconstitute_nans_in_place(lFixed, nanMaskFixed);

          lMoving = fltExtractMoving->GetOutput();
          if(nans_moving)
            LDDMMType::img_reconstitute_nans_in_place(lMoving, nanMaskMoving);
          }
        else
          {
          // Downsample the images
          lFixed = FloatImageType::New();
          lMoving = FloatImageType::New();
          LDDMMType::img_downsample(fltExtractFixed->GetOutput(), lFixed, m_PyramidFactors[i]);
          LDDMMType::img_downsample(fltExtractMoving->GetOutput(), lMoving, m_PyramidFactors[i]);

          // Downsample the nan-masks
          if(nans_fixed)
            {
            FloatImagePointer mask_ds = FloatImageType::New();
            LDDMMType::img_downsample(nanMaskFixed, mask_ds, m_PyramidFactors[i]);
            LDDMMType::img_threshold_in_place(mask_ds, 0.5, 100.0, 1, 0);
            LDDMMType::img_reconstitute_nans_in_place(lFixed, mask_ds);
            }

          if(nans_moving)
            {
            FloatImagePointer mask_ds = FloatImageType::New();
            LDDMMType::img_downsample(nanMaskMoving, mask_ds, m_PyramidFactors[i]);
            LDDMMType::img_threshold_in_place(mask_ds, 0.5, 100.0, 1, 0);
            LDDMMType::img_reconstitute_nans_in_place(lMoving, mask_ds);
            }

          // For the Mahalanobis metric, the fixed image needs to be scaled by the factor of the
          // pyramid level because it describes voxel coordinates
          if(m_ScaleFixedImageWithVoxelSize)
            LDDMMType::img_scale_in_place(lFixed, 1.0 / m_PyramidFactors[i]);
          }

        // Add some noise to the images
        if(noise_sigma_relative > 0.0)
          {
          vnl_random randy(12345);
          for(long i = 0; i < lFixed->GetPixelContainer()->Size(); i++)
            lFixed->GetBufferPointer()[i] += randy.normal() * noise_sigma_fixed;
          for(long i = 0; i < lMoving->GetPixelContainer()->Size(); i++)
            lMoving->GetBufferPointer()[i] += randy.normal() * noise_sigma_moving;
          }

        // Compute the gradient of the moving image
        //typename VectorImageType::Pointer gradMoving = LDDMMType::new_vimg(lMoving);
        //LDDMMType::image_gradient(lMoving, gradMoving);

        // Allocate the composite images if they have not been allocated
        if(j == 0 && k == 0)
          {
          m_FixedComposite[i] = MultiComponentImageType::New();
          m_FixedComposite[i]->CopyInformation(lFixed);
          m_FixedComposite[i]->SetNumberOfComponentsPerPixel(m_Weights.size());
          m_FixedComposite[i]->SetRegions(lFixed->GetBufferedRegion());
          m_FixedComposite[i]->Allocate();

          m_MovingComposite[i] = MultiComponentImageType::New();
          m_MovingComposite[i]->CopyInformation(lMoving);
          m_MovingComposite[i]->SetNumberOfComponentsPerPixel(m_Weights.size());
          m_MovingComposite[i]->SetRegions(lMoving->GetBufferedRegion());
          m_MovingComposite[i]->Allocate();
          }

        // Pack the data into the fixed and moving composite images
        this->PlaceIntoComposite(lFixed, m_FixedComposite[i], off_fixed);
        this->PlaceIntoComposite(lMoving, m_MovingComposite[i], off_moving);
        }

      // Update the offsets
      off_fixed++;
      off_moving++;
      }
    }

  // Set up the mask pyramid
  m_GradientMaskComposite.resize(m_PyramidFactors.size(), NULL);
  if(m_GradientMaskImage)
    {
    for(int i = 0; i < m_PyramidFactors.size(); i++)
      {
      // Downsample the image to the right pyramid level
      if (m_PyramidFactors[i] == 1)
        {
        m_GradientMaskComposite[i] = m_GradientMaskImage;
        }
      else
        {
        m_GradientMaskComposite[i] = FloatImageType::New();

        // Downsampling the mask involves smoothing, so the mask will no longer be binary
        LDDMMType::img_downsample(m_GradientMaskImage, m_GradientMaskComposite[i], m_PyramidFactors[i]);
        LDDMMType::img_threshold_in_place(m_GradientMaskComposite[i], 0.5, 1e100, 1.0, 0.0);
        }      
      }
    }
  else if(m_GradientMaskTrimRadius.size() > 0)
    {
    // User wants auto-generated box masks. Create them for every pyramid level
    for(int i = 0; i < m_PyramidFactors.size(); i++)
      {
      // Allocate the image
      m_GradientMaskComposite[i] = LDDMMType::new_img(m_FixedComposite[i]);

      // Fill out the image
      itk::Size<VDim> sz = m_GradientMaskComposite[i]->GetBufferedRegion().GetSize();

      typedef itk::ImageRegionIteratorWithIndex<FloatImageType> IterType;
      for(IterType it(m_GradientMaskComposite[i], m_GradientMaskComposite[i]->GetBufferedRegion());
          !it.IsAtEnd(); ++it)
        {
        TFloat mask_val = 1.0;
        for(unsigned int d = 0; d < VDim; d++)
          {
          if(it.GetIndex()[d] < m_GradientMaskTrimRadius[d]
             || sz[d] - it.GetIndex()[d] <= m_GradientMaskTrimRadius[d])
            {
            mask_val = 0.0;
            break;
            }
          }

        it.Set(mask_val);
        }
      }
    }

  // Set up the moving mask pyramid
  m_MovingMaskComposite.resize(m_PyramidFactors.size(), NULL);
  if(m_MovingMaskImage)
    {
    for(int i = 0; i < m_PyramidFactors.size(); i++)
      {
      // Downsample the image to the right pyramid level
      if (m_PyramidFactors[i] == 1)
        {
        m_MovingMaskComposite[i] = m_MovingMaskImage;
        }
      else
        {
        m_MovingMaskComposite[i] = FloatImageType::New();

        // Downsampling the mask involves smoothing, so the mask will no longer be binary
        LDDMMType::img_downsample(m_MovingMaskImage, m_MovingMaskComposite[i], m_PyramidFactors[i]);

        // We might not need the moving mask to be binary, we can leave it be floating point
        // but for now we binarize it
        LDDMMType::img_threshold_in_place(m_MovingMaskComposite[i], 0.5, 1e100, 1.0, 0.0);
        }
      }
    }

  // Set up the jitter images
  m_JitterComposite.resize(m_PyramidFactors.size(), NULL);
  if(m_JitterSigma > 0)
    {
    for(int i = 0; i < m_PyramidFactors.size(); i++)
      {
      // Get the reference space
      ImageBaseType *base = this->GetReferenceSpace(i);
      VectorImagePointer iJitter = VectorImageType::New();
      iJitter->CopyInformation(base);
      iJitter->SetRegions(base->GetBufferedRegion());
      iJitter->Allocate();

      vnl_random randy(12345);
      typedef itk::ImageRegionIterator<VectorImageType> IterType;
      for(IterType iter(iJitter, iJitter->GetBufferedRegion()); !iter.IsAtEnd(); ++iter)
        {
        for(int k = 0; k < VDim; k++)
          {
          iter.Value()[k] = randy.normal() * m_JitterSigma;
          }
        }

      m_JitterComposite[i] = iJitter;
      }
    }
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::ImageBaseType *
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetMovingReferenceSpace(int level)
{
  return m_MovingComposite[level];
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::ImageBaseType *
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetReferenceSpace(int level)
{
  return m_FixedComposite[level];
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::Vec
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetSmoothingSigmasInPhysicalUnits(int level, double sigma, bool in_physical_units)
{
  Vec sigmas;
  if(in_physical_units)
    {
    sigmas.Fill(sigma * m_PyramidFactors[level]);
    }
  else
    {
    for(int k = 0; k < VDim; k++)
      sigmas[k] = this->GetReferenceSpace(level)->GetSpacing()[k] * sigma;
    }
  return sigmas;
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeOpticalFlowField(int level,
                          VectorImageType *def,
                          FloatImageType *out_metric_image,
                          MultiComponentMetricReport &out_metric_report,
                          VectorImageType *out_gradient,
                          double result_scaling)
{
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiImageOpticalFlowImageFilter<TraitsType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for (unsigned i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i] * result_scaling;

  // TODO: this needs to be controlled by parameters, etc.
  // 'false' represents compatibility with previous greedy versions
  filter->SetUseDemonsGradientForm(false);
  filter->SetDemonsSigma(0.01);

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImage(m_MovingComposite[level]);
  filter->SetDeformationField(def);
  filter->SetWeights(wscaled);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric_image);
  filter->GetDeformationGradientOutput()->Graft(out_gradient);
  filter->Update();

  // Process the results
  out_metric_report.ComponentMetrics = filter->GetAllMetricValues();
  out_metric_report.TotalMetric = filter->GetMetricValue();
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeHistogramsIfNeeded(int level)
{
  typedef MutualInformationPreprocessingFilter<MultiComponentImageType, BinnedImageType> BinnerType;
  if(m_FixedBinnedImage.IsNull()
     || m_FixedBinnedImage->GetBufferedRegion() != m_FixedComposite[level]->GetBufferedRegion())
    {
    typename BinnerType::Pointer fixed_binner = BinnerType::New();
    fixed_binner->SetInput(m_FixedComposite[level]);
    fixed_binner->SetBins(128);
    fixed_binner->SetLowerQuantile(0.01);
    fixed_binner->SetUpperQuantile(0.99);
    fixed_binner->SetStartAtBinOne(true);
    fixed_binner->Update();
    m_FixedBinnedImage = fixed_binner->GetOutput();

    typename BinnerType::Pointer moving_binner = BinnerType::New();
    moving_binner = BinnerType::New();
    moving_binner->SetInput(m_MovingComposite[level]);
    moving_binner->SetBins(128);
    moving_binner->SetLowerQuantile(0.01);
    moving_binner->SetUpperQuantile(0.99);
    moving_binner->SetStartAtBinOne(true);
    moving_binner->Update();
    m_MovingBinnedImage = moving_binner->GetOutput();
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeMIFlowField(int level,
                     bool normalized_mutual_information,
                     VectorImageType *def,
                     FloatImageType *out_metric_image,
                     MultiComponentMetricReport &out_metric_report,
                     VectorImageType *out_gradient,
                     double result_scaling)
{
  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for (unsigned i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i] * result_scaling;

  // Set up the mutual information metric
  typedef DefaultMultiComponentMutualInfoImageMetricTraits<TFloat, unsigned char, VDim> TraitsType;
  typedef MultiComponentMutualInfoImageMetric<TraitsType> MetricType;

  typedef itk::VectorImage<unsigned char, VDim> BinnedImageType;
  typedef MutualInformationPreprocessingFilter<MultiComponentImageType, BinnedImageType> BinnerType;

  // Initialize the histograms
  this->ComputeHistogramsIfNeeded(level);

  typename MetricType::Pointer metric = MetricType::New();

  metric->SetComputeNormalizedMutualInformation(normalized_mutual_information);
  metric->SetFixedImage(m_FixedBinnedImage);
  metric->SetMovingImage(m_MovingBinnedImage);
  metric->SetDeformationField(def);
  metric->SetWeights(wscaled);
  metric->SetComputeGradient(true);
  metric->GetMetricOutput()->Graft(out_metric_image);
  metric->GetDeformationGradientOutput()->Graft(out_gradient);
  metric->GetMetricOutput()->Graft(out_metric_image);
  metric->SetBins(128);
  metric->Update();

  // Process the results
  out_metric_report.ComponentMetrics = metric->GetAllMetricValues();
  out_metric_report.TotalMetric = metric->GetMetricValue();
}

// #undef DUMP_NCC
#define DUMP_NCC 1


template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::SizeType
MultiImageOpticalFlowHelper<TFloat, VDim>
::AdjustNCCRadius(int level, const SizeType &radius, bool report_on_adjust)
{
  SizeType radius_fix = radius;
  for(int d = 0; d < VDim; d++)
    {
    int sz_d = (int) m_FixedComposite[level]->GetBufferedRegion().GetSize()[d];
    if(radius_fix[d] * 2 + 1 >= sz_d)
      radius_fix[d] = (sz_d - 1) / 2;
    }

  if(report_on_adjust && radius != radius_fix)
    {
    std::cout << "  *** NCC radius adjusted to " << radius_fix
              << " because image too small at level " << level
              << " (" << m_FixedComposite[level]->GetBufferedRegion().GetSize() << ")" << std::endl;
    }

  return radius_fix;
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeNCCMetricImage(int level,
                        VectorImageType *def,
                        const SizeType &radius,
                        FloatImageType *out_metric_image,
                        MultiComponentMetricReport &out_metric_report,
                        VectorImageType *out_gradient,
                        double result_scaling)
{
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiComponentNCCImageMetric<TraitsType> FilterType;
  // typedef MultiComponentApproximateNCCImageMetric<TraitsType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for (unsigned i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i] * result_scaling;

  // Allocate a working image
  if(m_NCCWorkingImage.IsNull())
    m_NCCWorkingImage = MultiComponentImageType::New();

  // Is this the first time that this function is being called with this image?
  bool first_run =
      m_NCCWorkingImage->GetBufferedRegion() != m_FixedComposite[level]->GetBufferedRegion();

  // Check the radius against the size of the image
  SizeType radius_fix = AdjustNCCRadius(level, radius, first_run);

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImage(m_MovingComposite[level]);
  filter->SetDeformationField(def);
  filter->SetWeights(wscaled);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric_image);
  filter->GetDeformationGradientOutput()->Graft(out_gradient);
  filter->SetRadius(radius_fix);
  filter->SetWorkingImage(m_NCCWorkingImage);
  filter->SetReuseWorkingImageFixedComponents(!first_run);
  filter->SetFixedMaskImage(m_GradientMaskComposite[level]);
  filter->SetMovingMaskImage(m_MovingMaskComposite[level]);


  // TODO: support moving masks...
  // filter->SetMovingMaskImage(m_MovingMaskComposite[level]);
  filter->Update();

  // Get the vector of the normalized metrics
  out_metric_report.ComponentMetrics = filter->GetAllMetricValues();
  out_metric_report.TotalMetric = filter->GetMetricValue();
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeMahalanobisMetricImage(int level, VectorImageType *def, 
  FloatImageType *out_metric_image, MultiComponentMetricReport &out_metric_report,
  VectorImageType *out_gradient)
{
  typedef DefaultMahalanobisDistanceToTargetMetricTraits<TFloat, VDim> TraitsType;
  typedef MahalanobisDistanceToTargetWarpMetric<TraitsType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();

  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImage(m_MovingComposite[level]);
  filter->SetDeformationField(def);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric_image);
  filter->GetDeformationGradientOutput()->Graft(out_gradient);
  filter->SetFixedMaskImage(m_GradientMaskComposite[level]);
  filter->SetMovingMaskImage(m_MovingMaskComposite[level]);

  filter->Update();

  out_metric_report.ComponentMetrics = filter->GetAllMetricValues();
  out_metric_report.TotalMetric = filter->GetMetricValue();
}


template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineMSDMatchAndGradient(int level,
    LinearTransformType *tran,
    FloatImageType *wrkMetric,
    FloatImageType *wrkMask,
    VectorImageType *wrkGradMetric,
    VectorImageType *wrkGradMask,
    VectorImageType *wrkPhi,
    MultiComponentMetricReport &out_metric,
    LinearTransformType *grad)
{
  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for (unsigned i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i];

  // Set up the optical flow computation
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiImageOpticalFlowImageFilter<TraitsType> MetricType;
  typename MetricType::Pointer metric = MetricType::New();

  metric->SetFixedImage(m_FixedComposite[level]);
  metric->SetMovingImage(m_MovingComposite[level]);
  metric->SetWeights(wscaled);
  metric->SetAffineTransform(tran);
  metric->SetComputeMovingDomainMask(true);
  metric->GetMetricOutput()->Graft(wrkMetric);
  metric->SetComputeGradient(grad != NULL);
  metric->SetFixedMaskImage(m_GradientMaskComposite[level]);
  metric->SetMovingMaskImage(m_MovingMaskComposite[level]);
  metric->SetJitterImage(m_JitterComposite[level]);
  metric->Update();

  // TODO: erase this
  /*
  std::cout << "SAVING METRIC, TRAN = " << tran->GetMatrix() << std::endl;
  static int iter = 0;
  std::ostringstream oss; oss << "metric_" << iter << ".nii.gz";
  LDDMMData<TFloat, VDim>::img_write(wrkMetric, oss.str().c_str());
  std::stringstream oss2; oss2 << "metric_mask_" << iter << ".nii.gz";
  LDDMMData<TFloat, VDim>::img_write(wrkMask, oss2.str().c_str());
  ++iter;
  */

  // Process the results
  if(grad)
    {
    grad->SetMatrix(metric->GetAffineTransformGradient()->GetMatrix());
    grad->SetOffset(metric->GetAffineTransformGradient()->GetOffset());
    }

  out_metric.TotalMetric = metric->GetMetricValue();
  out_metric.ComponentMetrics = metric->GetAllMetricValues();
}

#include "itkRescaleIntensityImageFilter.h"

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineMIMatchAndGradient(int level,
                                  bool normalized_mutual_info,
                                  LinearTransformType *tran,
                                  FloatImageType *wrkMetric,
                                  FloatImageType *wrkMask,
                                  VectorImageType *wrkGradMetric,
                                  VectorImageType *wrkGradMask,
                                  VectorImageType *wrkPhi,
                                  MultiComponentMetricReport &out_metric,
                                  LinearTransformType *grad)
{
  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for (unsigned i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i];

  // Set up the mutual information metric
  typedef DefaultMultiComponentMutualInfoImageMetricTraits<TFloat, unsigned char, VDim> TraitsType;
  typedef MultiComponentMutualInfoImageMetric<TraitsType> MetricType;

  typedef itk::VectorImage<unsigned char, VDim> BinnedImageType;
  typedef MutualInformationPreprocessingFilter<MultiComponentImageType, BinnedImageType> BinnerType;

  // Initialize the histograms
  this->ComputeHistogramsIfNeeded(level);

  typename MetricType::Pointer metric = MetricType::New();

  metric->SetComputeNormalizedMutualInformation(normalized_mutual_info);
  metric->SetFixedImage(m_FixedBinnedImage);
  metric->SetMovingImage(m_MovingBinnedImage);
  metric->SetWeights(wscaled);
  metric->SetAffineTransform(tran);
  metric->SetComputeMovingDomainMask(true);
  metric->GetMetricOutput()->Graft(wrkMetric);
  metric->SetComputeGradient(grad != NULL);
  metric->SetFixedMaskImage(m_GradientMaskComposite[level]);
  metric->SetMovingMaskImage(m_MovingMaskComposite[level]);
  metric->SetBins(128);
  metric->SetJitterImage(m_JitterComposite[level]);
  metric->Update();

  // Process the results
  if(grad)
    {
    grad->SetMatrix(metric->GetAffineTransformGradient()->GetMatrix());
    grad->SetOffset(metric->GetAffineTransformGradient()->GetOffset());
    }

  out_metric.TotalMetric = metric->GetMetricValue();
  out_metric.ComponentMetrics = metric->GetAllMetricValues();
}



// TODO: there is a lot of code duplication here!
template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineNCCMatchAndGradient(int level,
                                   LinearTransformType *tran,
                                   const SizeType &radius,
                                   FloatImageType *wrkMetric,
                                   FloatImageType *wrkMask,
                                   VectorImageType *wrkGradMetric,
                                   VectorImageType *wrkGradMask,
                                   VectorImageType *wrkPhi,
                                   MultiComponentMetricReport &out_metric,
                                   LinearTransformType *grad)
{
  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for (unsigned i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i];

  // Allocate a working image
  if(m_NCCWorkingImage.IsNull())
    m_NCCWorkingImage = MultiComponentImageType::New();

  // Set up the optical flow computation
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiComponentNCCImageMetric<TraitsType> MetricType;
  typename MetricType::Pointer metric = MetricType::New();

  // Is this the first time that this function is being called with this image?
  bool first_run =
      m_NCCWorkingImage->GetBufferedRegion() != m_FixedComposite[level]->GetBufferedRegion();

  // Check the radius against the size of the image
  SizeType radius_fix = AdjustNCCRadius(level, radius, first_run);

  metric->SetFixedImage(m_FixedComposite[level]);
  metric->SetMovingImage(m_MovingComposite[level]);
  metric->SetWeights(wscaled);
  metric->SetAffineTransform(tran);
  metric->SetComputeMovingDomainMask(false);
  metric->GetMetricOutput()->Graft(wrkMetric);
  metric->SetComputeGradient(grad != NULL);
  metric->SetRadius(radius_fix);
  metric->SetWorkingImage(m_NCCWorkingImage);
  metric->SetReuseWorkingImageFixedComponents(!first_run);
  metric->SetFixedMaskImage(m_GradientMaskComposite[level]);
  metric->SetMovingMaskImage(m_MovingMaskComposite[level]);
  metric->SetJitterImage(m_JitterComposite[level]);
  metric->Update();

  // Process the results
  if(grad)
    {
    grad->SetMatrix(metric->GetAffineTransformGradient()->GetMatrix());
    grad->SetOffset(metric->GetAffineTransformGradient()->GetOffset());
    }

  out_metric.TotalMetric = metric->GetMetricValue();
  out_metric.ComponentMetrics = metric->GetAllMetricValues();

  // TODO: delete this sht
  /*
  if(grad)
    {


    // Generate a phi from the affine transform
    typedef LDDMMData<TFloat, VDim> LDDMMType;
    VectorImagePointer phi = LDDMMType::new_vimg(m_FixedComposite[level]);
    for(itk::ImageRegionIteratorWithIndex<VectorImageType>
        it(phi, phi->GetBufferedRegion()); !it.IsAtEnd(); ++it)
      {
      typename VectorImageType::PixelType v;
      for(int d = 0; d < VDim; d++)
        {
        v[d] = tran->GetOffset()[d] - it.GetIndex()[d];
        for(int j = 0; j < VDim; j++)
          v[d] += tran->GetMatrix()(d,j) * it.GetIndex()[j];
        }
      it.Set(v);
      }

    // LDDMMType::vimg_write(phi, "/tmp/wtf.nii.gz");

    MultiComponentMetricReport dummy;
    FloatImagePointer tmp_met = LDDMMType::new_img(m_FixedComposite[level]);
    VectorImagePointer grad_phi = LDDMMType::new_vimg(m_FixedComposite[level]);
    tmp_met->FillBuffer(0.0);


    typename MetricType::Pointer filter2 = MetricType::New();

    // Run the filter
    MultiComponentImagePointer work2 = MultiComponentImageType::New();

    filter2->SetFixedImage(m_FixedComposite[level]);
    filter2->SetMovingImage(m_MovingComposite[level]);
    filter2->SetDeformationField(phi);
    filter2->SetWeights(wscaled);
    filter2->SetComputeGradient(true);
    filter2->GetMetricOutput()->Graft(tmp_met);
    filter2->GetDeformationGradientOutput()->Graft(grad_phi);
    filter2->SetRadius(radius_fix);
    filter2->SetWorkingImage(work2);
    filter2->SetReuseWorkingImageFixedComponents(false);
    filter2->SetFixedMaskImage(m_GradientMaskComposite[level]);

    // TODO: support moving masks...
    // filter->SetMovingMaskImage(m_MovingMaskComposite[level]);
    filter2->Update();
    dummy.TotalMetric = filter2->GetMetricValue();

    typename LinearTransformType::Pointer tran2 = LinearTransformType::New();
    typename LinearTransformType::MatrixType A2; A2.Fill(0.0);
    typename LinearTransformType::OffsetType b2; b2.Fill(0.0);

    double mtx = 0.0, msk = 0.0;
    for(itk::ImageRegionIteratorWithIndex<VectorImageType>
        it(grad_phi, grad_phi->GetBufferedRegion()); !it.IsAtEnd(); ++it)
      {
      if(!m_GradientMaskComposite[level] || m_GradientMaskComposite[level]->GetPixel(it.GetIndex()) > 0.5)
        {
        mtx += tmp_met->GetPixel(it.GetIndex());

        msk += 1.0;
        for(int d = 0; d < VDim; d++)
          {
          b2[d] += it.Value()[d];
          for(int j = 0; j < VDim; j++)
            A2(d,j) += it.Value()[d] * it.GetIndex()[j];
          }
        }
      }

    for(int d = 0; d < VDim; d++)
      {
      b2[d] /= msk;
      for(int j = 0; j < VDim; j++)
        A2(d,j) /= msk;
      }

    //grad->SetMatrix(A2);
    //grad->SetOffset(b2);
    //out_metric.TotalMetric = mtx / msk;
    //out_metric.ComponentMetrics = filter2->GetAllMetricValues();

    }
    */
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::AffineToField(LinearTransformType *tran, VectorImageType *def)
{
  // TODO: convert this to a filter
  typedef itk::ImageLinearIteratorWithIndex<VectorImageType> IterBase;
  typedef IteratorExtender<IterBase> Iter;
  Iter it(def, def->GetBufferedRegion());
  it.SetDirection(0);

  for(; !it.IsAtEnd(); it.NextLine())
    {
    // Get the pointer to the begin of line
    VectorType *ptr = const_cast<VectorType *>(it.GetPosition());
    VectorType *ptr_end = ptr + def->GetBufferedRegion().GetSize(0);

    // Get the initial index
    typename LinearTransformType::InputPointType pt;
    for(int k = 0; k < VDim; k++)
      pt[k] = it.GetIndex()[k];

    for(; ptr < ptr_end; ++ptr, ++pt[0])
      {
      // Apply transform to the index. TODO: this is stupid, just use an offset
      typename LinearTransformType::OutputPointType pp = tran->TransformPoint(pt);
      for(int k = 0; k < VDim; k++)
        (*ptr)[k] = pp[k] - pt[k];
      }
    }
}



template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::VoxelWarpToPhysicalWarp(VectorImageType *warp, ImageBaseType *moving_space, VectorImageType *result)
{
  typedef VoxelToPhysicalWarpFunctor<VectorImageType> Functor;
  typedef UnaryPositionBasedFunctorImageFilter<VectorImageType,VectorImageType,Functor> Filter;
  Functor functor(warp, moving_space);

  typename Filter::Pointer filter = Filter::New();
  filter->SetFunctor(functor);
  filter->SetInput(warp);
  filter->GraftOutput(result);
  filter->Update();
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::PhysicalWarpToVoxelWarp(VectorImageType *warp, ImageBaseType *moving_space, VectorImageType *result)
{
  typedef PhysicalToVoxelWarpFunctor<VectorImageType> Functor;
  typedef UnaryPositionBasedFunctorImageFilter<VectorImageType,VectorImageType,Functor> Filter;
  Functor functor(warp, moving_space);

  typename Filter::Pointer filter = Filter::New();
  filter->SetFunctor(functor);
  filter->SetInput(warp);
  filter->GraftOutput(result);
  filter->Update();
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::DownsampleWarp(VectorImageType *srcWarp, VectorImageType *trgWarp, int srcLevel, int trgLevel)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Get the factor by which to downsample
  int src_factor = m_PyramidFactors[srcLevel];
  int trg_factor = m_PyramidFactors[trgLevel];
  if(src_factor < trg_factor)
    {
    // Resample the warp - no smoothing
    LDDMMType::vimg_resample_identity(srcWarp, this->GetReferenceSpace(trgLevel), trgWarp);

    // Scale by the factor
    LDDMMType::vimg_scale_in_place(trgWarp, src_factor / trg_factor);
    }
  else if(src_factor == trg_factor)
    {
    LDDMMType::vimg_copy(srcWarp, trgWarp);
    }
  else
    {
    throw GreedyException("DownsampleWarp called for upsampling");
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::WriteCompressedWarpInPhysicalSpace(int level, VectorImageType *warp, const char *filename, double precision)
{
  WriteCompressedWarpInPhysicalSpace(warp, this->GetMovingReferenceSpace(level), filename, precision);
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::WriteCompressedWarpInPhysicalSpace(VectorImageType *warp, ImageBaseType *moving_ref_space, const char *filename, double precision)
{
  // Define a _float_ output type, even if working with double precision (less space on disk)
  typedef itk::CovariantVector<float, VDim> OutputVectorType;
  typedef itk::Image<OutputVectorType, VDim> OutputWarpType;
  typedef CompressWarpFunctor<VectorImageType, OutputWarpType> Functor;

  typedef UnaryPositionBasedFunctorImageFilter<VectorImageType,OutputWarpType,Functor> Filter;
  Functor functor(warp, moving_ref_space, precision);

  typename Filter::Pointer filter = Filter::New();
  filter->SetFunctor(functor);
  filter->SetInput(warp);
  filter->Update();

  LDDMMData<float, VDim>::vimg_write(filter->GetOutput(), filename);
}

template <class TFloat, unsigned int VDim>
void 
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeWarpSquareRoot(
  VectorImageType *warp, VectorImageType *out, VectorImageType *work, 
  FloatImageType *error_norm, double tol, int max_iter)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Use more convenient variables
  VectorImageType *u = warp, *v = out;

  // Initialize the iterate to zero
  v->FillBuffer(typename LDDMMType::Vec(0.0));

  // Perform iteration
  for(int i = 0; i < max_iter; i++)
    {
    // Min and max norm of the error at this iteration
    TFloat norm_max = tol, norm_min = 0.0; 

    // Perform a single iteration
    LDDMMType::interp_vimg(v, v, 1.0, work);   // work = v(v(x))
    LDDMMType::vimg_scale_in_place(work, -1.0); // work = -v(v(x))
    LDDMMType::vimg_add_scaled_in_place(work, v, -1.0); // work = -v(v) - v(v(x))
    LDDMMType::vimg_add_in_place(work, u); // work = u - v - v(v(x)) = f - g o g

    // At this point, 'work' stores the difference between actual warp and square of the
    // square root estimate, i.e., the estimation error
    if(error_norm)
      {
      LDDMMType::vimg_norm_min_max(work, error_norm, norm_min, norm_max);
      std::cout << " " << norm_max << " " << std::endl;
      }

    // Update v - which is being put into the output anyway
    LDDMMType::vimg_add_scaled_in_place(v, work, 0.5);

    // Check the maximum delta
    std::cout << "." << std::flush;

    // Break if the tolerance bound reached
    if(norm_max < tol)
      break;
    }
}


template <class TFloat, unsigned int VDim>
void 
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeWarpRoot(VectorImageType *warp, VectorImageType *root, int exponent, TFloat tol, int max_iter)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // If the exponent is zero, return the image itself
  if(exponent == 0)
    {
    LDDMMType::vimg_copy(warp, root);
    return;
    }

  // Create the current root and the next root
  VectorImagePointer u = LDDMMType::new_vimg(warp);
  LDDMMType::vimg_copy(warp, u);

  // Create a working image
  VectorImagePointer work = LDDMMType::new_vimg(warp);

  // If there is tolerance, create an error norm image
  FloatImagePointer error_norm;
  if(tol > 0.0)
    {
    error_norm = LDDMMType::new_img(warp);
    }
    
  // Compute the square root 'exponent' times
  for(int k = 0; k < exponent; k++)
    {
    // Square root of u goes into root
    ComputeWarpSquareRoot(u, root, work, error_norm, tol, max_iter);
    std::cout << std::endl;

    // Copy root into u
    LDDMMType::vimg_copy(root, u);
    }
}



template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeDeformationFieldInverse(
    VectorImageType *warp, VectorImageType *uInverse, int n_sqrt, bool verbose)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Create a copy of the forward warp
  VectorImagePointer uForward = LDDMMType::new_vimg(warp);
  LDDMMType::vimg_copy(warp, uForward);

  // Create a working image 
  VectorImagePointer uWork = LDDMMType::new_vimg(warp);

  // Take the desired square root of the input warp and place into uForward
  ComputeWarpRoot(warp, uForward, n_sqrt);

  // Clear uInverse
  uInverse->FillBuffer(itk::NumericTraits<typename LDDMMType::Vec>::ZeroValue());

  // At this point, uForward holds the small deformation
  // Try to compute the inverse of the current forward transformation
  for(int i = 0; i < 20; i++)
    {
    // We are using uPhys as temporary storage
    LDDMMType::interp_vimg(uForward, uInverse, 1.0, uWork);
    LDDMMType::vimg_scale_in_place(uWork, -1.0);

    // Compute the maximum change from last iteration
    LDDMMType::vimg_subtract_in_place(uInverse, uWork);

    // std::cout << "inverse iter " << i << " change " << norm_max << std::endl;
    LDDMMType::vimg_copy(uWork, uInverse);
    }

  // Compose the inverses
  for(int i = 0; i < n_sqrt; i++)
    {
    LDDMMType::interp_vimg(uInverse, uInverse, 1.0, uWork);
    LDDMMType::vimg_add_in_place(uInverse, uWork);
    }

  // If verbose, compute the maximum error
  if(verbose)
    {
    FloatImagePointer iNorm = LDDMMType::new_img(uWork);
    LDDMMType::interp_vimg(uInverse, uForward, 1.0, uWork);
    LDDMMType::vimg_add_in_place(uWork, uForward);
    TFloat norm_min, norm_max;
    LDDMMType::vimg_norm_min_max(uWork, iNorm, norm_min, norm_max);
    std::cout << "Warp inverse max residual: " << norm_max << std::endl;
    }
}

// Explicitly instantiate this class
template class MultiImageOpticalFlowHelper<float, 2>;
template class MultiImageOpticalFlowHelper<float, 3>;
template class MultiImageOpticalFlowHelper<float, 4>;
template class MultiImageOpticalFlowHelper<double, 2>;
template class MultiImageOpticalFlowHelper<double, 3>;
template class MultiImageOpticalFlowHelper<double, 4>;
