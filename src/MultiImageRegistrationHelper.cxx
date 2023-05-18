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
#include <vnl/vnl_random.h>
#include "lddmm_data.h"
#include "MultiImageOpticalFlowImageFilter.h"
#include "MultiComponentNCCImageMetric.h"
#include "MultiComponentMutualInfoImageMetric.h"
#include "MultiComponentWeightedNCCImageMetric.h"
#include "MahalanobisDistanceToTargetWarpMetric.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
#include "OneDimensionalInPlaceAccumulateFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkImageFileWriter.h"
#include "GreedyException.h"
#include "WarpFunctors.h"
#include "CompositeImageNanMaskingFilter.h"
#include "MultiComponentQuantileBasedNormalizationFilter.h"


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
::NewInputGroup()
{
  m_InputGroups.push_back(InputGroup());
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight)
{
  // Collect the weights
  for(unsigned i = 0; i < fixed->GetNumberOfComponentsPerPixel(); i++)
    m_InputGroups.back().m_Weights.push_back(weight);

  // Store the images
  m_InputGroups.back().m_Fixed.push_back(fixed);
  m_InputGroups.back().m_Moving.push_back(moving);
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetFixedMask(FloatImageType *maskImage)
{
  m_InputGroups.back().m_FixedMaskImage = maskImage;
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetMovingMask(FloatImageType *maskImage)
{
  m_InputGroups.back().m_MovingMaskImage = maskImage;
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
    for(unsigned int k = 0; k < VDim; k++)
      *trg_ptr++ = vsrc[k];
    trg_ptr += trg_skip;
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::DilateMask(FloatImageType *mask, SizeType radius, bool two_layer)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Threshold the mask itself
  LDDMMType::img_threshold_in_place(mask, 0.5, 1e100, 0.5, 0);

  // Make a copy of the mask
  typename FloatImageType::Pointer mask_copy = LDDMMType::new_img(mask);
  LDDMMType::img_copy(mask, mask_copy);

  // Run the accumulation filter on the mask
  typename FloatImageType::Pointer mask_accum =
      AccumulateNeighborhoodSumsInPlace(mask_copy.GetPointer(), radius);

  if(two_layer)
    {
    // Threshold the mask copy
    LDDMMType::img_threshold_in_place(mask_accum, 0.25, 1e100, 0.5, 0);

    // Add the two images - the result has 1 for the initial mask, 0.5 for the 'outer' mask
    LDDMMType::img_add_in_place(mask, mask_accum);
    }
  else
    {
    // Threshold the mask copy
    LDDMMType::img_threshold_in_place(mask_accum, 0.25, 1e100, 1.0, 0);

    // Add the two images - the result has 1 for the initial mask, 0.5 for the 'outer' mask
    LDDMMType::img_copy(mask_accum, mask);
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::DilateCompositeGradientMasksForNCC(SizeType radius)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Iterate over the image groups
  for(auto &group : m_InputGroups)
    {
    for(unsigned int level = 0; level < m_PyramidFactors.size(); level++)
      {
      // Dilate the fixed mask
      if(group.m_FixedPyramid.mask_pyramid[level])
        DilateMask(group.m_FixedPyramid.mask_pyramid[level], radius, true);
      }
    }
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::IMPair
MultiImageOpticalFlowHelper<TFloat, VDim>
::MergeMaskWithNanMask(
    MultiComponentImageType *src_image,
    FloatImageType *src_mask,
    bool have_nans,
    SizeType dilate_radius)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Make a copy of the supplied mask and dilate if needed
  FloatImagePointer mask = LDDMMType::img_dup(src_mask);
  if(mask && dilate_radius != SizeType::Filled(0))
    DilateMask(mask, dilate_radius, false);

  // Apply the nan mask to the current mask and image
  if(have_nans)
    {
    if(!mask)
      mask = LDDMMType::new_img(src_image, 1.0);

    // We duplicate the source image because the filter below runs in-place and
    // we might still need this image later
    MultiComponentImagePointer src_image_copy = LDDMMType::cimg_dup(src_image);

    typedef CompositeImageNanMaskingFilter<MultiComponentImageType, FloatImageType> NaNFilterType;
    typename NaNFilterType::Pointer nanfilter = NaNFilterType::New();
    nanfilter->SetInputCompositeImage(src_image_copy);
    nanfilter->SetInputMaskImage(mask);
    nanfilter->Update();
    return std::make_pair(nanfilter->GetOutputCompositeImage(), nanfilter->GetOutputMaskImage());
    }
  else return std::make_pair(src_image, mask);
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::InitializePyramid(const MultiCompImageSet &src,
                    FloatImageType *mask,
                    ImagePyramid &pyramid,
                    double noise_sigma_rel,
                    bool masked_downsampling,
                    SizeType mask_dilate_radius,
                    bool scale_intensity_by_voxel_size,
                    bool zero_last_dim)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Concatentate the input images into a single multi-component image
  MultiComponentImagePointer img_concat = LDDMMType::cimg_concat(src);

  unsigned int nc = img_concat->GetNumberOfComponentsPerPixel();

  // Noise for each components
  std::vector<double> comp_noise;
  bool have_nans = false;

  // If noise is requested, compute the range of the input image
  if(noise_sigma_rel > 0)
    {
    // Use the quantile filter
    typedef MultiComponentQuantileBasedNormalizationFilter<
        MultiComponentImageType, MultiComponentImageType> QuantileFilter;
    typename QuantileFilter::Pointer fltQuantile = QuantileFilter::New();
    fltQuantile->SetLowerQuantile(0.01);
    fltQuantile->SetUpperQuantile(0.99);
    fltQuantile->SetInput(img_concat);
    fltQuantile->SetNoRemapping(true);
    fltQuantile->Update();

    // Get the noise sigmas and nan count
    for(unsigned int j = 0; j < nc; j++)
      {
      double range = fltQuantile->GetUpperQuantileValue(j) - fltQuantile->GetLowerQuantileValue(j);
      comp_noise.push_back(noise_sigma_rel * range);

      if(fltQuantile->GetNumberOfNaNs(j) > 0)
        have_nans = true;
      }
    }
  else if(!mask)
    {
    // Otherwise check for nans but only if mask is not supplied
    have_nans = LDDMMType::cimg_nancount(img_concat) > 0;
    }

  // Get the full resolution image and mask after removing nans and dilating user mask
  IMPair q = MergeMaskWithNanMask(img_concat, mask, have_nans, mask_dilate_radius);
  pyramid.image_full = q.first;
  pyramid.mask_full = q.second;

  // Compute the pyramid
  pyramid.image_pyramid.resize(m_PyramidFactors.size());
  pyramid.mask_pyramid.resize(m_PyramidFactors.size());
  for(unsigned int i = 0; i < m_PyramidFactors.size(); i++)
    {
    if(m_PyramidFactors[i] == 1)
      {
      // Retain the image
      pyramid.image_pyramid[i] = pyramid.image_full;
      pyramid.mask_pyramid[i] = pyramid.mask_full;
      }
    else
      {
      // Determine the scaling factors depending on the image size
      typename LDDMMType::Vec adj_factors;
      SizeType level_mask_dilate_radius = mask_dilate_radius;
      for(unsigned int d = 0; d < VDim; d++)
        {
        int dim = pyramid.image_full->GetBufferedRegion().GetSize()[d];
        int max_factor = m_PyramidFactors[i];
        while(dim < max_factor && max_factor > 1)
          max_factor = max_factor >> 1;
        adj_factors[d] = std::min(m_PyramidFactors[i], max_factor);
        }

      if(zero_last_dim)
        adj_factors[VDim-1] = 1;

      // Adjust the dilation radius to get desired effect
      for(unsigned int d = 0; d < VDim; d++)
        level_mask_dilate_radius[d] *= adj_factors[d];

      // Get the full resolution image and mask after removing nans and dilating user mask
      IMPair imp_level = std::make_pair(pyramid.image_full, pyramid.mask_full);
      if(level_mask_dilate_radius != mask_dilate_radius)
        imp_level = MergeMaskWithNanMask(img_concat, mask, have_nans, level_mask_dilate_radius);

      // Downsample the image itself
      pyramid.image_pyramid[i] = LDDMMType::cimg_downsample(imp_level.first, adj_factors);
      if(imp_level.second)
        {
        // Downsample the mask
        pyramid.mask_pyramid[i] = LDDMMType::img_downsample(imp_level.second, adj_factors);

        // This command applies masked downsampling, i.e., we normalize the smoothed downsampled image
        // by the smoothed downsampled mask to avoid the bleeding in of pixels outside of the mask into
        // the downsampled image. The alternative below, is to threshold the mask and leave the image
        // without downsampling. This should be used with the NCC metric, which does not account for 'missing'
        // values and would be confused by the sharp boundary between the masked region and the background
        if(masked_downsampling)
          LDDMMType::cimg_mask_smooth_adjust_in_place(pyramid.image_pyramid[i], pyramid.mask_pyramid[i], 0.5);
        else
          LDDMMType::img_threshold_in_place(pyramid.mask_pyramid[i], 0.5,
                                            std::numeric_limits<double>::infinity(), 1.0, 0.0);
        }
      }

    // For some metrics, image intensity needs scaling (TODO: implement scaling for cimg)
    // if(scale_intensity_by_voxel_size)
    //   LDDMMType::img_scale_in_place(pyramid.image_pyramid[i], 1.0 / m_PyramidFactors[i]);

    // Apply random noise if requested to the pixels (using this randomly picked prime number of
    // random samples, repeated over and over
    if(comp_noise.size())
      LDDMMType::cimg_add_gaussian_noise_in_place(pyramid.image_pyramid[i], comp_noise, 17317);
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::BuildCompositeImages(double noise_sigma_relative, bool masked_downsampling,
                       SizeType fixed_mask_dilate_radius, SizeType moving_mask_dilate_radius,
                       bool zero_last_dim)
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Iterate over the image groups
  for(auto &group : m_InputGroups)
    {
    // Build the fixed pyramid
    this->InitializePyramid(group.m_Fixed, group.m_FixedMaskImage, group.m_FixedPyramid,
                            noise_sigma_relative,
                            masked_downsampling,
                            fixed_mask_dilate_radius,
                            m_ScaleFixedImageWithVoxelSize,
                            zero_last_dim);

    // Release memory
    group.m_Fixed.clear(); group.m_FixedMaskImage = nullptr;

    // Build the moving pyramid
    this->InitializePyramid(group.m_Moving, group.m_MovingMaskImage, group.m_MovingPyramid,
                            noise_sigma_relative,
                            masked_downsampling,
                            moving_mask_dilate_radius,
                            false,
                            zero_last_dim);

    // Release memory
    group.m_Moving.clear(); group.m_MovingMaskImage = nullptr;
    }

  // Set up the jitter images
  m_JitterComposite.resize(m_PyramidFactors.size(), nullptr);
  if(m_JitterSigma > 0)
    {
    for(unsigned int i = 0; i < m_PyramidFactors.size(); i++)
      {
      // Create a jitter image
      m_JitterComposite[i] = LDDMMType::new_vimg(this->GetReferenceSpace(i));
      LDDMMType::vimg_add_gaussian_noise_in_place(m_JitterComposite[i], m_JitterSigma, 17317);
      }
    }
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::ImageBaseType *
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetMovingReferenceSpace(unsigned int group, unsigned int level)
{
  return m_InputGroups[group].m_MovingPyramid.image_pyramid[level];
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::ImageBaseType *
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetReferenceSpace(unsigned int level)
{
  return m_InputGroups.front().m_FixedPyramid.image_pyramid[level];
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::Vec
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetSmoothingSigmasInPhysicalUnits(int level, double sigma, bool in_physical_units, bool zero_last_dim)
{
  Vec sigmas;
  if(in_physical_units)
    {
    sigmas.Fill(sigma * m_PyramidFactors[level]);
    }
  else
    {
    for(unsigned int k = 0; k < VDim; k++)
      sigmas[k] = this->GetReferenceSpace(level)->GetSpacing()[k] * sigma;
    }

  if(zero_last_dim)
    sigmas[VDim-1] = 0.0;

  return sigmas;
}

template<class TFloat, unsigned int VDim>
vnl_vector<float>
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetWeights(unsigned int group, double scaling)
{
  auto &w = m_InputGroups[group].m_Weights;
  vnl_vector<float> wscaled(w.size());
  for (unsigned i = 0; i < wscaled.size(); i++)
    wscaled[i] = w[i] * scaling;

  return wscaled;
}



template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeSSDMetricAndGradient(unsigned int group, unsigned int level,
                              VectorImageType *def,
                              bool weighted, double background_value,
                              FloatImageType *out_metric_image,
                              MultiComponentMetricReport &out_metric_report,
                              VectorImageType *out_gradient,
                              double result_scaling)
{
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiImageOpticalFlowImageFilter<TraitsType> FilterType;

  // Create a new filter
  typename FilterType::Pointer filter = FilterType::New();

  // TODO: this needs to be controlled by parameters, etc.
  // 'false' represents compatibility with previous greedy versions
  filter->SetUseDemonsGradientForm(false);
  filter->SetDemonsSigma(0.01);

  // Run the filter
  filter->SetFixedImage(this->GetFixedComposite(group, level));
  filter->SetMovingImage(this->GetMovingComposite(group, level));
  filter->SetFixedMaskImage(this->GetFixedMask(group, level));
  filter->SetMovingMaskImage(this->GetMovingMask(group, level));
  filter->SetWeights(this->GetWeights(group, result_scaling));
  filter->SetDeformationField(def);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric_image);
  filter->GetDeformationGradientOutput()->Graft(out_gradient);
  filter->SetWeighted(weighted);
  filter->SetBackgroundValue(background_value);
  filter->Update();

  // Process the results
  out_metric_report.ComponentPerPixelMetrics = filter->GetAllMetricValues();
  out_metric_report.TotalPerPixelMetric = filter->GetMetricValue();
  out_metric_report.MaskVolume = filter->GetMaskValue();
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeHistogramsIfNeeded(unsigned int group, unsigned int level)
{
  typedef MultiComponentQuantileBasedNormalizationFilter<
      MultiComponentImageType, BinnedImageType> BinnerType;
  InputGroup &grp = m_InputGroups[group];

  if(grp.m_FixedBinnedImage.IsNull()
     || (grp.m_FixedBinnedImage->GetBufferedRegion()
         != grp.m_FixedPyramid.image_pyramid[level]->GetBufferedRegion()))
    {
    typename BinnerType::Pointer fixed_binner = BinnerType::New();
    fixed_binner->SetInput(grp.m_FixedPyramid.image_pyramid[level]);
    fixed_binner->SetLowerQuantile(0.01);
    fixed_binner->SetUpperQuantile(0.99);
    fixed_binner->SetLowerQuantileOutputValue(1);
    fixed_binner->SetUpperQuantileOutputValue(127);
    fixed_binner->SetLowerOutOfRangeOutputValue(0);
    fixed_binner->Update();
    grp.m_FixedBinnedImage = fixed_binner->GetOutput();

    typename BinnerType::Pointer moving_binner = BinnerType::New();
    moving_binner = BinnerType::New();
    moving_binner->SetInput(grp.m_MovingPyramid.image_pyramid[level]);
    moving_binner->SetLowerQuantile(0.01);
    moving_binner->SetUpperQuantile(0.99);
    moving_binner->SetLowerQuantileOutputValue(1);
    moving_binner->SetUpperQuantileOutputValue(127);
    moving_binner->SetLowerOutOfRangeOutputValue(0);
    moving_binner->Update();
    grp.m_MovingBinnedImage = moving_binner->GetOutput();
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeNMIMetricAndGradient(unsigned int group, unsigned int level,
                              bool normalized_mutual_information,
                              VectorImageType *def,
                              FloatImageType *out_metric_image,
                              MultiComponentMetricReport &out_metric_report,
                              VectorImageType *out_gradient,
                              double result_scaling)
{
  // Set up the mutual information metric
  typedef DefaultMultiComponentMutualInfoImageMetricTraits<TFloat, unsigned char, VDim> TraitsType;
  typedef MultiComponentMutualInfoImageMetric<TraitsType> MetricType;

  typedef itk::VectorImage<unsigned char, VDim> BinnedImageType;
  typedef MutualInformationPreprocessingFilter<MultiComponentImageType, BinnedImageType> BinnerType;

  // Initialize the histograms
  this->ComputeHistogramsIfNeeded(group, level);

  typename MetricType::Pointer metric = MetricType::New();

  metric->SetComputeNormalizedMutualInformation(normalized_mutual_information);
  metric->SetBins(128);

  metric->SetFixedImage(m_InputGroups[group].m_FixedBinnedImage);
  metric->SetMovingImage(m_InputGroups[group].m_MovingBinnedImage);
  metric->SetWeights(this->GetWeights(group, result_scaling));
  metric->SetDeformationField(def);
  metric->SetComputeGradient(true);
  metric->GetMetricOutput()->Graft(out_metric_image);
  metric->GetDeformationGradientOutput()->Graft(out_gradient);
  metric->Update();

  // Process the results
  out_metric_report.ComponentPerPixelMetrics = metric->GetAllMetricValues();
  out_metric_report.TotalPerPixelMetric = metric->GetMetricValue();
  out_metric_report.MaskVolume = metric->GetMaskValue();
}

// #undef DUMP_NCC
#define DUMP_NCC 1


template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::SizeType
MultiImageOpticalFlowHelper<TFloat, VDim>
::AdjustNCCRadius(int level, const SizeType &radius, bool report_on_adjust)
{
  SizeType radius_fix = radius;
  SizeType image_dim = this->GetReferenceSpace(level)->GetBufferedRegion().GetSize();
  for(unsigned int d = 0; d < VDim; d++)
    {
    if(radius_fix[d] * 2 + 1 >= image_dim[d])
      radius_fix[d] = (image_dim[d] - 1) / 2;
    }

  if(report_on_adjust && radius != radius_fix)
    {
    std::cout << "  *** NCC radius adjusted to " << radius_fix
              << " because image too small at level " << level
              << " (" << image_dim << ")" << std::endl;
    }

  return radius_fix;
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeNCCMetricAndGradient(unsigned int group, unsigned int level,
                              VectorImageType *def,
                              const SizeType &radius,
                              bool weighted,
                              FloatImageType *out_metric_image,
                              MultiComponentMetricReport &out_metric_report,
                              VectorImageType *out_gradient,
                              double result_scaling,
                              bool minimization_mode)
{
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiComponentWeightedNCCImageMetric<TraitsType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  // Access current group
  InputGroup &grp = m_InputGroups[group];

  // Allocate a working image
  if(grp.m_NCCWorkingImage.IsNull())
    grp.m_NCCWorkingImage = MultiComponentImageType::New();

  // Is this the first time that this function is being called with this image?
  bool first_run =
      grp.m_NCCWorkingImage->GetBufferedRegion() != this->GetReferenceSpace(level)->GetBufferedRegion();

  // Check the radius against the size of the image
  SizeType radius_fix = AdjustNCCRadius(level, radius, first_run);

  // NCC-specific settings
  filter->SetRadius(radius_fix);
  filter->SetReuseWorkingImageFixedComponents(!first_run);
  filter->SetWeighted(weighted);
  filter->SetWorkingImage(grp.m_NCCWorkingImage);

  // Images and masks
  filter->SetFixedImage(grp.m_FixedPyramid.image_pyramid[level]);
  filter->SetMovingImage(grp.m_MovingPyramid.image_pyramid[level]);
  filter->SetFixedMaskImage(grp.m_FixedPyramid.mask_pyramid[level]);
  filter->SetMovingMaskImage(grp.m_MovingPyramid.mask_pyramid[level]);
  filter->SetWeights(this->GetWeights(group, result_scaling));
  filter->SetGradientDescentMinimizationMode(minimization_mode);

  // Inputs and outputs
  filter->SetDeformationField(def);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric_image);
  filter->GetDeformationGradientOutput()->Graft(out_gradient);

  filter->Update();

  // Get the vector of the normalized metrics
  out_metric_report.ComponentPerPixelMetrics = filter->GetAllMetricValues();
  out_metric_report.TotalPerPixelMetric = filter->GetMetricValue();
  out_metric_report.MaskVolume = filter->GetMaskValue();
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeMahalanobisMetricImage(
    unsigned int group, unsigned int level,
    VectorImageType *def,
    FloatImageType *out_metric_image, MultiComponentMetricReport &out_metric_report,
    VectorImageType *out_gradient)
{
  // Access current group
  InputGroup &grp = m_InputGroups[group];

  typedef DefaultMahalanobisDistanceToTargetMetricTraits<TFloat, VDim> TraitsType;
  typedef MahalanobisDistanceToTargetWarpMetric<TraitsType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();

  filter->SetFixedImage(grp.m_FixedPyramid.image_pyramid[level]);
  filter->SetMovingImage(grp.m_MovingPyramid.image_pyramid[level]);
  filter->SetFixedMaskImage(grp.m_FixedPyramid.mask_pyramid[level]);
  filter->SetMovingMaskImage(grp.m_MovingPyramid.mask_pyramid[level]);

  filter->SetDeformationField(def);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric_image);
  filter->GetDeformationGradientOutput()->Graft(out_gradient);

  filter->Update();

  out_metric_report.ComponentPerPixelMetrics = filter->GetAllMetricValues();
  out_metric_report.TotalPerPixelMetric = filter->GetMetricValue();
  out_metric_report.MaskVolume = filter->GetMaskValue();
}


template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineSSDMetricAndGradient(
    unsigned int group, unsigned int level,
    LinearTransformType *tran,
    bool weighted, double background_value,
    FloatImageType *wrkMetric,
    MultiComponentMetricReport &out_metric,
    LinearTransformType *grad_metric,
    LinearTransformType *grad_mask)
{
  // Set up the optical flow computation
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiImageOpticalFlowImageFilter<TraitsType> MetricType;
  typename MetricType::Pointer metric = MetricType::New();

  InputGroup &grp = m_InputGroups[group];

  metric->SetFixedImage(grp.m_FixedPyramid.image_pyramid[level]);
  metric->SetMovingImage(grp.m_MovingPyramid.image_pyramid[level]);
  metric->SetFixedMaskImage(grp.m_FixedPyramid.mask_pyramid[level]);
  metric->SetMovingMaskImage(grp.m_MovingPyramid.mask_pyramid[level]);
  metric->SetWeights(this->GetWeights(group));

  metric->SetAffineTransform(tran);
  metric->SetComputeMovingDomainMask(true);
  metric->GetMetricOutput()->Graft(wrkMetric);
  metric->SetComputeGradient(grad_metric != NULL);
  metric->SetJitterImage(m_JitterComposite[level]);

  metric->SetWeighted(weighted);
  metric->SetBackgroundValue(background_value);

  metric->Update();

  // Process the results
  if(grad_metric)
    {
    grad_metric->SetMatrix(metric->GetAffineTransformGradient()->GetMatrix());
    grad_metric->SetOffset(metric->GetAffineTransformGradient()->GetOffset());
    }
  if(grad_mask)
    {
    grad_mask->SetMatrix(metric->GetAffineTransformMaskGradient()->GetMatrix());
    grad_mask->SetOffset(metric->GetAffineTransformMaskGradient()->GetOffset());
    }

  out_metric.TotalPerPixelMetric = metric->GetMetricValue();
  out_metric.ComponentPerPixelMetrics = metric->GetAllMetricValues();
  out_metric.MaskVolume = metric->GetMaskValue();
}

#include "itkRescaleIntensityImageFilter.h"

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineNMIMetricAndGradient(
    unsigned int group, unsigned int level,
    bool normalized_mutual_info,
    LinearTransformType *tran,
    FloatImageType *wrkMetric,
    MultiComponentMetricReport &out_metric,
    LinearTransformType *grad_metric,
    LinearTransformType *grad_mask)
{
  // Set up the mutual information metric
  typedef DefaultMultiComponentMutualInfoImageMetricTraits<TFloat, unsigned char, VDim> TraitsType;
  typedef MultiComponentMutualInfoImageMetric<TraitsType> MetricType;
  typedef itk::VectorImage<unsigned char, VDim> BinnedImageType;
  typedef MutualInformationPreprocessingFilter<MultiComponentImageType, BinnedImageType> BinnerType;

  InputGroup &grp = m_InputGroups[group];

  // Initialize the histograms
  this->ComputeHistogramsIfNeeded(group, level);

  typename MetricType::Pointer metric = MetricType::New();

  metric->SetComputeNormalizedMutualInformation(normalized_mutual_info);
  metric->SetFixedImage(grp.m_FixedBinnedImage);
  metric->SetMovingImage(grp.m_MovingBinnedImage);
  metric->SetFixedMaskImage(grp.m_FixedPyramid.mask_pyramid[level]);
  metric->SetMovingMaskImage(grp.m_MovingPyramid.mask_pyramid[level]);
  metric->SetWeights(this->GetWeights(group));
  metric->SetAffineTransform(tran);
  metric->SetComputeMovingDomainMask(true);
  metric->GetMetricOutput()->Graft(wrkMetric);
  metric->SetComputeGradient(grad_metric != NULL);
  metric->SetBins(128);
  metric->SetJitterImage(m_JitterComposite[level]);
  metric->Update();

  // Process the results
  if(grad_metric)
    {
    grad_metric->SetMatrix(metric->GetAffineTransformGradient()->GetMatrix());
    grad_metric->SetOffset(metric->GetAffineTransformGradient()->GetOffset());
    }
  if(grad_mask)
    {
    grad_mask->SetMatrix(metric->GetAffineTransformMaskGradient()->GetMatrix());
    grad_mask->SetOffset(metric->GetAffineTransformMaskGradient()->GetOffset());
    }

  out_metric.TotalPerPixelMetric = metric->GetMetricValue();
  out_metric.ComponentPerPixelMetrics = metric->GetAllMetricValues();
  out_metric.MaskVolume = metric->GetMaskValue();
}



// TODO: there is a lot of code duplication here!
template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineNCCMetricAndGradient(
    unsigned int group, unsigned int level,
    LinearTransformType *tran,
    const SizeType &radius,
    bool weighted,
    FloatImageType *wrkMetric,
    MultiComponentMetricReport &out_metric,
    LinearTransformType *grad_metric,
    LinearTransformType *grad_mask)
{  
  InputGroup &grp = m_InputGroups[group];

  // Allocate a working image
  if(grp.m_NCCWorkingImage.IsNull())
    grp.m_NCCWorkingImage = MultiComponentImageType::New();

  // Set up the optical flow computation
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiComponentWeightedNCCImageMetric<TraitsType> MetricType;
  typename MetricType::Pointer metric = MetricType::New();

  // Is this the first time that this function is being called with this image?
  bool first_run =
      grp.m_NCCWorkingImage->GetBufferedRegion() != this->GetReferenceSpace(level)->GetBufferedRegion();

  // Check the radius against the size of the image
  SizeType radius_fix = AdjustNCCRadius(level, radius, first_run);

  metric->SetFixedImage(grp.m_FixedPyramid.image_pyramid[level]);
  metric->SetMovingImage(grp.m_MovingPyramid.image_pyramid[level]);
  metric->SetFixedMaskImage(grp.m_FixedPyramid.mask_pyramid[level]);
  metric->SetMovingMaskImage(grp.m_MovingPyramid.mask_pyramid[level]);
  metric->SetWeights(this->GetWeights(group));
  metric->SetAffineTransform(tran);
  metric->SetComputeMovingDomainMask(false);
  metric->GetMetricOutput()->Graft(wrkMetric);
  metric->SetComputeGradient(grad_metric != NULL);
  metric->SetRadius(radius_fix);
  metric->SetWorkingImage(grp.m_NCCWorkingImage);
  metric->SetReuseWorkingImageFixedComponents(!first_run);
  metric->SetJitterImage(m_JitterComposite[level]);
  // metric->GetDeformationGradientOutput()->Graft(wrkGradMetric);
  metric->SetWeighted(weighted);
  metric->Update();

  // Process the results
  if(grad_metric)
    {
    grad_metric->SetMatrix(metric->GetAffineTransformGradient()->GetMatrix());
    grad_metric->SetOffset(metric->GetAffineTransformGradient()->GetOffset());
    }
  if(grad_mask)
    {
    grad_mask->SetMatrix(metric->GetAffineTransformMaskGradient()->GetMatrix());
    grad_mask->SetOffset(metric->GetAffineTransformMaskGradient()->GetOffset());
    }

  out_metric.TotalPerPixelMetric = metric->GetMetricValue();
  out_metric.ComponentPerPixelMetrics = metric->GetAllMetricValues();
  out_metric.MaskVolume = metric->GetMaskValue();
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
    for(unsigned int k = 0; k < VDim; k++)
      pt[k] = it.GetIndex()[k];

    for(; ptr < ptr_end; ++ptr, ++pt[0])
      {
      // Apply transform to the index. TODO: this is stupid, just use an offset
      typename LinearTransformType::OutputPointType pp = tran->TransformPoint(pt);
      for(unsigned int k = 0; k < VDim; k++)
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

/*
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
*/

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
