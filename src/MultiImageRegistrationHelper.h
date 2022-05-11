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
#ifndef __MultiImageRegistrationHelper_h
#define __MultiImageRegistrationHelper_h

#include "itkImageBase.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkMatrixOffsetTransformBase.h"

#include "MultiComponentMetricReport.h"

template <class MultiComponentImageType, class BinnedImageType> class MutualInformationPreprocessingFilter;

/**
 * This class is used to perform mean square intensity difference type
 * registration with multiple images. The filter is designed for speed
 * of interpolation.
 */
template <class TFloat, unsigned int VDim>
class MultiImageOpticalFlowHelper
{
public:

  typedef itk::VectorImage<TFloat, VDim> MultiComponentImageType;
  typedef itk::Image<TFloat, VDim> FloatImageType;
  typedef itk::CovariantVector<TFloat, VDim> VectorType;
  typedef itk::Image<VectorType, VDim> VectorImageType;
  typedef itk::ImageBase<VDim> ImageBaseType;

  typedef typename MultiComponentImageType::Pointer MultiComponentImagePointer;
  typedef typename FloatImageType::Pointer FloatImagePointer;
  typedef typename VectorImageType::Pointer VectorImagePointer;

  typedef std::vector<int> PyramidFactorsType;
  typedef itk::Size<VDim> SizeType;
  typedef itk::CovariantVector<TFloat, VDim> Vec;

  typedef itk::MatrixOffsetTransformBase<TFloat, VDim, VDim> LinearTransformType;

  /** Set default (power of two) pyramid factors */
  void SetDefaultPyramidFactors(int n_levels);

  /** Set the pyramid factors - for multi-resolution (e.g., 8,4,2) */
  void SetPyramidFactors(const PyramidFactorsType &factors);

  /** 
   * Set whether the fixed images should be scaled down by the pyramid factors
   * when subsampling. This is needed for the Mahalanobis distance metric, but not for
   * any of the metrics that use image intensities 
   */
  void SetScaleFixedImageWithVoxelSize(bool onoff) { m_ScaleFixedImageWithVoxelSize = onoff; }

  /**
   * Start a new image group - an group is a set of fixed/moving multicomponent images that
   * share the same fixed and moving masks. Normally all input images will be placed in just
   * one group, but there are times where you want to group components based on different masks
   */
  void NewInputGroup();

  /** Add a pair of multi-component images to the class - same weight for each component */
  void AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight);

  /** Set the fixed image mask. It will just be used to set NaNs in the fixed image. */
  void SetFixedMask(FloatImageType *maskImage);

  /** Set the moving image mask */
  void SetMovingMask(FloatImageType *maskImage);

  /** Set jitter sigma - for jittering image samples in affine mode */
  void SetJitterSigma(double sigma);

  /** Set automatic mask generation radius */
  void SetGradientMaskTrimRadius(const std::vector<int> &radius);

  /** Downsample an image, taking care of NaNs if necessary */
  void DownsampleImage(VectorImageType *src, VectorImageType *dst, int factor, bool has_nans);
  
  /** Compute the composite image - must be run before any sampling is done */
  void BuildCompositeImages(double noise_sigma_relative, bool masked_downsampling,
                            SizeType fixed_mask_dilate_radius, SizeType moving_mask_dilate_radius,
                            bool zero_last_dim);

  /**
   * Apply a dilation to the masks - this is used with the NCC and WNCC metrics.
   * The dilated mask has two bands, as below:
   *   1.0 : voxel is inside the user-specified mask
   *   0.5 : voxel is within radius of the user-specified mask
   *   0.0 : voxel is outside of the user-specified mask
   *
   * NCC metric exploits this mask format for faster processing - region where the mask is zero are
   * excluded from NCC computation and accumulation.
   */
  void DilateCompositeGradientMasksForNCC(SizeType radius);

  /** Get number of pyramid levels */
  unsigned int GetNumberOfLevels() const { return m_PyramidFactors.size(); }

  /** Get number of input groups */
  unsigned int GetNumberOfInputGroups() const { return m_InputGroups.size(); }

  /** Get the reference image for level k */
  ImageBaseType *GetReferenceSpace(unsigned int level);

  /** Get the reference image for level k */
  ImageBaseType *GetMovingReferenceSpace(unsigned int group, unsigned int level);

  /** Get the fixed mask in given group at given pyramid level */
  FloatImageType *GetFixedMask(unsigned int group, unsigned int level)
    { return m_InputGroups[group].m_FixedPyramid.mask_pyramid[level]; }

  /** Get the moving mask in given group at given pyramid level */
  FloatImageType *GetMovingMask(unsigned int group, unsigned int level)
    { return m_InputGroups[group].m_MovingPyramid.mask_pyramid[level]; }

  /** Get the fixed image in given group at given pyramid level */
  MultiComponentImageType *GetFixedComposite(unsigned int group, unsigned int level)
    { return m_InputGroups[group].m_FixedPyramid.image_pyramid[level]; }

  /** Get the moving image in given group at given pyramid level */
  MultiComponentImageType *GetMovingComposite(unsigned int group, unsigned int level)
    { return m_InputGroups[group].m_MovingPyramid.image_pyramid[level]; }

  /** Get the smoothing factor for given level based on parameters */
  Vec GetSmoothingSigmasInPhysicalUnits(int level, double sigma, bool in_physical_units, bool zero_last_dim);

  /** Get the component weights in a group */
  vnl_vector<float> GetWeights(unsigned int group, double scaling = 1.0);

  /** Perform interpolation - compute [(I - J(Tx)) GradJ(Tx)] */
  void ComputeSSDMetricAndGradient(
      unsigned int group, unsigned int level,
      VectorImageType *def,
      bool weighted, double background_value,
      FloatImageType *out_metric_image,
      MultiComponentMetricReport &out_metric_report,
      VectorImageType *out_gradient, double result_scaling = 1.0);

  /** Perform interpolation - compute mutual information metric */
  void ComputeNMIMetricAndGradient(
      unsigned int group, unsigned int level,
      bool normalized_mutual_information,
      VectorImageType *def, FloatImageType *out_metric_image,
      MultiComponentMetricReport &out_metric_report,
      VectorImageType *out_gradient, double result_scaling = 1.0);

  /** Compute the NCC metric without gradient */
  void ComputeNCCMetricAndGradient(
      unsigned int group, unsigned int level,
      VectorImageType *def, const SizeType &radius, bool weighted,
      FloatImageType *out_metric_image,
      MultiComponentMetricReport &out_metric_report,
      VectorImageType *out_gradient = NULL, double result_scaling = 1.0);

  /** Compute the Mahalanobis metric with gradient */
  void ComputeMahalanobisMetricImage(
      unsigned int group, unsigned int level,
      VectorImageType *def,
      FloatImageType *out_metric_image, MultiComponentMetricReport &out_metric_report,
      VectorImageType *out_gradient = NULL);

  /** Compute affine similarity and gradient */
  void ComputeAffineSSDMetricAndGradient(
      unsigned int group, unsigned int level,
      LinearTransformType *tran,
      bool weighted, double background_value,
      FloatImageType *wrkMetric,
      MultiComponentMetricReport &metrics,
      LinearTransformType *grad_metric = NULL,
      LinearTransformType *grad_mask = NULL);

  void ComputeAffineNMIMetricAndGradient(
      unsigned int group, unsigned int level,
      bool normalized_mutual_info,
      LinearTransformType *tran,
      FloatImageType *wrkMetric,
      MultiComponentMetricReport &metrics,
      LinearTransformType *grad_metric = NULL,
      LinearTransformType *grad_mask = NULL);

  void ComputeAffineNCCMetricAndGradient(
      unsigned int group, unsigned int level,
      LinearTransformType *tran,
      const SizeType &radius,
      bool weighted,
      FloatImageType *wrkMetric,
      MultiComponentMetricReport &metrics,
      LinearTransformType *grad_metric = NULL,
      LinearTransformType *grad_mask = NULL);

  static void AffineToField(LinearTransformType *tran, VectorImageType *def);

  void DownsampleWarp(VectorImageType *srcWarp, VectorImageType *trgWarp, int srcLevel, int trgLevel);

  /** Convert a warp to physical space */
  static void VoxelWarpToPhysicalWarp(VectorImageType *warp, ImageBaseType *moving_space, VectorImageType *result);
  static void PhysicalWarpToVoxelWarp(VectorImageType *warp, ImageBaseType *moving_space, VectorImageType *result);

  /* 
   * Write a warp to a file. The warp must be in voxel space, not physical space 
   * this is the static version of this method
   */
  // static void WriteCompressedWarpInPhysicalSpace(
  //  VectorImageType *warp, ImageBaseType *moving_ref_space, const char *filename, double precision);

  /** Write a warp to a file. The warp must be in voxel space, not physical space */
  // void WriteCompressedWarpInPhysicalSpace(int level, VectorImageType *warp, const char *filename, double precision);

  /**
   * Invert a deformation field by first dividing it into small transformations using the
   * square root command, and then inverting the small transformations
   */
  static void ComputeDeformationFieldInverse(
    VectorImageType *warp, VectorImageType *result, int n_sqrt, bool verbose = false);

  /**
   * Compute the (2^k)-th root of a warp using an iterative scheme. For each
   * square root computation, the following iteration is used, where f = x + u
   * is the input warp, and g is the square root.
   */
  static void ComputeWarpRoot(
    VectorImageType *warp, VectorImageType *root, int exponent, TFloat tol = 0, int max_iter = 20);

  /**
   * Compute the square root of an input warp f = x + u(x) using an iterative scheme
   *
   *    g[0] = Id
   *    g[t+1] = g[t] + (f - g[t] o g[t]) / 2
   *
   * A working image of the same size as the input and output must be provided
   */
  static void ComputeWarpSquareRoot(
    VectorImageType *warp, VectorImageType *out, VectorImageType *work, 
    FloatImageType *error_norm = NULL, double tol = 0.0, int max_iter = 20);

  /**
   * Apply dilation to a mask, with option to make the dilated portion have value 0.5
   */
  static void DilateMask(FloatImageType *mask, SizeType radius, bool two_layer);

  /**
   * Internal method used to pack a bunch of multi-component images into a single one
   */

  MultiImageOpticalFlowHelper() : 
    m_JitterSigma(0.0), m_ScaleFixedImageWithVoxelSize(false) {}

protected:

  // Pyramid factors
  PyramidFactorsType m_PyramidFactors;

  // Vector of images
  typedef std::vector<typename MultiComponentImageType::Pointer> MultiCompImageSet;
  typedef std::vector<typename FloatImageType::Pointer> FloatImageSet;
  typedef std::vector<typename VectorImageType::Pointer> VectorImageSet;

  // A structure represetning an image pyramid layer
  struct ImagePyramid {
    // Multi-component image at full resolution
    typename MultiComponentImageType::Pointer image_full;

    // Mask image at full resolution
    typename FloatImageType::Pointer mask_full;

    // Pyramid of images
    MultiCompImageSet image_pyramid;

    // Pyramid of masks
    FloatImageSet mask_pyramid;

    // Noise factor for each component
    std::vector<double> noise_sigma;

    // Whether the original image had nans
    bool have_nans = false;
  };

  // Fixed and moving images intensity mapped into histogram binned
  typedef itk::VectorImage<unsigned char, VDim> BinnedImageType;

  // A structure representing a set of multicomponent fixed/moving images that
  // share the same fixed and moving masks. Typically there will only be one of
  // these image sets at a time, but sometimes you need to perform registration
  // with different inputs having different masks. In this case, you would split
  // the input into these groups.
  struct InputGroup
  {
    // Fixed and moving images (deallocated when pyramid is formed)
    MultiCompImageSet m_Fixed, m_Moving;

    // Fixed and moving masks (deallocated when pyramid is formed)
    typename FloatImageType::Pointer m_FixedMaskImage, m_MovingMaskImage;

    // Pyramid of images/masks at different resolution levels
    ImagePyramid m_FixedPyramid, m_MovingPyramid;

    // Weights for the components
    std::vector<double> m_Weights;

    // Working memory image for NCC computation
    typename MultiComponentImageType::Pointer m_NCCWorkingImage;

    // Binned images for mutual information
    typename BinnedImageType::Pointer m_FixedBinnedImage, m_MovingBinnedImage;
  };

  // Array of image assemblies
  std::vector<InputGroup> m_InputGroups;

  // Amount of jitter - for affine only
  double m_JitterSigma;

  // Gradient mask trim radius
  std::vector<int> m_GradientMaskTrimRadius;

  // Jitter composite
  VectorImageSet m_JitterComposite;

  // Remove NaNs and determine noise sigmas for either fixed or moving image
  void InitializePyramid(const MultiCompImageSet &src, FloatImageType *mask,
                         ImagePyramid &pyramid, double noise_sigma_rel,
                         bool masked_downsampling,
                         SizeType mask_dilate_radius,
                         bool scale_intensity_by_voxel_size,
                         bool zero_last_dim);

  void PlaceIntoComposite(FloatImageType *src, MultiComponentImageType *target, int offset);
  void PlaceIntoComposite(VectorImageType *src, MultiComponentImageType *target, int offset);

  // Adjust NCC radius to be smaller than half image size
  SizeType AdjustNCCRadius(int level, const SizeType &radius, bool report_on_adjust);

  // Precompute histograms for MI/NMI
  void ComputeHistogramsIfNeeded(unsigned int group, unsigned int level);

  // Whether the fixed images should be scaled down by the pyramid factors
  // when subsampling. This is needed for the Mahalanobis distance metric, but not for
  // any of the metrics that use image intensities
  bool m_ScaleFixedImageWithVoxelSize;
};

#endif
