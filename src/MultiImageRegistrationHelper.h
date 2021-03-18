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

  /** Add a pair of multi-component images to the class - same weight for each component */
  void AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight);

  /** Set the fixed image mask. It will just be used to set NaNs in the fixed image. */
  void SetFixedMask(FloatImageType *maskImage) { m_FixedMaskImage = maskImage; }

  /** Set the moving image mask */
  void SetMovingMask(FloatImageType *maskImage) { m_MovingMaskImage = maskImage; }

  /** Set jitter sigma - for jittering image samples in affine mode */
  void SetJitterSigma(double sigma);

  /** Set automatic mask generation radius */
  void SetGradientMaskTrimRadius(const std::vector<int> &radius);

  /** Downsample an image, taking care of NaNs if necessary */
  void DownsampleImage(VectorImageType *src, VectorImageType *dst, int factor, bool has_nans);
  
  /** Compute the composite image - must be run before any sampling is done */
  void BuildCompositeImages(double noise_sigma_relative, bool masked_downsampling);

  /**
   * Apply a dilation to the fixed gradient masks - this is used with the NCC metric. The initial
   * user-specified mask is transformed into a dilated mask, with values as follows:
   *   1.0 : voxel is inside the user-specified mask
   *   0.5 : voxel is within radius of the user-specified mask
   *   0.0 : voxel is outside of the user-specified mask
   *
   * NCC metrics exploit this mask format for faster processing - region where the mask is zero are
   * excluded from NCC computation and accumulation
   */
  void DilateCompositeGradientMasksForNCC(SizeType radius);

  /** Get number of pyramid levels */
  unsigned int GetNumberOfLevels() const { return m_PyramidFactors.size(); }

  /** Get the reference image for level k */
  ImageBaseType *GetReferenceSpace(int level);

  /** Get the reference image for level k */
  ImageBaseType *GetMovingReferenceSpace(int level);

  /** Get the moving mask at a pyramid level */
  FloatImageType *GetFixedMask(int level) { return m_FixedPyramid.mask_pyramid[level]; }

  /** Get the moving mask at a pyramid level */
  FloatImageType *GetMovingMask(int level) { return m_MovingPyramid.mask_pyramid[level]; }

  /** Get the fixed image at a pyramid level */
  MultiComponentImageType *GetFixedComposite(int level) { return m_FixedPyramid.image_pyramid[level]; }

  /** Get the moving image at a pyramid level */
  MultiComponentImageType *GetMovingComposite(int level) { return m_MovingPyramid.image_pyramid[level]; }

  /** Get the smoothing factor for given level based on parameters */
  Vec GetSmoothingSigmasInPhysicalUnits(int level, double sigma, bool in_physical_units);

  /** Get the component weights in the composite */
  const std::vector<double> &GetWeights() const { return m_Weights; }

  /** Perform interpolation - compute [(I - J(Tx)) GradJ(Tx)] */
  void ComputeOpticalFlowField(
      int level, VectorImageType *def, FloatImageType *out_metric_image,
      MultiComponentMetricReport &out_metric_report,
      VectorImageType *out_gradient, double result_scaling = 1.0);

  /** Perform interpolation - compute mutual information metric */
  void ComputeMIFlowField(
      int level, bool normalized_mutual_information,
      VectorImageType *def, FloatImageType *out_metric_image,
      MultiComponentMetricReport &out_metric_report,
      VectorImageType *out_gradient, double result_scaling = 1.0);

  /** Compute the NCC metric without gradient */
  void ComputeNCCMetricImage(int level, VectorImageType *def, const SizeType &radius, bool weighted,
                             FloatImageType *out_metric_image, MultiComponentMetricReport &out_metric_report,
                             VectorImageType *out_gradient = NULL, double result_scaling = 1.0);

  /** Compute the Mahalanobis metric with gradient */
  void ComputeMahalanobisMetricImage(int level, VectorImageType *def,
                                     FloatImageType *out_metric_image, MultiComponentMetricReport &out_metric_report,
                                     VectorImageType *out_gradient = NULL);

  /** Compute affine similarity and gradient */
  void ComputeAffineMSDMatchAndGradient(int level, LinearTransformType *tran,
                                        FloatImageType *wrkMetric,
                                        FloatImageType *wrkMask,
                                        VectorImageType *wrkGradMetric,
                                        VectorImageType *wrkGradMask,
                                        VectorImageType *wrkPhi,
                                        MultiComponentMetricReport &metrics,
                                        LinearTransformType *grad = NULL);


  void ComputeAffineMIMatchAndGradient(int level, bool normalized_mutual_info,
                                       LinearTransformType *tran,
                                       FloatImageType *wrkMetric,
                                       FloatImageType *wrkMask,
                                       VectorImageType *wrkGradMetric,
                                       VectorImageType *wrkGradMask,
                                       VectorImageType *wrkPhi,
                                       MultiComponentMetricReport &metrics,
                                       LinearTransformType *grad = NULL);

  void ComputeAffineNCCMatchAndGradient(int level, LinearTransformType *tran,
                                        const SizeType &radius,
                                        bool weighted,
                                        FloatImageType *wrkMetric,
                                        FloatImageType *wrkMask,
                                        VectorImageType *wrkGradMetric,
                                        VectorImageType *wrkGradMask,
                                        VectorImageType *wrkPhi,
                                        MultiComponentMetricReport &metrics,
                                        LinearTransformType *grad = NULL);

  static void AffineToField(LinearTransformType *tran, VectorImageType *def);

  void DownsampleWarp(VectorImageType *srcWarp, VectorImageType *trgWarp, int srcLevel, int trgLevel);

  /** Convert a warp to physical space */
  static void VoxelWarpToPhysicalWarp(VectorImageType *warp, ImageBaseType *moving_space, VectorImageType *result);
  static void PhysicalWarpToVoxelWarp(VectorImageType *warp, ImageBaseType *moving_space, VectorImageType *result);

  /* 
   * Write a warp to a file. The warp must be in voxel space, not physical space 
   * this is the static version of this method
   */
  static void WriteCompressedWarpInPhysicalSpace(
    VectorImageType *warp, ImageBaseType *moving_ref_space, const char *filename, double precision);

  /** Write a warp to a file. The warp must be in voxel space, not physical space */
  void WriteCompressedWarpInPhysicalSpace(int level, VectorImageType *warp, const char *filename, double precision);

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
   * Internal method used to pack a bunch of multi-component images into a single one
   */

  MultiImageOpticalFlowHelper() : 
    m_JitterSigma(0.0), m_ScaleFixedImageWithVoxelSize(false) {}

protected:

  // Pyramid factors
  PyramidFactorsType m_PyramidFactors;

  // Weights
  std::vector<double> m_Weights;

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

  // Fixed and moving images
  MultiCompImageSet m_Fixed, m_Moving;

  // Composite image at each resolution level
  ImagePyramid m_FixedPyramid, m_MovingPyramid;

  // Working memory image for NCC computation
  typename MultiComponentImageType::Pointer m_NCCWorkingImage;

  // Moving mask image - used to reduce region where metric is computed
  typename FloatImageType::Pointer m_MovingMaskImage;

  // Fixed mask image - used to reduce region where metric is computed
  typename FloatImageType::Pointer m_FixedMaskImage;

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
                         bool scale_intensity_by_voxel_size);

  void PlaceIntoComposite(FloatImageType *src, MultiComponentImageType *target, int offset);
  void PlaceIntoComposite(VectorImageType *src, MultiComponentImageType *target, int offset);

  // Adjust NCC radius to be smaller than half image size
  SizeType AdjustNCCRadius(int level, const SizeType &radius, bool report_on_adjust);

  // Precompute histograms for MI/NMI
  void ComputeHistogramsIfNeeded(int level);

  // Fixed and moving images intensity mapped into histogram binned
  typedef itk::VectorImage<unsigned char, VDim> BinnedImageType;
  typename BinnedImageType::Pointer m_FixedBinnedImage, m_MovingBinnedImage;

  // Whether the fixed images should be scaled down by the pyramid factors
  // when subsampling. This is needed for the Mahalanobis distance metric, but not for
  // any of the metrics that use image intensities
  bool m_ScaleFixedImageWithVoxelSize;
};

#endif
