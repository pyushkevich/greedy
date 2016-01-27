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
#include "itkVectorImage.h"
#include "itkMatrixOffsetTransformBase.h"



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

  /** Add a pair of multi-component images to the class - same weight for each component */
  void AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight);

  /** Set the gradient image mask */
  void SetGradientMask(FloatImageType *maskImage) { m_GradientMaskImage = maskImage; }

  /** Set jitter sigma - for jittering image samples in affine mode */
  void SetJitterSigma(double sigma);

  /** Compute the composite image - must be run before any sampling is done */
  void BuildCompositeImages(double noise_sigma_relative = 0.0);

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

  /** Get the reference image for level k */
  ImageBaseType *GetReferenceSpace(int level);

  /** Get the reference image for level k */
  ImageBaseType *GetMovingReferenceSpace(int level);

  /** Get the gradient mask at a pyramid level */
  FloatImageType *GetGradientMask(int level) { return m_GradientMaskComposite[level]; }

  /** Get the fixed image at a pyramid level */
  MultiComponentImageType *GetFixedComposite(int level) { return m_FixedComposite[level]; }

  /** Get the moving image at a pyramid level */
  MultiComponentImageType *GetMovingComposite(int level) { return m_MovingComposite[level]; }

  /** Get the smoothing factor for given level based on parameters */
  Vec GetSmoothingSigmasInPhysicalUnits(int level, double sigma, bool in_physical_units);

  /** Perform interpolation - compute [(I - J(Tx)) GradJ(Tx)] */
  vnl_vector<double> ComputeOpticalFlowField(
      int level, VectorImageType *def, FloatImageType *out_metric,
      VectorImageType *out_gradient, double result_scaling = 1.0);

  /** Perform interpolation - compute mutual information metric */
  vnl_vector<double> ComputeMIFlowField(
      int level, bool normalized_mutual_information,
      VectorImageType *def, FloatImageType *out_metric,
      VectorImageType *out_gradient, double result_scaling = 1.0);

  /** Compute the NCC metric without gradient */
  double ComputeNCCMetricImage(int level, VectorImageType *def, const SizeType &radius,
                              FloatImageType *out_metric, VectorImageType *out_gradient = NULL,
                               double result_scaling = 1.0);



  /** Compute affine similarity and gradient */
  double ComputeAffineMSDMatchAndGradient(int level, LinearTransformType *tran,
                                          FloatImageType *wrkMetric,
                                          FloatImageType *wrkMask,
                                          VectorImageType *wrkGradMetric,
                                          VectorImageType *wrkGradMask,
                                          VectorImageType *wrkPhi,
                                          LinearTransformType *grad = NULL);


  double ComputeAffineMIMatchAndGradient(int level, bool normalized_mutual_info,
                                         LinearTransformType *tran,
                                         FloatImageType *wrkMetric,
                                         FloatImageType *wrkMask,
                                         VectorImageType *wrkGradMetric,
                                         VectorImageType *wrkGradMask,
                                         VectorImageType *wrkPhi,
                                         LinearTransformType *grad = NULL);

  double ComputeAffineNCCMatchAndGradient(int level, LinearTransformType *tran,
                                          const SizeType &radius,
                                          FloatImageType *wrkMetric,
                                          FloatImageType *wrkMask,
                                          VectorImageType *wrkGradMetric,
                                          VectorImageType *wrkGradMask,
                                          VectorImageType *wrkPhi,
                                          LinearTransformType *grad = NULL);

  static void AffineToField(LinearTransformType *tran, VectorImageType *def);

  /** Convert a warp to physical space */
  void VoxelWarpToPhysicalWarp(int level, VectorImageType *warp, VectorImageType *result);

  /** Write a warp to a file. The warp must be in voxel space, not physical space */
  void WriteCompressedWarpInPhysicalSpace(int level, VectorImageType *warp, const char *filename, double precision);

  /**
   * Invert a deformation field by first dividing it into small transformations using the
   * square root command, and then inverting the small transformations
   */
  void ComputeDeformationFieldInverse(VectorImageType *warp, VectorImageType *result, int n_sqrt);

  MultiImageOpticalFlowHelper() : m_JitterSigma(0.0) {}

protected:

  // Pyramid factors
  PyramidFactorsType m_PyramidFactors;

  // Weights
  std::vector<double> m_Weights;

  // Vector of images
  typedef std::vector<typename MultiComponentImageType::Pointer> MultiCompImageSet;
  typedef std::vector<typename FloatImageType::Pointer> FloatImageSet;
  typedef std::vector<typename VectorImageType::Pointer> VectorImageSet;

  // Fixed and moving images
  MultiCompImageSet m_Fixed, m_Moving;

  // Composite image at each resolution level
  MultiCompImageSet m_FixedComposite, m_MovingComposite;

  // Working memory image for NCC computation
  typename MultiComponentImageType::Pointer m_NCCWorkingImage;

  // Gradient mask image - used to multiply the gradient
  typename FloatImageType::Pointer m_GradientMaskImage;

  // Gradient mask composite
  FloatImageSet m_GradientMaskComposite;

  // Amount of jitter - for affine only
  double m_JitterSigma;

  // Jitter composite
  VectorImageSet m_JitterComposite;

  void PlaceIntoComposite(FloatImageType *src, MultiComponentImageType *target, int offset);
  void PlaceIntoComposite(VectorImageType *src, MultiComponentImageType *target, int offset);
};


#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiImageRegistrationHelper.txx"
#endif

#endif
