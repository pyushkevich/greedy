/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: SimpleWarpImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2009-10-29 11:19:00 $
  Version:   $Revision: 1.31 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

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

  typedef std::vector<int> PyramidFactorsType;
  typedef itk::Size<VDim> SizeType;

  typedef itk::MatrixOffsetTransformBase<double, VDim, VDim> LinearTransformType;

  /** Set default (power of two) pyramid factors */
  void SetDefaultPyramidFactors(int n_levels);

  /** Set the pyramid factors - for multi-resolution (e.g., 8,4,2) */
  void SetPyramidFactors(const PyramidFactorsType &factors);

  /** Add a pair of multi-component images to the class - same weight for each component */
  void AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight);

  /** Set the gradient image mask */
  void SetGradientMask(FloatImageType *maskImage) { m_GradientMaskImage = maskImage; }

  /** Compute the composite image - must be run before any sampling is done */
  void BuildCompositeImages(bool add_noise);

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

  /** Perform interpolation - compute [(I - J(Tx)) GradJ(Tx)] */
  vnl_vector<double> ComputeOpticalFlowField(
      int level, VectorImageType *def, FloatImageType *out_metric,
      VectorImageType *out_gradient, double result_scaling = 1.0);

  /** Perform interpolation - compute mutual information metric */
  vnl_vector<double> ComputeMIFlowField(
      int level, VectorImageType *def, FloatImageType *out_metric,
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


  double ComputeAffineMIMatchAndGradient(int level, LinearTransformType *tran,
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

  void VoxelWarpToPhysicalWarp(int level, VectorImageType *warp, VectorImageType *result);

protected:

  // Pyramid factors
  PyramidFactorsType m_PyramidFactors;

  // Weights
  std::vector<double> m_Weights;

  // Vector of images
  typedef std::vector<typename MultiComponentImageType::Pointer> MultiCompImageSet;
  typedef std::vector<typename FloatImageType::Pointer> FloatImageSet;

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

  void PlaceIntoComposite(FloatImageType *src, MultiComponentImageType *target, int offset);
  void PlaceIntoComposite(VectorImageType *src, MultiComponentImageType *target, int offset);
};


#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiImageRegistrationHelper.txx"
#endif

#endif
