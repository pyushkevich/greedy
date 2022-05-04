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
#ifndef MULTICOMPONENTIMAGEMETRICBASE_H
#define MULTICOMPONENTIMAGEMETRICBASE_H

#include "itkImageBase.h"
#include "itkImageToImageFilter.h"
#include "itkPoint.h"
#include "itkFixedArray.h"
#include "itkVectorImage.h"
#include "itkMatrixOffsetTransformBase.h"
#include "lddmm_common.h"

/**
 * Default traits for parameterizing the metric filters below
 */
template <class TReal, unsigned int VDim> struct DefaultMultiComponentImageMetricTraits
{
  typedef itk::VectorImage<TReal, VDim> InputImageType;
  typedef itk::Image<TReal, VDim> ScalarImageType;
  typedef itk::Image<itk::CovariantVector<TReal, VDim>, VDim> VectorImageType;

  typedef ScalarImageType MaskImageType;
  typedef VectorImageType DeformationFieldType;
  typedef VectorImageType GradientImageType;
  typedef ScalarImageType MetricImageType;
  typedef itk::MatrixOffsetTransformBase<TReal, VDim, VDim> TransformType;

  typedef TReal RealType;
};


/**
 * \class MultiComponentImageMetricBase
 *
 * Base class for metrics that compute similarity between two multi-component
 * images based on a deformation field. This filter is extended to support
 * normalized cross-correlation and least squares metrics.
 *
 * The metric takes the following inputs:
 *    Fixed multicomponent image (type TMetricTraits::InputImageType)
 *    Moving multicomponent image (type TMetricTraits::InputImageType)
 *    A mask for the fixed image (type TMetricTraits::MaskImageType) [optional]
 *    A deformation field (type TMetricTraits::DeformationFieldType)
 *
 * It produces the following outputs
 *    Metric image (type TMetricTraits::MetricImageType)
 *    Metric image gradient (type TMetricTraits::GradientImageType)
 *    Moving image domain mask (type TMetricTraits::MetricImageType)
 *    Moving image domain mask gradient (type TMetricTraits::GradientImageType)
 */
template <class TMetricTraits>
class ITK_EXPORT MultiComponentImageMetricBase :
    public itk::ImageToImageFilter<typename TMetricTraits::InputImageType,
                                   typename TMetricTraits::MetricImageType>
{
public:
  /** Type definitions from the traits class */
  typedef typename TMetricTraits::InputImageType        InputImageType;
  typedef typename TMetricTraits::MaskImageType         MaskImageType;
  typedef typename TMetricTraits::DeformationFieldType  DeformationFieldType;
  typedef typename TMetricTraits::MetricImageType       MetricImageType;
  typedef typename TMetricTraits::GradientImageType     GradientImageType;
  typedef typename TMetricTraits::TransformType         TransformType;
  typedef typename TMetricTraits::RealType              RealType;

  /** Standard class typedefs. */
  typedef MultiComponentImageMetricBase                            Self;
  typedef itk::ImageToImageFilter<InputImageType,MetricImageType>  Superclass;
  typedef itk::SmartPointer<Self>                                  Pointer;
  typedef itk::SmartPointer<const Self>                            ConstPointer;

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiComponentImageMetricBase, ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      MetricImageType::ImageDimension );

  /** Typedef to describe the output image region type. */
  typedef typename InputImageType::RegionType OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename InputImageType::PixelType          InputPixelType;
  typedef typename InputImageType::InternalPixelType  InputComponentType;
  typedef typename MetricImageType::PixelType         MetricPixelType;
  typedef typename MetricImageType::IndexType         IndexType;
  typedef typename MetricImageType::IndexValueType    IndexValueType;
  typedef typename MetricImageType::SizeType          SizeType;
  typedef typename MetricImageType::SpacingType       SpacingType;
  typedef typename MetricImageType::DirectionType     DirectionType;
  typedef typename GradientImageType::PixelType       GradientPixelType;
  typedef itk::ImageBase<ImageDimension>              ImageBaseType;

  typedef typename Superclass::DataObjectIdentifierType DataObjectIdentifierType;

  /** Information from the deformation field class */
  typedef typename DeformationFieldType::Pointer      DeformationFieldPointer;
  typedef typename DeformationFieldType::PixelType    DeformationVectorType;

  /** Weight vector */
  typedef vnl_vector<float>                           WeightVectorType;

  /** Set the fixed image(s) */
  itkNamedInputMacro(FixedImage, InputImageType, "Primary")

  /** Set the moving image(s) and their gradients */
  itkNamedInputMacro(MovingImage, InputImageType, "moving")

  /** Set the optional mask input */
  itkNamedInputMacro(FixedMaskImage, MaskImageType, "fixed_mask")

  /** Set the optional moving mask input */
  itkNamedInputMacro(MovingMaskImage, MaskImageType, "moving_mask")

  /** Set the optional jitter input - for affine images*/
  itkNamedInputMacro(JitterImage, DeformationFieldType, "jitter")

  /**
   * Set the deformation field. If the deformation field is set, the affine
   * transform gradients will not be computed.
   */
  void SetDeformationField(DeformationFieldType *phi)
  {
    this->itk::ProcessObject::SetInput("phi", phi);
    this->UpdateOutputs();
  }

  itkNamedInputGetMacro(DeformationField, DeformationFieldType, "phi")

  /**
   * Set the affine transform. If the affine transform is set, affine transform
   * gradients will be computed, but not the deformation field gradients
   */
  void SetAffineTransform(TransformType *transform)
  {
    this->m_AffineTransform = transform;
    this->m_ComputeAffine = true;
    this->UpdateOutputs();
  }

  itkGetMacro(AffineTransform, TransformType *)

  /** Set the weight vector - for different components in the input image */
  itkSetMacro(Weights, WeightVectorType)
  itkGetConstMacro(Weights, WeightVectorType)

  /** Whether or not the gradient is required */
  itkGetMacro(ComputeGradient, bool)

  /** Whether the transformation is affine */
  itkGetMacro(ComputeAffine, bool)

  /** Background value, i.e., default value of lookups outside of the mask */
  itkGetMacro(BackgroundValue, InputComponentType);
  itkSetMacro(BackgroundValue, InputComponentType);

  /**
   * Whether the metric is weighted by the moving image mask/domain. When a metric is weighted,
   * pixels that map outside of the moving image (or outside of the moving image mask) do not
   * count towards the metric. In other words, the outside is treated like missing data. This is
   * not supporeted by all the metrics
   */
  itkGetMacro(Weighted, bool)
  itkSetMacro(Weighted, bool)

  /** Specify whether the filter should compute gradients (whether affine or deformable) */
  void SetComputeGradient(bool flag)
  {
    this->m_ComputeGradient = flag;
    this->UpdateOutputs();
  }

  itkBooleanMacro(ComputeMovingDomainMask)

  /** Specify whether the metric should be normalized by the moving image domain */
  void SetComputeMovingDomainMask(bool flag)
  {
    this->m_ComputeMovingDomainMask = flag;
    this->UpdateOutputs();
  }

  /** Get the metric image output - this is the main output */
  itkNamedOutputMacro(MetricOutput, MetricImageType, "Primary")

  /** Get the metric dense gradient output. The gradient may be arbitrarily scaled. */
  itkNamedOutputMacro(DeformationGradientOutput, GradientImageType, "phi_gradient")

  /** Get the gradient of the affine transform */
  itkGetMacro(AffineTransformGradient, TransformType *)

  /** Get the gradient of the affine transform */
  itkGetMacro(AffineTransformMaskGradient, TransformType *)

  /**
   * Get the gradient scaling factor. To get the actual gradient of the metric, multiply the
   * gradient output of this filter by the scaling factor. Explanation: for efficiency, the
   * metrics return an arbitrarily scaled vector, such that adding the gradient to the
   * deformation field would INCREASE SIMILARITY. For metrics that are meant to be minimized,
   * this is the opposite of the gradient direction. For metrics that are meant to be maximized,
   * it is the gradient direction.
   */
  virtual double GetGradientScalingFactor() const = 0;

  /** Summary results after running the filter */
  itkGetConstMacro(MetricValue, double)
  itkGetConstMacro(MaskValue, double)

  /** Get the metric values per component (each component weighted) */
  vnl_vector<double> GetAllMetricValues() const;

  /**
   Data accumulated across multiple threads. The mutex should be used when accessing
   this data.
  */
  struct ThreadAccumulatedData {

    static const unsigned int GradientSize = ImageDimension * (ImageDimension+1);
    double metric, mask;
    vnl_vector<double> gradient, grad_mask, comp_metric;
    std::mutex mutex;

    ThreadAccumulatedData() : metric(0.0), mask(0.0),
      gradient(GradientSize, 0.0),
      grad_mask(GradientSize, 0.0) {}

    ThreadAccumulatedData(unsigned int ncomp) :
      ThreadAccumulatedData() { comp_metric.set_size(ncomp); comp_metric.fill(0.0); }

    void Accumulate(const ThreadAccumulatedData &other);
  };

protected:
  MultiComponentImageMetricBase();
  ~MultiComponentImageMetricBase() {}

  virtual void VerifyInputInformation() const ITK_OVERRIDE {}

  /** It is difficult to compute in advance the input image region
   * required to compute the requested output region. Thus the safest
   * thing to do is to request for the whole input image.
   *
   * For the deformation field, the input requested region
   * set to be the same as that of the output requested region. */
  virtual void GenerateInputRequestedRegion() ITK_OVERRIDE;

  virtual typename itk::DataObject::Pointer MakeOutput(const DataObjectIdentifierType &) ITK_OVERRIDE;

  void UpdateOutputs();
  void ToggleOutput(bool flag, const DataObjectIdentifierType &key);

  virtual void BeforeThreadedGenerateData() ITK_OVERRIDE;
  virtual void AfterThreadedGenerateData() ITK_OVERRIDE;



  // Weight vector
  WeightVectorType                m_Weights;

  bool m_ComputeMovingDomainMask;
  bool m_ComputeGradient;
  bool m_ComputeAffine;

  // Whether the metric is weighted
  bool m_Weighted = false;

  // The background value
  InputComponentType m_BackgroundValue = 0;

  // Per-thread data
  ThreadAccumulatedData m_AccumulatedData;

  // Accumulated metric value
  double m_MetricValue, m_MaskValue;

  // Affine transform
  typename TransformType::Pointer m_AffineTransform;

  // Gradient of the metric and mask with respect to affine parameters
  typename TransformType::Pointer m_AffineTransformGradient, m_AffineTransformMaskGradient;

private:
  MultiComponentImageMetricBase(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiComponentImageMetricBase.txx"
#endif


#endif // MULTICOMPONENTIMAGEMETRICBASE_H
