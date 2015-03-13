/*=========================================================================

  Program:   ALFABIS fast image registration
  Language:  C++

  Copyright (c) Paul Yushkevich. All rights reserved.

  This program is part of ALFABIS

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
#ifndef __MultiImageOpticalFlowImageFilter_h
#define __MultiImageOpticalFlowImageFilter_h
#include "itkImageBase.h"
#include "itkImageToImageFilter.h"
#include "itkPoint.h"
#include "itkFixedArray.h"
#include "itkVectorImage.h"

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
};

/** \class MultiComponentImageMetricBase
 * \brief Warps an image using an input deformation field (for LDDMM)
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

  /** Standard class typedefs. */
  typedef MultiComponentImageMetricBase                            Self;
  typedef itk::ImageToImageFilter<InputImageType,MetricImageType>  Superclass;
  typedef itk::SmartPointer<Self>                                  Pointer;
  typedef itk::SmartPointer<const Self>                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

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
  void SetFixedImage(InputImageType *fixed)
    { this->itk::ProcessObject::SetInput("Primary", fixed); }

  InputImageType *GetFixedImage()
    { return dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary")); }

  /** Set the moving image(s) and their gradients */
  void SetMovingImage(InputImageType *moving)
    { this->itk::ProcessObject::SetInput("moving", moving); }

  InputImageType *GetMovingImage()
    { return dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving")); }

  /** Set the optional mask input */
  void SetFixedMaskImage(MaskImageType *fixed_mask)
    { this->itk::ProcessObject::SetInput("fixed_mask", fixed_mask); }

  MaskImageType *GetFixedMaskImage()
    { return dynamic_cast<MaskImageType *>(this->ProcessObject::GetInput("fixed_mask")); }


  /**
   * Set the deformation field. An affine transformation should be converted
   * to a deformation field first
   */
  void SetDeformationField(DeformationFieldType *phi)
    { this->itk::ProcessObject::SetInput("phi", phi); }
\
  DeformationFieldType *GetDeformationField()
    { return dynamic_cast<DeformationFieldType *>(this->ProcessObject::GetInput("phi")); }


  /** Set the weight vector */
  itkSetMacro(Weights, WeightVectorType)
  itkGetConstMacro(Weights, WeightVectorType)

  itkBooleanMacro(ComputeGradient)

  /** Specify whether the filter should compute gradients */
  void SetComputeGradient(bool flag)
  {
    this->m_ComputeGradient = flag;
    this->UpdateOutputs();
  }

  itkBooleanMacro(ComputeMovingDomainMask)

  /** Specify whether the filter should compute a mask based on the moving image domain */
  void SetComputeMovingDomainMask(bool flag)
  {
    this->m_ComputeMovingDomainMask = flag;
    this->UpdateOutputs();
  }

  /** Get the metric output */
  MetricImageType *GetMetricOutput()
    { return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput("Primary")); }

  /** Get the mask output */
  MetricImageType *GetMovingDomainMaskOutput()
    { return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput("moving_mask")); }

  /** Get the metric output */
  GradientImageType *GetGradientOutput()
    { return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput("gradient")); }

  /** Get the mask output */
  GradientImageType *GetMovingDomainMaskGradientOutput()
    { return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput("moving_mask_gradient")); }



  /** This filter produces an image which is a different
   * size than its input image. As such, it needs to provide an
   * implemenation for GenerateOutputInformation() which set
   * the output information according the OutputSpacing, OutputOrigin
   * and the deformation field's LargestPossibleRegion. */
  virtual void GenerateOutputInformation();

  /** It is difficult to compute in advance the input image region
   * required to compute the requested output region. Thus the safest
   * thing to do is to request for the whole input image.
   *
   * For the deformation field, the input requested region
   * set to be the same as that of the output requested region. */
  virtual void GenerateInputRequestedRegion();

  virtual typename itk::DataObject::Pointer MakeOutput(const DataObjectIdentifierType &);

protected:
  MultiComponentImageMetricBase();
  ~MultiComponentImageMetricBase() {}

  void VerifyInputInformation() {}

  void UpdateOutputs();
  void ToggleOutput(bool flag, const DataObjectIdentifierType &key);

  // Weight vector
  WeightVectorType                m_Weights;

  bool m_ComputeMovingDomainMask;
  bool m_ComputeGradient;

private:
  MultiComponentImageMetricBase(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};







/** \class MultiImageOpticalFlowImageFilter
 * \brief Warps an image using an input deformation field (for LDDMM)
 *
 * This filter efficiently computes the optical flow field between a
 * set of image pairs, given a transformation phi. This filter is the
 * workhorse of deformable and affine rigid registration algorithms that
 * use the mean squared difference metric. Given a set of fixed images F_i
 * and moving images M_i, it computes
 *
 *   v(x) = Sum_i w_i \[ F_i(x) - M_i(Phi(x)) ] \Grad M_i (Phi(x))
 *
 * The efficiency of this filter comes from combining the interpolation of
 * all the M and GradM terms in one loop, so that all possible computations
 * are reused
 *
 * The fixed and moving images must be passed in to the filter in the form
 * of VectorImages of size K and (VDim+K), respectively - i.e., the moving
 * images and their gradients are packed together.
 *
 * The output should be an image of CovariantVector type
 *
 * \warning This filter assumes that the input type, output type
 * and deformation field type all have the same number of dimensions.
 *
 */
template <class TMetricTraits>
class ITK_EXPORT MultiImageOpticalFlowImageFilter :
    public MultiComponentImageMetricBase<TMetricTraits>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageOpticalFlowImageFilter<TMetricTraits>   Self;
  typedef MultiComponentImageMetricBase<TMetricTraits>      Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageOpticalFlowImageFilter, MultiComponentImageMetricBase )

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename Superclass::InputPixelType                InputPixelType;
  typedef typename Superclass::InputComponentType            InputComponentType;
  typedef typename Superclass::MetricImageType               MetricImageType;
  typedef typename Superclass::GradientImageType             GradientImageType;
  typedef typename Superclass::MetricPixelType               MetricPixelType;
  typedef typename Superclass::GradientPixelType             GradientPixelType;

  typedef typename Superclass::IndexType                     IndexType;
  typedef typename Superclass::IndexValueType                IndexValueType;
  typedef typename Superclass::SizeType                      SizeType;
  typedef typename Superclass::SpacingType                   SpacingType;
  typedef typename Superclass::DirectionType                 DirectionType;
  typedef typename Superclass::ImageBaseType                 ImageBaseType;

  /** Information from the deformation field class */
  typedef typename Superclass::DeformationFieldType          DeformationFieldType;
  typedef typename Superclass::DeformationFieldPointer       DeformationFieldPointer;
  typedef typename Superclass::DeformationVectorType         DeformationVectorType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, InputImageType::ImageDimension );

  /** Summary results after running the filter */
  itkGetConstMacro(MetricValue, double)

  /** Get the metric values per component (each component weighted) */
  vnl_vector<double> GetAllMetricValues() const;

protected:
  MultiImageOpticalFlowImageFilter() {}
  ~MultiImageOpticalFlowImageFilter() {}

  /** This method is used to set the state of the filter before
   * multi-threading. */
  virtual void BeforeThreadedGenerateData();

  /** This method is used to set the state of the filter after
   * multi-threading. */
  virtual void AfterThreadedGenerateData();

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for
   * ThreadedGenerateData(). */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

private:
  MultiImageOpticalFlowImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // Structure describing the summary metric statistics
  struct MetricStats {
    vnl_vector<double> metric_values;
    double num_voxels;
  };

  // Vector of accumulated data (difference, gradient of affine transform, etc)
  double                          m_MetricValue;
  std::vector<MetricStats>        m_MetricStatsPerThread;
  MetricStats                     m_MetricStats;
};








/**
 * Normalized cross-correlation metric. This filter sets up a mini-pipeline with
 * a pre-compute filter that interpolates the moving image, N one-dimensional
 * mean filters, and a post-compute filter that generates the metric and the
 * gradient.
 */
template <class TMetricTraits>
class ITK_EXPORT MultiComponentNCCImageMetric :
    public MultiComponentImageMetricBase<TMetricTraits>
{
public:
  /** Standard class typedefs. */
  typedef MultiComponentNCCImageMetric<TMetricTraits>       Self;
  typedef MultiComponentImageMetricBase<TMetricTraits>      Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiComponentNCCImageMetric, MultiComponentImageMetricBase )

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename Superclass::InputPixelType                InputPixelType;
  typedef typename Superclass::InputComponentType            InputComponentType;
  typedef typename Superclass::MetricImageType               MetricImageType;
  typedef typename Superclass::GradientImageType             GradientImageType;
  typedef typename Superclass::MaskImageType                 MaskImageType;


  typedef typename Superclass::IndexType                     IndexType;
  typedef typename Superclass::IndexValueType                IndexValueType;
  typedef typename Superclass::SizeType                      SizeType;
  typedef typename Superclass::SpacingType                   SpacingType;
  typedef typename Superclass::DirectionType                 DirectionType;
  typedef typename Superclass::ImageBaseType                 ImageBaseType;

  /** Information from the deformation field class */
  typedef typename Superclass::DeformationFieldType          DeformationFieldType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, InputImageType::ImageDimension );

  /** Set the radius of the cross-correlation */
  itkSetMacro(Radius, SizeType)

  /** Get the radius of the cross-correlation */
  itkGetMacro(Radius, SizeType)

  /**
   * Set the working memory image for this filter. This function should be used to prevent
   * repeated allocation of memory when the metric is created/destructed in a loop. The
   * user can just pass in a pointer to a blank image, the filter will take care of allocating
   * the image as necessary
   */
  itkSetObjectMacro(WorkingImage, InputImageType)

  /** Summary results after running the filter */
  itkGetConstMacro(MetricValue, double)

protected:
  MultiComponentNCCImageMetric() {}
  ~MultiComponentNCCImageMetric() {}

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for
   * ThreadedGenerateData(). */
  void GenerateData();

private:
  MultiComponentNCCImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // A pointer to the working image. The user should supply this image in order to prevent
  // unnecessary memory allocation
  typename InputImageType::Pointer m_WorkingImage;

  // Radius of the cross-correlation
  SizeType m_Radius;

  // Vector of accumulated data (difference, gradient of affine transform, etc)
  double                          m_MetricValue;
};




/** \class MultiImageNCCPrecomputeFilter
 * \brief Warps an image using an input deformation field (for LDDMM)
 *
 * This filter takes a pair of images plus a warp and computes the components that
 * are used to calculate the cross-correlation metric between them and
 * the gradient. These components are in the form I, I*J, I * gradJ, and
 * so on. These components must then be mean-filtered and combined to get the
 * metric and the gradient.
 *
 * The output of this filter must be a vector image. The input may be a vector image.
 *
 */
template <class TMetricTraits, class TOutputImage>
class ITK_EXPORT MultiImageNCCPrecomputeFilter :
    public itk::ImageToImageFilter<typename TMetricTraits::InputImageType, TOutputImage>
{
public:

  /** Types from the traits */
  typedef typename TMetricTraits::InputImageType        InputImageType;
  typedef typename TMetricTraits::MaskImageType         MaskImageType;
  typedef typename TMetricTraits::DeformationFieldType  DeformationFieldType;
  typedef typename TMetricTraits::MetricImageType       MetricImageType;
  typedef typename TMetricTraits::GradientImageType     GradientImageType;
  typedef TOutputImage                                  OutputImageType;


  /** Standard class typedefs. */
  typedef MultiImageNCCPrecomputeFilter                         Self;
  typedef itk::ImageToImageFilter<InputImageType, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename InputImageType::PixelType          InputPixelType;
  typedef typename InputImageType::InternalPixelType  InputComponentType;
  typedef typename TOutputImage::PixelType            OutputPixelType;
  typedef typename TOutputImage::InternalPixelType    OutputComponentType;
  typedef typename DeformationFieldType::PixelType    DeformationVectorType;
  typedef typename MetricImageType::PixelType         MetricPixelType;
  typedef typename GradientImageType::PixelType       GradientPixelType;
  typedef typename MetricImageType::IndexType         IndexType;
  typedef typename MetricImageType::IndexValueType    IndexValueType;
  typedef typename MetricImageType::SizeType          SizeType;
  typedef typename MetricImageType::SpacingType       SpacingType;
  typedef typename MetricImageType::DirectionType     DirectionType;

  typedef typename Superclass::DataObjectIdentifierType DataObjectIdentifierType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageNCCPrecomputeFilter, ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension );

  itkBooleanMacro(ComputeGradient)

  /**
   * Whether the gradient of the NCC metric will be computed. Depending on this, the filter
   * will generate 5 components per pixel (x,y,xy,x2,y2) or a bunch more needed for the
   * gradient computation. Default is Off.
   */
  void SetComputeGradient(bool flag)
  {
    this->m_ComputeGradient = flag;
    this->UpdateOutputs();
  }

  itkBooleanMacro(ComputeMovingDomainMask)

  /** Specify whether the filter should compute a mask based on the moving image domain */
  void SetComputeMovingDomainMask(bool flag)
  {
    this->m_ComputeMovingDomainMask = flag;
    this->UpdateOutputs();
  }


  /** Set the fixed image(s) */
  void SetFixedImage(InputImageType *fixed)
    { this->itk::ProcessObject::SetInput("Primary", fixed); }

  InputImageType *GetFixedImage()
    { return dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary")); }

  /** Set the moving image(s) and their gradients */
  void SetMovingImage(InputImageType *moving)
    { this->itk::ProcessObject::SetInput("moving", moving); }

  InputImageType *GetMovingImage()
    { return dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving")); }

  /**
   * Set the deformation field. An affine transformation should be converted
   * to a deformation field first
   */
  void SetDeformationField(DeformationFieldType *phi)
    { this->itk::ProcessObject::SetInput("phi", phi); }
\
  DeformationFieldType *GetDeformationField()
    { return dynamic_cast<DeformationFieldType *>(this->ProcessObject::GetInput("phi")); }

  /** Get the metric output */
  MetricImageType *GetMetricOutput()
    { return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput("Primary")); }

  /** Get the mask output */
  MetricImageType *GetMovingDomainMaskOutput()
    { return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput("moving_mask")); }

  /** Get the metric output */
  GradientImageType *GetGradientOutput()
    { return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput("gradient")); }

  /** Get the mask output */
  GradientImageType *GetMovingDomainMaskGradientOutput()
    { return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput("moving_mask_gradient")); }


  /** Get the number of components in the output */
  int GetNumberOfOutputComponents();


protected:
  MultiImageNCCPrecomputeFilter();
  ~MultiImageNCCPrecomputeFilter() {}

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for
   * ThreadedGenerateData(). */
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

  /** Set up the output information */
  virtual void GenerateOutputInformation();


  virtual typename itk::DataObject::Pointer MakeOutput(const DataObjectIdentifierType &);

  void UpdateOutputs();
  void ToggleOutput(bool flag, const DataObjectIdentifierType &key);

private:
  MultiImageNCCPrecomputeFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // Whether the gradient of the NCC metric will be computed. Depending on this, the filter
  // will generate 5 components per pixel (x,y,xy,x2,y2) or a bunch more needed for the
  // gradient computation.
  bool m_ComputeGradient;
  bool m_ComputeMovingDomainMask;
};





template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
class MultiImageNCCPostcomputeFilter : public itk::ImageToImageFilter<TInputImage, TMetricImage>
{
public:

  /** Standard class typedefs. */
  typedef MultiImageNCCPostcomputeFilter                      Self;
  typedef itk::ImageToImageFilter<TInputImage,TMetricImage>   Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageNCCPostcomputeFilter, ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension );

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef TInputImage                                 InputImageType;
  typedef typename TInputImage::PixelType             InputPixelType;
  typedef typename TInputImage::InternalPixelType     InputComponentType;
  typedef TMetricImage                                MetricImageType;
  typedef typename MetricImageType::PixelType         MetricPixelType;
  typedef TGradientImage                              GradientImageType;
  typedef typename GradientImageType::PixelType       GradientPixelType;
  typedef TMaskImage                                  MaskImageType;
  typedef typename MaskImageType::PixelType           MaskPixelType;
  typedef typename InputImageType::IndexType          IndexType;
  typedef typename InputImageType::SizeType           SizeType;

  typedef typename Superclass::DataObjectPointerArraySizeType  DataObjectPointerArraySizeType;


  /** Weight vector */
  typedef vnl_vector<float>                           WeightVectorType;

  /** Is the gradient of the metric being computed */
  itkSetMacro(ComputeGradient, bool)
  itkBooleanMacro(ComputeGradient)

  /** Set the weight vector */
  itkSetMacro(Weights, WeightVectorType)
  itkGetConstMacro(Weights, WeightVectorType)

  /** Set the mask image */
  void SetMaskImage(MaskImageType *mask)
    { this->itk::ProcessObject::SetInput("mask", mask); }

  /** Get the metric image */
  MetricImageType *GetMetricOutput();

  /** Get the gradient image */
  GradientImageType *GetGradientOutput();

  /** Get the metric value */
  itkGetMacro(MetricValue, double)

protected:

  MultiImageNCCPostcomputeFilter();
  ~MultiImageNCCPostcomputeFilter() {}

  virtual void BeforeThreadedGenerateData();
  virtual void AfterThreadedGenerateData();

  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

  typename itk::DataObject::Pointer MakeOutput(DataObjectPointerArraySizeType idx);

  // Whether or not the gradient is computed
  bool m_ComputeGradient;

  // Weight vector
  WeightVectorType m_Weights;

  // Vector of accumulated data (difference, gradient of affine transform, etc)
  double                          m_MetricValue;
  std::vector<double>             m_MetricPerThread;

  // TODO: handle this better!
  typename MetricImageType::Pointer m_MetricImage;
};




#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiImageOpticalFlowImageFilter.txx"
#endif

#endif
