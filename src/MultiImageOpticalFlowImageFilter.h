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
#ifndef __MultiImageOpticalFlowImageFilter_h
#define __MultiImageOpticalFlowImageFilter_h
#include "itkImageBase.h"
#include "itkImageToImageFilter.h"
#include "itkPoint.h"
#include "itkFixedArray.h"
#include "itkVectorImage.h"



/** \class MultiImageOpticalFlowImageFilterBase
 * \brief Warps an image using an input deformation field (for LDDMM)
 *
 * Base class for filters that take a pair of vector images and a
 * deformation field and produce an output image. Can be used to compute
 * the mean squared difference metric and gradient or to compute the
 * components necessary to compute the normalized cross-correlation metric
 * and gradient
 *
 */
template <class TInputImage, class TOutputImage, class TDeformationField>
class ITK_EXPORT MultiImageOpticalFlowImageFilterBase :
    public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageOpticalFlowImageFilterBase              Self;
  typedef itk::ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageOpticalFlowImageFilterBase, ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension );

  /** Typedef to describe the output image region type. */
  typedef typename TInputImage::RegionType OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef TInputImage                                 InputImageType;
  typedef typename TInputImage::PixelType             InputPixelType;
  typedef typename TInputImage::InternalPixelType     InputComponentType;
  typedef TOutputImage                                OutputImageType;
  typedef typename OutputImageType::PixelType         OutputPixelType;
  typedef typename OutputPixelType::ComponentType     OutputComponentType;
  typedef typename OutputImageType::IndexType         IndexType;
  typedef typename OutputImageType::IndexValueType    IndexValueType;
  typedef typename OutputImageType::SizeType          SizeType;
  typedef typename OutputImageType::SpacingType       SpacingType;
  typedef typename OutputImageType::DirectionType     DirectionType;
  typedef itk::ImageBase<ImageDimension>              ImageBaseType;

  /** Information from the deformation field class */
  typedef TDeformationField                           DeformationFieldType;
  typedef typename DeformationFieldType::Pointer      DeformationFieldPointer;
  typedef typename DeformationFieldType::PixelType    DeformationVectorType;

  /** Weight vector */
  typedef vnl_vector<float>                           WeightVectorType;

  /** Set the fixed image(s) */
  void SetFixedImage(InputImageType *fixed)
    { this->itk::ProcessObject::SetInput("Primary", fixed); }

  /** Set the moving image(s) and their gradients */
  void SetMovingImage(InputImageType *moving)
    { this->itk::ProcessObject::SetInput("moving", moving); }

  /** Set the weight vector */
  itkSetMacro(Weights, WeightVectorType)
  itkGetConstMacro(Weights, WeightVectorType)

  /** Set the deformation field. */
  void SetDeformationField(DeformationFieldType *phi)
    {
    m_Deformation = phi;
    this->itk::ProcessObject::SetInput("phi", m_Deformation);
    }

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

protected:
  MultiImageOpticalFlowImageFilterBase() {}
  ~MultiImageOpticalFlowImageFilterBase() {}

  void VerifyInputInformation() {}

  // Weight vector
  WeightVectorType                m_Weights;

  // Transform pointer
  DeformationFieldPointer         m_Deformation;


private:
  MultiImageOpticalFlowImageFilterBase(const Self&); //purposely not implemented
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
template <class TInputImage, class TOutputImage, class TDeformationField>
class ITK_EXPORT MultiImageOpticalFlowImageFilter :
    public MultiImageOpticalFlowImageFilterBase<TInputImage, TOutputImage, TDeformationField>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageOpticalFlowImageFilter                  Self;
  typedef MultiImageOpticalFlowImageFilterBase<TInputImage,TOutputImage,TDeformationField>
                                                            Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageOpticalFlowImageFilter, MultiImageOpticalFlowImageFilterBase )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension );

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename Superclass::InputPixelType                InputPixelType;
  typedef typename Superclass::InputComponentType            InputComponentType;
  typedef typename Superclass::OutputImageType               OutputImageType;
  typedef typename Superclass::OutputPixelType               OutputPixelType;
  typedef typename Superclass::OutputComponentType           OutputComponentType;
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
template <class TInputImage, class TOutputImage, class TDeformationField>
class ITK_EXPORT MultiImageNCCPrecomputeFilter :
    public MultiImageOpticalFlowImageFilterBase<TInputImage, TOutputImage, TDeformationField>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageNCCPrecomputeFilter                     Self;
  typedef MultiImageOpticalFlowImageFilterBase<TInputImage,TOutputImage,TDeformationField>
                                                            Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageNCCPrecomputeFilter, MultiImageOpticalFlowImageFilterBase )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension );

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename Superclass::InputPixelType                InputPixelType;
  typedef typename Superclass::InputComponentType            InputComponentType;
  typedef typename Superclass::OutputImageType               OutputImageType;
  typedef typename Superclass::OutputPixelType               OutputPixelType;
  typedef typename Superclass::OutputComponentType           OutputComponentType;
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

protected:
  MultiImageNCCPrecomputeFilter() {}
  ~MultiImageNCCPrecomputeFilter() {}

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for
   * ThreadedGenerateData(). */
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

  /** Set up the output information */
  virtual void GenerateOutputInformation();

private:
  MultiImageNCCPrecomputeFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};





template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
class MultiImageNCCPostcomputeFilter : public itk::ImageToImageFilter<TInputImage, TGradientImage>
{
public:

  /** Standard class typedefs. */
  typedef MultiImageNCCPostcomputeFilter                      Self;
  typedef itk::ImageToImageFilter<TInputImage,TGradientImage> Superclass;
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

  /** Weight vector */
  typedef vnl_vector<float>                           WeightVectorType;

  /** Set the weight vector */
  itkSetMacro(Weights, WeightVectorType)
  itkGetConstMacro(Weights, WeightVectorType)

  /** Set the mask image */
  void SetMaskImage(MaskImageType *mask)
    { this->itk::ProcessObject::SetInput("mask", mask); }

  /** Get the metric image */
  itkGetObjectMacro(MetricImage, MetricImageType)

  /** Get the metric value */
  itkGetMacro(MetricValue, double)

protected:

  MultiImageNCCPostcomputeFilter() {}
  ~MultiImageNCCPostcomputeFilter() {}

  /** This method is used to set the state of the filter before
   * multi-threading. */
  virtual void BeforeThreadedGenerateData();

  /** This method is used to set the state of the filter after
   * multi-threading. */
  virtual void AfterThreadedGenerateData();

  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

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
