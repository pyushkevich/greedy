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
#ifndef __MultiImageSimpleWarpImageFilter_h
#define __MultiImageSimpleWarpImageFilter_h
#include "itkImageBase.h"
#include "itkImageFunction.h"
#include "itkImageToImageFilter.h"
#include "itkPoint.h"
#include "itkFixedArray.h"

namespace itk
{

/** \class MultiImageOpticalFlowImageFilter
 * \brief Warps an image using an input deformation field (for LDDMM)
 *
 * SimpleWarpImageFilter warps an existing image with respect to
 * a given deformation field.
 *
 * A deformation field is represented as a image whose pixel type is some
 * vector type with at least N elements, where N is the dimension of
 * the input image. The vector type must support element access via operator
 * [].
 *
 * The output image is produced by inverse mapping: the output pixels
 * are mapped back onto the input image. This scheme avoids the creation of
 * any holes and overlaps in the output image.
 *
 * Each vector in the deformation field represent the distance between
 * a geometric point in the input space and a point in the output space 
 * in VOXEL COORDINATES (why? because it's faster!)
 *
 * \f[ p_{in} = p_{out} + d \f]
 *
 * Linear interpolation is used
 *
 * Position mapped to outside of the input image buffer are assigned
 * a edge padding value.
 *
 * This class is templated over the type of the input image, the
 * type of the output image and the type of the deformation field.
 *
 * The input image is set via SetInput. The input deformation field
 * is set via SetDeformationField.
 *
 * This filter is implemented as a multithreaded filter.
 *
 * \warning This filter assumes that the input type, output type
 * and deformation field type all have the same number of dimensions.
 *
 */
template <
  class TImage,
  class TVectorImage,
  class TFloat
  >
class ITK_EXPORT MultiImageOpticalFlowImageFilter :
    public ImageToImageFilter<TImage, TImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageOpticalFlowImageFilter             Self;
  typedef ImageToImageFilter<TImage,TImage>            Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageOpticalFlowImageFilter, ImageToImageFilter );

  /** Typedef to describe the output image region type. */
  typedef typename TImage::RegionType OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename TImage                             InputImageType;
  typedef typename TVectorImage                       InputVectorImageType;
  typedef typename TVectorImage                       OutputImageType;
  typedef typename OutputImageType::IndexType         IndexType;
  typedef typename OutputImageType::IndexValueType    IndexValueType;
  typedef typename OutputImageType::SizeType          SizeType;
  typedef typename OutputImageType::PixelType         PixelType;
  typedef typename OutputImageType::SpacingType       SpacingType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      OutputImageType::ImageDimension );
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      InputImageType::ImageDimension );
  itkStaticConstMacro(DeformationFieldDimension, unsigned int,
                      InputVectorImageType::ImageDimension );
  /** typedef for base image type at the current ImageDimension */
  typedef ImageBase<itkGetStaticConstMacro(ImageDimension)> ImageBaseType;

  /** Deformation field typedef support. */
  typedef TVectorImage                             DeformationFieldType;
  typedef typename DeformationFieldType::Pointer   DeformationFieldPointer;
  typedef typename DeformationFieldType::PixelType DisplacementType;

  /** Type for representing the direction of the output image */
  typedef typename OutputImageType::DirectionType     DirectionType;

  /** Set the deformation field. */
  void SetDeformationField( const DeformationFieldType * field );

  /** Get a pointer the deformation field. */
  DeformationFieldType * GetDeformationField(void);

  /** Add a fixed / moving / gradient image pair */
  void AddImagePair(
          InputImageType *fixed,
          InputImageType *moving,
          InputVectorImageType *gradMoving,
          double weight);

  /** Set the edge padding value */
  itkSetMacro( EdgePaddingValue, PixelType );

  /** Get the edge padding value */
  itkGetConstMacro( EdgePaddingValue, PixelType );

  /** Set scaling factor for the deformation field */
  itkSetMacro(DeformationScaling, TFloat);
  itkGetMacro(DeformationScaling, TFloat);

  /** SimpleWarpImageFilter produces an image which is a different
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

  /** This method is used to set the state of the filter before 
   * multi-threading. */
  virtual void BeforeThreadedGenerateData();

  /** This method is used to set the state of the filter after 
   * multi-threading. */
  virtual void AfterThreadedGenerateData();

protected:
  MultiImageOpticalFlowImageFilter();
  ~MultiImageOpticalFlowImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for 
   * ThreadedGenerateData(). */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            ThreadIdType threadId );

private:
  MultiImageSimpleWarpImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  PixelType                  m_EdgePaddingValue;

  // Scaling for the deformation field
  TFloat m_DeformationScaling;

  // Interpolator
  typename InterpolatorType::Pointer m_Interpolator;

  struct ImageSet {
      InputImageType *fixed, *moving;
      InputVectorImageType *grad_moving;
      double weight;
  };

  std::vector<ImageSet> m_ImageSet;
  InputVectorImageType *m_Deformation;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "SimpleWarpImageFilter.txx"
#endif

#endif
