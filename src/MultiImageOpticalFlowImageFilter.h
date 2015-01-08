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

/*

template<class TInputImage, class TOutputImage, class TDeformationImage>
class MultiImageOpticalFlowWarpTraits
{
public:
  typedef TDeformationImage TransformType;

  typedef typename TInputImage::InternalPixelType InputPixelType;
  typedef typename TOutputImage::InternalPixelType OutputPixelType;

  itkStaticConstMacro(ImageDimension, unsigned int,
                      TDeformationImage::ImageDimension );

  static DataObject *AsDataObject(TransformType *t) { return t; }
  static ImageBase<ImageDimension> *AsImageBase(TransformType *t) { return t; }

  static int GetResultAccumSize(int) { return 1; }

  static int GetStride(int) { return 1; }

  static void TransformIndex(const itk::Index<ImageDimension> &pos,
                            TransformType *transform, long offset,
                            float *ptran)
    {
    typename TDeformationImage::InternalPixelType &def = transform->GetBufferPointer()[offset];
    for(int i = 0; i < ImageDimension; i++)
      ptran[i] = pos[i] + def[i];
    }

  static void PostInterpolate(
      const itk::Index<ImageDimension> &pos,
      const InputPixelType *pFix, const InputPixelType *pMov, int nComp,
      float *weight, float mask, double *summary, OutputPixelType &vOut)
  {
    for(int i = 0; i < ImageDimension; i++)
      vOut[i] = 0;

    const InputPixelType *pMovEnd = pMov + nComp;
    while(pMov < pMovEnd)
      {
      double del = (*pFix++) - *(pMov++);
      double delw = (*weight++) * del;
      for(int i = 0; i < ImageDimension; i++)
        vOut[i] += delw * *(pMov++);
      *summary += delw * del;
      }
  }
};

template<class TInputImage, class TOutputImage>
class MultiImageOpticalFlowAffineGradientTraits
{
public:

  typedef typename TInputImage::InternalPixelType InputPixelType;
  typedef typename TOutputImage::InternalPixelType OutputPixelType;

  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension );

  typedef MatrixOffsetTransformBase<double, ImageDimension, ImageDimension> TransformType;


  static DataObject *AsDataObject(TransformType *t) { return NULL; }
  static ImageBase<ImageDimension> *AsImageBase(TransformType *t) { return NULL; }

  static int GetResultAccumSize(int nComp) { return 1 + ImageDimension * (1 + ImageDimension); }

  static int GetStride(int) { return 1; }

  static void TransformIndex(const itk::Index<ImageDimension> &pos,
                            TransformType *transform, long offset,
                            float *ptran)
    {
    for(int i = 0; i < ImageDimension; i++)
      {
      ptran[i] = transform->GetOffset()[i];
      for(int j = 0; j < ImageDimension; j++)
        ptran[i] += transform->GetMatrix()(i,j) * pos[j];
      }
    }

  static void PostInterpolate(
      const itk::Index<ImageDimension> &pos,
      const InputPixelType *pFix, const InputPixelType *pMov, int nComp,
      float *weight, float mask, double *summary, OutputPixelType &vOut)
  {
    const InputPixelType *pMovEnd = pMov + nComp;
    for(int i = 0; i < ImageDimension; i++)
      vOut[i] = 0;

    if(mask == 1.0)
      {
      while(pMov < pMovEnd)
        {
        double del = (*pFix++) - *(pMov++);
        double delw = (*weight++) * del;
        for(int i = 0; i < ImageDimension; i++)
          vOut[i] += delw * *(pMov++);
        *summary += delw * del;
        }

      for(int i = 0; i < ImageDimension; i++)
        {
        *(++summary) += vOut[i];
        for(int j = 0; j < ImageDimension; j++)
          {
          *(++summary) += vOut[i] * pos[j];
          }
        }
      }
    else if(mask > 0.0)
      {
      while(pMov < pMovEnd)
        {
        double del = (*pFix++) - *(pMov++);
        double delw = (*weight++) * del;
        //for(int i = 0; i < ImageDimension; i++)
        //  vOut[i] += delw * ( *(pMov++) * mask + del *
        // *summary += delw * del;
        }

      for(int i = 0; i < ImageDimension; i++)
        {
        *(++summary) += vOut[i];
        for(int j = 0; j < ImageDimension; j++)
          {
          *(++summary) += vOut[i] * pos[j];
          }
        }
      }





  }
};




template<class TInputImage, class TOutputImage>
class MultiImageOpticalFlowAffineObjectiveTraits
{
public:
  typedef MultiImageOpticalFlowAffineGradientTraits<TInputImage,TOutputImage> SourceTraits;
  typedef typename SourceTraits::TransformType TransformType;

  typedef typename TInputImage::InternalPixelType InputPixelType;
  typedef typename TOutputImage::InternalPixelType OutputPixelType;

  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension );

  static DataObject *AsDataObject(TransformType *t) { return NULL; }
  static ImageBase<ImageDimension> *AsImageBase(TransformType *t) { return NULL; }

  static int GetResultAccumSize(int) { return 2; }

  static int GetStride(int) { return 1 + ImageDimension; }

  static void TransformIndex(const itk::Index<ImageDimension> &pos,
                            TransformType *transform, long offset,
                            float *ptran)
    {
    return SourceTraits::TransformIndex(pos, transform, offset, ptran);
    }

  static void PostInterpolate(
      const itk::Index<ImageDimension> &pos,
      const InputPixelType *pFix, const InputPixelType *pMov, int nComp,
      float *weight, float mask, double *summary, OutputPixelType &vOut)
  {
    double wdiff = 0.0;

    if(mask > 0.0)
      {
      for(int i = 0; i < nComp; i+=(1+ImageDimension))
        {
        double del = (*pFix++) - *(pMov++);
        double delw = (*weight++) * del;
        wdiff += delw * del;
        }

      summary[0] += wdiff * mask;
      summary[1] += mask;
      }
  }
};

*/


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
    public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageOpticalFlowImageFilter                  Self;
  typedef itk::ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageOpticalFlowImageFilter, ImageToImageFilter )

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
    { this->ProcessObject::SetInput("Primary", fixed); }

  /** Set the moving image(s) and their gradients */
  void SetMovingImageAndGradient(InputImageType *moving)
    { this->ProcessObject::SetInput("moving", moving); }

  /** Set the weight vector */
  itkSetMacro(Weights, WeightVectorType)
  itkGetConstMacro(Weights, WeightVectorType)

  /** Set the deformation field. */
  void SetDeformationField(DeformationFieldType *phi)
    {
    m_Deformation = phi;
    this->ProcessObject::SetInput("phi", m_Deformation);
    }

  /** Summary results after running the filter */
  itkGetConstMacro(MetricValue, double)

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

  /** This method is used to set the state of the filter before
   * multi-threading. */
  virtual void BeforeThreadedGenerateData();

  /** This method is used to set the state of the filter after
   * multi-threading. */
  virtual void AfterThreadedGenerateData();

protected:
  MultiImageOpticalFlowImageFilter();
  ~MultiImageOpticalFlowImageFilter() {}
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for
   * ThreadedGenerateData(). */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

  void VerifyInputInformation() {}

private:
  MultiImageOpticalFlowImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // Weight vector
  WeightVectorType                m_Weights;

  // Transform pointer
  DeformationFieldPointer         m_Deformation;

  // Vector of accumulated data (difference, gradient of affine transform, etc)
  double                          m_MetricValue;
  std::vector<double>             m_MetricPerThread;
};



#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiImageOpticalFlowImageFilter.txx"
#endif

#endif
