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
#include "itkVectorImage.h"

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

  /** Set default (power of two) pyramid factors */
  void SetDefaultPyramidFactors(int n_levels);

  /** Set the pyramid factors - for multi-resolution (e.g., 8,4,2) */
  void SetPyramidFactors(const PyramidFactorsType &factors);

  /** Add a pair of multi-component images to the class - same weight for each component */
  void AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight);

  /** Compute the composite image - must be run before any sampling is done */
  void BuildCompositeImages();

  /** Get the reference image for level k */
  ImageBaseType *GetReferenceSpace(int level);

  /** Perform interpolation - compute [(I - J(Tx)) GradJ(Tx)] */
  double ComputeOpticalFlowField(int level, VectorImageType *def, VectorImageType *result,
                                 double result_scaling = 1.0);


protected:

  // Pyramid factors
  PyramidFactorsType m_PyramidFactors;

  // Weights
  std::vector<double> m_Weights;

  // Vector of images
  typedef std::vector<typename MultiComponentImageType::Pointer> MultiCompImageSet;

  // Fixed and moving images
  MultiCompImageSet m_Fixed, m_Moving;

  // Composite image at each resolution level
  MultiCompImageSet m_FixedComposite, m_MovingComposite;

  void PlaceIntoComposite(FloatImageType *src, MultiComponentImageType *target, int offset);
  void PlaceIntoComposite(VectorImageType *src, MultiComponentImageType *target, int offset);
};

namespace itk
{

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
template <class TInputImage, class TOutputImage, class TDeformationField = TOutputImage>
class ITK_EXPORT MultiImageOpticalFlowImageFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageOpticalFlowImageFilter             Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageOpticalFlowImageFilter, ImageToImageFilter )

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

  /** Weight vector */
  typedef vnl_vector<float>                           WeightVectorType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      OutputImageType::ImageDimension );

  /** typedef for base image type at the current ImageDimension */
  typedef ImageBase<itkGetStaticConstMacro(ImageDimension)> ImageBaseType;

  /** Deformation field typedef support. */
  typedef TDeformationField                        DeformationFieldType;
  typedef typename DeformationFieldType::Pointer   DeformationFieldPointer;
  typedef typename DeformationFieldType::PixelType DisplacementType;

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
  void SetDeformationField(DeformationFieldType *field)
    { this->ProcessObject::SetInput("deformation", field); }

  /** Set constant scaling factor for the deformation field */
  itkSetMacro(DeformationScaling, float)
  itkGetConstMacro(DeformationScaling, float)

  /** Get the total energy of optical flow - only after Update has been called */
  itkGetConstMacro(TotalEnergy, double)

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
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for 
   * ThreadedGenerateData(). */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            ThreadIdType threadId );

  void VerifyInputInformation() {}

  // Object to assist specializaiton
  struct DispatchBase {};
  template <unsigned int VDim> struct Dispatch : public DispatchBase {};

  /** Fast interpolation method */
  double OpticalFlowFastInterpolate(const Dispatch<3> &dispatch,
                                    float *cix,
                                    const InputComponentType *fixed_ptr,
                                    const InputComponentType *moving_ptr,
                                    OutputPixelType &outVector,
                                    int *movSize,
                                    int nComp,
                                    const InputComponentType *def_value);

  // Dummy implementation
  double OpticalFlowFastInterpolate(const DispatchBase &base,
                                    float *cix,
                                    const InputComponentType *fixed_ptr,
                                    const InputComponentType *moving_ptr,
                                    OutputPixelType &outVector,
                                    int *movSize,
                                    int nComp,
                                    const InputComponentType *def_value)
    { return 0.0; }

private:
  MultiImageOpticalFlowImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // Scaling for the deformation field
  float                     m_DeformationScaling;

  // Weight vector
  WeightVectorType          m_Weights;

  // Total energy - Sum |I_k - J_k|^2
  double                    m_TotalEnergy;
  std::vector<double>       m_TotalEnergyPerThread;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiImageSimpleWarpImageFilter.txx"
#endif

#endif
