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
#ifndef __WarpFunctors_h_
#define __WarpFunctors_h_


template <class TInputImage, class TOutputImage, class TFunctor>
class UnaryPositionBasedFunctorImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:

  typedef UnaryPositionBasedFunctorImageFilter<TInputImage,TOutputImage,TFunctor> Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( UnaryPositionBasedFunctorImageFilter, itk::ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension );

  void SetFunctor(const TFunctor &f) { this->m_Functor = f; }

protected:
  UnaryPositionBasedFunctorImageFilter() {}
  ~UnaryPositionBasedFunctorImageFilter() {}

  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                    itk::ThreadIdType threadId) ITK_OVERRIDE 
  {
    typedef itk::ImageRegionConstIteratorWithIndex<TInputImage> InputIter;
    InputIter it_in(this->GetInput(), outputRegionForThread);

    typedef itk::ImageRegionIterator<TOutputImage> OutputIter;
    OutputIter it_out(this->GetOutput(), outputRegionForThread);

    for(; !it_out.IsAtEnd(); ++it_out, ++it_in)
      {
      it_out.Set(m_Functor(it_in.Get(), it_in.GetIndex()));
      }
  }

  TFunctor m_Functor;
};

template <class TWarpImage>
class VoxelToPhysicalWarpFunctor
{
public:
  typedef itk::ImageBase<TWarpImage::ImageDimension> ImageBaseType;
  typedef typename TWarpImage::PixelType VectorType;
  typedef itk::Index<TWarpImage::ImageDimension> IndexType;

  VectorType operator()(const VectorType &v, const IndexType &pos)
  {
    // Get the physical point for the tail of the arrow
    typedef itk::ContinuousIndex<double, TWarpImage::ImageDimension> CIType;
    typedef typename TWarpImage::PointType PtType;

    CIType ia, ib;
    PtType pa, pb;
    for(int i = 0; i < TWarpImage::ImageDimension; i++)
      {
      ia[i] = pos[i];
      ib[i] = pos[i] + v[i];
      }

    m_Warp->TransformContinuousIndexToPhysicalPoint(ia, pa);
    m_MovingSpace->TransformContinuousIndexToPhysicalPoint(ib, pb);

    VectorType y;
    for(int i = 0; i < TWarpImage::ImageDimension; i++)
      y[i] = pb[i] - pa[i];

    return y;
  }

  VoxelToPhysicalWarpFunctor(TWarpImage *warp, ImageBaseType *moving)
    : m_Warp(warp), m_MovingSpace(moving) {}

  VoxelToPhysicalWarpFunctor() {}

protected:

  TWarpImage *m_Warp;
  ImageBaseType *m_MovingSpace;
};


template <class TWarpImage>
class PhysicalToVoxelWarpFunctor
{
public:
  typedef itk::ImageBase<TWarpImage::ImageDimension> ImageBaseType;
  typedef typename TWarpImage::PixelType VectorType;
  typedef itk::Index<TWarpImage::ImageDimension> IndexType;

  VectorType operator()(const VectorType &v, const IndexType &pos)
  {
    // Get the voxel offset between the tip of the arrow and the input position
    // Get the physical point for the tail of the arrow
    typedef itk::ContinuousIndex<double, TWarpImage::ImageDimension> CIType;
    typedef typename TWarpImage::PointType PtType;

    CIType ia, ib;
    PtType pa, pb;

    // Get the base physical position
    for(int i = 0; i < TWarpImage::ImageDimension; i++)
      ia[i] = pos[i];
    m_Warp->TransformContinuousIndexToPhysicalPoint(ia, pa);

    // Compute the tip physical position
    for(int i = 0; i < TWarpImage::ImageDimension; i++)
      pb[i] = pa[i] + v[i];

    // Map the tip into continuous index
    m_MovingSpace->TransformPhysicalPointToContinuousIndex(pb, ib);

    VectorType y;
    for(int i = 0; i < TWarpImage::ImageDimension; i++)
      y[i] = ib[i] - ia[i];

    return y;
  }

  PhysicalToVoxelWarpFunctor(TWarpImage *warp, ImageBaseType *moving)
    : m_Warp(warp), m_MovingSpace(moving) {}

  PhysicalToVoxelWarpFunctor() {}

protected:

  TWarpImage *m_Warp;
  ImageBaseType *m_MovingSpace;
};


/**
 * This functor is used to compress a warp before saving it. The input
 * to this functor is a voxel-space warp, and the output is a physical
 * space warp, with the precision of the voxel-space warp reduced to a
 * prescribed value. The functor will also cast the warp to desired
 * output type
 */
template <class TInputWarp, class TOutputWarp>
class CompressWarpFunctor
{
public:
  typedef VoxelToPhysicalWarpFunctor<TInputWarp> PhysFunctor;
  typedef typename PhysFunctor::ImageBaseType ImageBaseType;

  typedef typename TInputWarp::IndexType IndexType;
  typedef typename TInputWarp::PixelType InputVectorType;
  typedef typename TOutputWarp::PixelType OutputVectorType;

  CompressWarpFunctor() {}

  CompressWarpFunctor(TInputWarp *input, ImageBaseType *mov_space, double precision)
    : m_InputWarp(input), m_Precision(precision), m_ScaleFactor(1.0 / m_Precision),
      m_PhysFunctor(input, mov_space) {}

  OutputVectorType operator()(const InputVectorType &v, const IndexType &pos)
  {
    InputVectorType w;

    // Round to precision
    if(m_Precision > 0)
      {
      for(int i = 0; i < TInputWarp::ImageDimension; i++)
        w[i] = std::floor(v[i] * m_ScaleFactor + 0.5) * m_Precision;
      }
    else
      {
      w = v;
      }

    // Map to physical space
    w = m_PhysFunctor(w, pos);

    // Cast to output type
    InputVectorType y;
    for(int i = 0; i < TInputWarp::ImageDimension; i++)
      y[i] = static_cast<typename OutputVectorType::ValueType>(w[i]);

    return y;
  }

protected:
  TInputWarp *m_InputWarp;
  double m_Precision, m_ScaleFactor;
  PhysFunctor m_PhysFunctor;
};




#endif
