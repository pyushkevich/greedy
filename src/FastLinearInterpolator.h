#ifndef __FastLinearInterpolator_h_
#define __FastLinearInterpolator_h_

#include "itkVectorImage.h"
#include "itkNumericTraits.h"

/**
 * Base class for the fast linear interpolators
 */
template<class TImage,
         class TFloat = typename itk::NumericTraits<typename TImage::InternalPixelType>::RealType>
class FastLinearInterpolatorBase
{
public:
  typedef TImage                                 ImageType;
  typedef TFloat                                 RealType;
  typedef typename ImageType::InternalPixelType  InputComponentType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, ImageType::ImageDimension );

  enum InOut { INSIDE, OUTSIDE, BORDER };

  FastLinearInterpolatorBase(ImageType *image)
  {
    buffer = image->GetBufferPointer();
    nComp = image->GetNumberOfComponentsPerPixel();
    def_value_store = vnl_vector<InputComponentType>(nComp, itk::NumericTraits<InputComponentType>::Zero);
    def_value = def_value_store.data_block();
  }

protected:


  int nComp;
  const InputComponentType *buffer;

  // Default value - for interpolation outside of the image bounds
  const InputComponentType *def_value;
  vnl_vector<InputComponentType> def_value_store;

  InOut status;


  inline RealType lerp(RealType a, RealType l, RealType h)
  {
    return l+((h-l)*a);
  }
};


/**
 * Arbitrary dimension fast linear interpolator - meant to be slow
 */
template<class TImage,
         class TFloat = typename itk::NumericTraits<typename TImage::InternalPixelType>::RealType>
class FastLinearInterpolator : public FastLinearInterpolatorBase<TImage, TFloat>
{
public:
  typedef FastLinearInterpolatorBase<TImage, TFloat>   Superclass;
  typedef typename Superclass::ImageType               ImageType;
  typedef typename Superclass::InputComponentType      InputComponentType;
  typedef typename Superclass::RealType                RealType;
  typedef typename Superclass::InOut                   InOut;

  FastLinearInterpolator(ImageType *image) : Superclass(image) {}

  InOut InterpolateWithGradient(RealType *cix, RealType *out, RealType **grad)
    { return Superclass::INSIDE; }

  InOut Interpolate(RealType *cix, RealType *out)
    { return Superclass::INSIDE; }

  TFloat GetMask() { return 0.0; }

  TFloat GetMaskAndGradient(RealType *mask_gradient) { return 0.0; }

  template <class THistContainer>
  void PartialVolumeHistogramSample(RealType *cix, const InputComponentType *fixptr, THistContainer &hist) {}

  template <class THistContainer>
  void PartialVolumeHistogramGradientSample(RealType *cix, const InputComponentType *fix_ptr, const THistContainer &hist_w, RealType *out_grad) {}


protected:
};

/**
 * 3D fast linear interpolator - optimized for speed
 */
template <class TPixel, class TFloat>
class FastLinearInterpolator<itk::VectorImage<TPixel, 3>, TFloat>
    : public FastLinearInterpolatorBase<itk::VectorImage<TPixel, 3>, TFloat>
{
public:
  typedef itk::VectorImage<TPixel, 3>                   ImageType;
  typedef FastLinearInterpolatorBase<ImageType, TFloat> Superclass;
  typedef typename Superclass::InputComponentType       InputComponentType;
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::InOut                    InOut;

  FastLinearInterpolator(ImageType *image) : Superclass(image)
  {
    xsize = image->GetLargestPossibleRegion().GetSize()[0];
    ysize = image->GetLargestPossibleRegion().GetSize()[1];
    zsize = image->GetLargestPossibleRegion().GetSize()[2];
  }

  /**
   * Compute the pointers to the eight corners of the interpolating cube
   */
  InOut ComputeCorners(RealType *cix)
  {
    const InputComponentType *dp;

    x0 = floor(cix[0]); fx = cix[0] - x0;
    y0 = floor(cix[1]); fy = cix[1] - y0;
    z0 = floor(cix[2]); fz = cix[2] - z0;

    x1 = x0 + 1;
    y1 = y0 + 1;
    z1 = z0 + 1;

    if (x0 >= 0 && x1 < xsize &&
        y0 >= 0 && y1 < ysize &&
        z0 >= 0 && z1 < zsize)
      {
      // The sample point is completely inside
      dp = dens(x0, y0, z0);
      d000 = dp;
      d100 = dp+this->nComp;
      dp += xsize*this->nComp;
      d010 = dp;
      d110 = dp+this->nComp;
      dp += xsize*ysize*this->nComp;
      d011 = dp;
      d111 = dp+this->nComp;
      dp -= xsize*this->nComp;
      d001 = dp;
      d101 = dp+this->nComp;

      // The mask is one
      this->status = Superclass::INSIDE;
      }
    else if (x0 >= -1 && x1 <= xsize &&
             y0 >= -1 && y1 <= ysize &&
             z0 >= -1 && z1 <= zsize)
      {
      // The sample point is on the border region
      d000 = border_check(x0, y0, z0, m000);
      d001 = border_check(x0, y0, z1, m001);
      d010 = border_check(x0, y1, z0, m010);
      d011 = border_check(x0, y1, z1, m011);
      d100 = border_check(x1, y0, z0, m100);
      d101 = border_check(x1, y0, z1, m101);
      d110 = border_check(x1, y1, z0, m110);
      d111 = border_check(x1, y1, z1, m111);

      // The mask is between 0 and 1
      this->status = Superclass::BORDER;
      }
    else
      {
      // The mask is zero
      this->status = Superclass::OUTSIDE;
      }

    return this->status;
  }

  /**
   * Interpolate at position cix, placing the intensity values in out and gradient
   * values in grad (in strides of VDim)
   */
  InOut InterpolateWithGradient(RealType *cix, RealType *out, RealType **grad)
  {
    RealType dx00, dx01, dx10, dx11, dxy0, dxy1;
    RealType dx00_x, dx01_x, dx10_x, dx11_x, dxy0_x, dxy1_x;
    RealType dxy0_y, dxy1_y;

    // Compute the corners
    this->ComputeCorners(cix);

    if(this->status != Superclass::OUTSIDE)
      {
      // Loop over the components
      for(int iComp = 0; iComp < this->nComp; iComp++, grad++,
          d000++, d001++, d010++, d011++,
          d100++, d101++, d110++, d111++)
        {
        // Interpolate the image intensity
        dx00 = Superclass::lerp(fx, *d000, *d100);
        dx01 = Superclass::lerp(fx, *d001, *d101);
        dx10 = Superclass::lerp(fx, *d010, *d110);
        dx11 = Superclass::lerp(fx, *d011, *d111);
        dxy0 = Superclass::lerp(fy, dx00, dx10);
        dxy1 = Superclass::lerp(fy, dx01, dx11);
        *(out++) = Superclass::lerp(fz, dxy0, dxy1);

        // Interpolate the gradient in x
        dx00_x = *d100 - *d000;
        dx01_x = *d101 - *d001;
        dx10_x = *d110 - *d010;
        dx11_x = *d111 - *d011;
        dxy0_x = this->lerp(fy, dx00_x, dx10_x);
        dxy1_x = this->lerp(fy, dx01_x, dx11_x);
        (*grad)[0] = this->lerp(fz, dxy0_x, dxy1_x);

        // Interpolate the gradient in y
        dxy0_y = dx10 - dx00;
        dxy1_y = dx11 - dx01;
        (*grad)[1] = this->lerp(fz, dxy0_y, dxy1_y);

        // Interpolate the gradient in z
        (*grad)[2] = dxy1 - dxy0;
        }
      }

    return this->status;
  }

  InOut Interpolate(RealType *cix, RealType *out)
  {
    RealType dx00, dx01, dx10, dx11, dxy0, dxy1;

    // Compute the corners
    this->ComputeCorners(cix);

    if(this->status != Superclass::OUTSIDE)
      {
      // Loop over the components
      for(int iComp = 0; iComp < this->nComp; iComp++,
          d000++, d001++, d010++, d011++,
          d100++, d101++, d110++, d111++)
        {
        // Interpolate the image intensity
        dx00 = Superclass::lerp(fx, *d000, *d100);
        dx01 = Superclass::lerp(fx, *d001, *d101);
        dx10 = Superclass::lerp(fx, *d010, *d110);
        dx11 = Superclass::lerp(fx, *d011, *d111);
        dxy0 = Superclass::lerp(fy, dx00, dx10);
        dxy1 = Superclass::lerp(fy, dx01, dx11);
        *(out++) = Superclass::lerp(fz, dxy0, dxy1);
        }
      }

    return this->status;
  }

  template <class THistContainer>
  void PartialVolumeHistogramSample(RealType *cix, const InputComponentType *fixptr, THistContainer &hist)
  {
    // Compute the corners
    this->ComputeCorners(cix);

    if(this->status != Superclass::OUTSIDE)
      {
      // Compute the corner weights using 4 multiplications (not 16)
      RealType fxy = fx * fy, fyz = fy * fz, fxz = fx * fz, fxyz = fxy * fz;

      double w111 = fxyz;
      double w011 = fyz - fxyz;
      double w101 = fxz - fxyz;
      double w110 = fxy - fxyz;
      double w001 = fz - fxz - w011;
      double w010 = fy - fyz - w110;
      double w100 = fx - fxy - w101;
      double w000 = 1.0 - fx - fy + fxy - w001;

      // Loop over the components
      for(int iComp = 0; iComp < this->nComp; iComp++,
          d000++, d001++, d010++, d011++,
          d100++, d101++, d110++, d111++, fixptr++)
        {
        // Just this line in the histogram
        RealType *hist_line = hist[iComp][*fixptr];

        // Assign the appropriate weight to each part of the histogram
        hist_line[*d000] += w000;
        hist_line[*d001] += w001;
        hist_line[*d010] += w010;
        hist_line[*d011] += w011;
        hist_line[*d100] += w100;
        hist_line[*d101] += w101;
        hist_line[*d110] += w110;
        hist_line[*d111] += w111;
        }
      }
    else
      {
      for(int iComp = 0; iComp < this->nComp; iComp++, fixptr++)
        {
        // Just this line in the histogram
        RealType *hist_line = hist[iComp][*fixptr];
        hist_line[0] += 1.0;
        }
      }
  }

  template <class THistContainer>
  void PartialVolumeHistogramGradientSample(RealType *cix, const InputComponentType *fixptr, const THistContainer &hist_w, RealType *out_grad)
  {
    // Compute the corners
    this->ComputeCorners(cix);

    if(this->status == Superclass::OUTSIDE)
      {
      d000 = d001 = d010 = d011 = this->def_value;
      d100 = d101 = d110 = d111 = this->def_value;
      }

    // Compute the corner weights using 4 multiplications (not 16)
    RealType fxy = fx * fy, fyz = fy * fz, fxz = fx * fz;

    // Some horrendous derivatives here! Wow!
    double w111x = fyz,             w111y = fxz,             w111z = fxy;
    double w011x = -fyz,            w011y = fz - fxz,        w011z = fy - fxy;
    double w101x = fz - fyz,        w101y = -fxz,            w101z = fx - fxy;
    double w110x = fy - fyz,        w110y = fx - fxz,        w110z = -fxy;
    double w001x = -fz - w011x,     w001y = -w011y,          w001z = 1 - fx - w011z;
    double w010x = -w110x,          w010y = 1 - fz - w110y,  w010z = -fy - w110z;
    double w100x = 1 - fy - w101x,  w100y = -fx - w101y,     w100z = -w101z;
    double w000x = -1 + fy - w001x, w000y = -1 + fx - w001y, w000z = -w001z;

    // Initialize gradient to zero
    out_grad[0] = 0.0;
    out_grad[1] = 0.0;
    out_grad[2] = 0.0;

    // Loop over the components
    for(int iComp = 0; iComp < this->nComp; iComp++,
        d000++, d001++, d010++, d011++,
        d100++, d101++, d110++, d111++, fixptr++)
      {
      // Just this line in the histogram
      RealType *f = hist_w[iComp][*fixptr];

      // Take the weighted sum
      RealType f000 = f[*d000], f001 = f[*d001], f010 = f[*d010], f011 = f[*d011];
      RealType f100 = f[*d100], f101 = f[*d101], f110 = f[*d110], f111 = f[*d111];

      out_grad[0] += w000x * f000 + w001x * f001 + w010x * f010 + w011x * f011 +
                     w100x * f100 + w101x * f101 + w110x * f110 + w111x * f111;

      out_grad[1] += w000y * f000 + w001y * f001 + w010y * f010 + w011y * f011 +
                     w100y * f100 + w101y * f101 + w110y * f110 + w111y * f111;

      out_grad[2] += w000z * f000 + w001z * f001 + w010z * f010 + w011z * f011 +
                     w100z * f100 + w101z * f101 + w110z * f110 + w111z * f111;
      }
  }

  RealType GetMask()
  {
    // Interpolate the mask
    double dx00, dx01, dx10, dx11, dxy0, dxy1;
    dx00 = this->lerp(fx, m000, m100);
    dx01 = this->lerp(fx, m001, m101);
    dx10 = this->lerp(fx, m010, m110);
    dx11 = this->lerp(fx, m011, m111);
    dxy0 = this->lerp(fy, dx00, dx10);
    dxy1 = this->lerp(fy, dx01, dx11);
    return this->lerp(fz, dxy0, dxy1);
  }

  RealType GetMaskAndGradient(RealType *mask_gradient)
  {
    // Interpolate the mask
    double dx00, dx01, dx10, dx11, dxy0, dxy1;
    dx00 = this->lerp(fx, m000, m100);
    dx01 = this->lerp(fx, m001, m101);
    dx10 = this->lerp(fx, m010, m110);
    dx11 = this->lerp(fx, m011, m111);
    dxy0 = this->lerp(fy, dx00, dx10);
    dxy1 = this->lerp(fy, dx01, dx11);
    double mask = this->lerp(fz, dxy0, dxy1);

    // Compute the gradient of the mask
    double dx00_x, dx01_x, dx10_x, dx11_x, dxy0_x, dxy1_x;
    dx00_x = m100 - m000;
    dx01_x = m101 - m001;
    dx10_x = m110 - m010;
    dx11_x = m111 - m011;
    dxy0_x = this->lerp(fy, dx00_x, dx10_x);
    dxy1_x = this->lerp(fy, dx01_x, dx11_x);
    mask_gradient[0] = this->lerp(fz, dxy0_x, dxy1_x);

    double dxy0_y, dxy1_y;
    dxy0_y = dx10 - dx00;
    dxy1_y = dx11 - dx01;
    mask_gradient[1] = this->lerp(fz, dxy0_y, dxy1_y);

    mask_gradient[2] = dxy1 - dxy0;

    return mask;
  }

protected:

  inline const InputComponentType *border_check(int X, int Y, int Z, RealType &mask)
  {
    if(X >= 0 && X < xsize && Y >= 0 && Y < ysize && Z >= 0 && Z < zsize)
      {
      mask = 1.0;
      return dens(X,Y,Z);
      }
    else
      {
      mask = 0.0;
      return this->def_value;
      }
   }

  inline const InputComponentType *dens(int X, int Y, int Z)
  {
    return this->buffer + this->nComp * (X+xsize*(Y+ysize*Z));
  }

  // Image size
  int xsize, ysize, zsize;

  // State of current interpolation
  const InputComponentType *d000, *d001, *d010, *d011, *d100, *d101, *d110, *d111;
  RealType m000, m001, m010, m011, m100, m101, m110, m111;

  RealType fx, fy, fz;
  int	 x0, y0, z0, x1, y1, z1;

};

#endif
