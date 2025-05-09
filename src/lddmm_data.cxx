/*=========================================================================

Program:   ALFABIS fast medical image registration programs
Language:  C++
Website:   github.com/pyushkevich/greedy
Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

This program is part of ALFABIS: Adaptive Large-Scale Framework for
Automatic Biomedical Image Segmentation.

ALFABIS development is funded by the NIH grant R01 EB017255.

ALFABIS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as publishGed by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ALFABIS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#include "lddmm_data.h"
#include "itkImageRegionIterator.h"
#include "itkNumericTraitsCovariantVectorPixel.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkAddImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkBinaryFunctorImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVectorImage.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkMinimumMaximumImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkComposeImageFilter.h"
#include "itkMinimumMaximumImageFilter.h"
#include "itkTernaryFunctorImageFilter.h"
#include "itkShiftScaleImageFilter.h"
#include "itkImageDuplicator.h"

#include "FastWarpCompositeImageFilter.h"
#include <mutex>

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::new_vf(VelocityField &vf, uint nt, ImageBaseType *ref)
{
  vf.resize(nt);
  for(uint i = 0; i < nt; i++)
    vf[i] = new_vimg(ref);
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::alloc_vimg(VectorImageType *img, ImageBaseType *ref, TFloat fill_value)
{
  img->SetRegions(ref->GetBufferedRegion());
  img->CopyInformation(ref);
  img->Allocate();
  img->FillBuffer(Vec(fill_value));
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::VectorImagePointer
    LDDMMData<TFloat, VDim>
    ::new_vimg(ImageBaseType *ref, TFloat fill_value)
{
  VectorImagePointer p = VectorImageType::New();
  alloc_vimg(p, ref, fill_value);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::alloc_mimg(MatrixImageType *img, ImageBaseType *ref)
{
  img->SetRegions(ref->GetBufferedRegion());
  img->CopyInformation(ref);
  img->Allocate();
  img->FillBuffer(Mat());
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::MatrixImagePointer
    LDDMMData<TFloat, VDim>
    ::new_mimg(ImageBaseType *ref)
{
  MatrixImagePointer p = MatrixImageType::New();
  alloc_mimg(p, ref);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::alloc_cimg(CompositeImageType *img, ImageBaseType *ref, int n_comp, TFloat fill_value)
{
  img->SetRegions(ref->GetBufferedRegion());
  img->CopyInformation(ref);
  img->SetNumberOfComponentsPerPixel(n_comp);
  img->Allocate();

  typename CompositeImageType::PixelType cpix;
  cpix.SetSize(n_comp);
  cpix.Fill(fill_value);
  img->FillBuffer(cpix);
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::new_cimg(ImageBaseType *ref, int n_comp, TFloat fill_value)
{
  CompositeImagePointer p = CompositeImageType::New();
  alloc_cimg(p, ref, n_comp, fill_value);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::alloc_img(ImageType *img, ImageBaseType *ref, TFloat fill_value)
{
  img->SetRegions(ref->GetBufferedRegion());
  img->CopyInformation(ref);
  img->Allocate();
  img->FillBuffer(fill_value);
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::ImagePointer
    LDDMMData<TFloat, VDim>
    ::new_img(ImageBaseType *ref, TFloat fill_value)
{
  ImagePointer p = ImageType::New();
  alloc_img(p, ref, fill_value);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::init(LDDMMData<TFloat, VDim> &p,
        ImageType *fix, ImageType *mov,
        uint nt, double alpha, double gamma, double sigma)
{
  p.fix = fix;
  p.mov = mov;
  p.alpha = alpha;
  p.sigma = sigma;
  p.gamma = gamma;
  p.nt = nt;
  p.dt = 1.0 / (nt - 1.0);
  p.sigma_sq = sigma * sigma;

  // Initialize N and R
  p.r = fix->GetBufferedRegion();
  p.nv = fix->GetBufferedRegion().GetNumberOfPixels();
  for(uint i = 0; i < VDim; i++)
    p.n[i] = p.r.GetSize()[i];

  // Initialize the velocity fields
  new_vf(p.v, nt, fix);
  new_vf(p.a, nt, fix);
  new_vf(p.f, nt, fix);

  // Initialize kernel terms
  p.f_kernel = new_img(fix);
  p.f_kernel_sq = new_img(fix);

  // Compute these images
  ImageIterator it(p.f_kernel, p.r), itsq(p.f_kernel_sq, p.r);
  for(; !it.IsAtEnd(); ++it, ++itsq)
    {
      TFloat val = 0.0;
      for(uint j = 0; j < VDim; j++)
        val += 1.0 - cos(it.GetIndex()[j] * 2.0 * vnl_math::pi / p.n[j]);
      it.Set(p.gamma + 2.0 * p.alpha * p.nv * val);
      itsq.Set(it.Get() * it.Get());
    }

  // Initialize temporary vector field
  p.vtmp = new_vimg(fix);
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::compute_navier_stokes_kernel(ImageType *kernel, double alpha, double gamma)
{
  ImageIterator it(kernel, kernel->GetBufferedRegion());
  itk::Size<VDim> sz = kernel->GetBufferedRegion().GetSize();
  double alpha_scale = 2.0 * alpha * kernel->GetBufferedRegion().GetNumberOfPixels();

  for(; !it.IsAtEnd(); ++it)
    {
      TFloat val = 0.0;
      for(uint j = 0; j < VDim; j++)
        val += 1.0 - cos(it.GetIndex()[j] * 2.0 * vnl_math::pi / sz[j]);
      double k = gamma + alpha_scale * val;
      it.Set(k * k);
    }
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::interp_vimg(VectorImageType *data, VectorImageType *field,
        TFloat def_scale, VectorImageType *out, bool use_nn, bool phys_space)
{
  typedef FastWarpCompositeImageFilter<VectorImageType, VectorImageType, VectorImageType> WF;
  typename WF::Pointer wf = WF::New();
  wf->SetDeformationField(field);
  wf->SetMovingImage(data);
  wf->GraftOutput(out);
  wf->SetDeformationScaling(def_scale);
  wf->SetUseNearestNeighbor(use_nn);
  wf->SetUsePhysicalSpace(phys_space);
  wf->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_exp(
        const VectorImageType *src, VectorImageType *trg, VectorImageType *work,
        int exponent, TFloat scale)
{
  // Scale the image if needed
  if(scale != 1.0)
    vimg_scale(src, scale, trg);
  else
    vimg_copy(src, trg);

  for(int q = 0; q < exponent; q++)
    {
      interp_vimg(trg, trg, 1.0, work);
      vimg_add_in_place(trg, work);
    }
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_exp_with_jacobian(
        const VectorImageType *src, VectorImageType *trg, VectorImageType *work,
        MatrixImageType *trg_jac, MatrixImageType *work_mat,
        int exponent, TFloat scale)
{
  // Scale the image if needed
  if(scale != 1.0)
    vimg_scale(src, scale, trg);
  else
    vimg_copy(src, trg);

  // Compute the initial Jacobian
  field_jacobian(trg, trg_jac);

  // Perform the exponentiation
  for(int q = 0; q < exponent; q++)
    {
      // Compute the composition of the Jacobian with itself, place in jac_work
      jacobian_of_composition(trg_jac, trg_jac, trg, work_mat);

      // Copy the data (TODO: this is a little wasteful)
      mimg_copy(work_mat, trg_jac);

      // Update the velocity field
      interp_vimg(trg, trg, 1.0, work);
      vimg_add_in_place(trg, work);
    }
}


template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::interp_mimg(MatrixImageType *data, VectorImageType *field,
        MatrixImageType *out, bool use_nn, bool phys_space)
{
  // Decorate the matrix images as multi-component images
  CompositeImagePointer wrap_data = CompositeImageType::New();
  wrap_data->SetRegions(data->GetBufferedRegion());
  wrap_data->CopyInformation(data);
  wrap_data->SetNumberOfComponentsPerPixel(VDim * VDim);
  wrap_data->GetPixelContainer()->SetImportPointer(
      (TFloat *)(data->GetPixelContainer()->GetImportPointer()),
      VDim * VDim * data->GetPixelContainer()->Size(), false);

  // Decorate the output image in the same way
  CompositeImagePointer wrap_out = CompositeImageType::New();
  wrap_out->SetRegions(out->GetBufferedRegion());
  wrap_out->CopyInformation(out);
  wrap_out->SetNumberOfComponentsPerPixel(VDim * VDim);
  wrap_out->GetPixelContainer()->SetImportPointer(
      (TFloat *)(out->GetPixelContainer()->GetImportPointer()),
      VDim * VDim * out->GetPixelContainer()->Size(), false);

  // Perform the interpolation
  LDDMMData<TFloat, VDim>::interp_cimg(wrap_data, field, wrap_out, use_nn, phys_space);
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::interp_img(ImageType *data, VectorImageType *field, ImageType *out,
        bool use_nn, bool phys_space, TFloat outside_value)
{
  typedef FastWarpCompositeImageFilter<ImageType, ImageType, VectorImageType> WF;
  typename WF::Pointer wf = WF::New();
  wf->SetDeformationField(field);
  wf->SetMovingImage(data);
  wf->GraftOutput(out);
  wf->SetUseNearestNeighbor(use_nn);
  wf->SetUsePhysicalSpace(phys_space);
  wf->SetOutsideValue(outside_value);
  wf->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::interp_cimg(CompositeImageType *data, VectorImageType *field, CompositeImageType *out,
        bool use_nn, bool phys_space, TFloat outside_value)
{
  typedef FastWarpCompositeImageFilter<CompositeImageType, CompositeImageType, VectorImageType> WF;
  typename WF::Pointer wf = WF::New();
  wf->SetDeformationField(field);
  wf->SetMovingImage(data);
  wf->GraftOutput(out);
  wf->SetUseNearestNeighbor(use_nn);
  wf->SetUsePhysicalSpace(phys_space);
  wf->SetOutsideValue(outside_value);
  wf->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_add_in_place(VectorImageType *trg, VectorImageType *a)
{
  typedef itk::AddImageFilter<VectorImageType> AddFilter;
  typename AddFilter::Pointer flt = AddFilter::New();
  flt->SetInput(0,trg);
  flt->SetInput(1,a);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_subtract_in_place(VectorImageType *trg, VectorImageType *a)
{
  typedef itk::SubtractImageFilter<VectorImageType> SubtractFilter;
  typename SubtractFilter::Pointer flt = SubtractFilter::New();
  flt->SetInput(0,trg);
  flt->SetInput(1,a);
  flt->GraftOutput(trg);
  flt->Update();
}

// Scalar math
template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_multiply_in_place(VectorImageType *trg, ImageType *s)
{
  typedef itk::MultiplyImageFilter<
      VectorImageType, ImageType, VectorImageType> MultiplyFilter;
  typename MultiplyFilter::Pointer flt = MultiplyFilter::New();
  flt->SetInput1(trg);
  flt->SetInput2(s);
  flt->GraftOutput(trg);
  flt->Update();
}

template<class TFloat, uint VDim>
void LDDMMData<TFloat, VDim>::vimg_multiply_in_place(VectorImageType *trg, VectorImageType *s)
{
  typedef itk::MultiplyImageFilter<
      VectorImageType, VectorImageType, VectorImageType> MultiplyFilter;
  typename MultiplyFilter::Pointer flt = MultiplyFilter::New();
  flt->SetInput1(trg);
  flt->SetInput2(s);
  flt->GraftOutput(trg);
  flt->Update();
}

template<class TFloat, uint VDim>
double LDDMMData<TFloat, VDim>::vimg_dot_product(VectorImageType *a, VectorImageType *b)
{
  double value = 0.0;
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  std::mutex pooling_mutex;

  mt->ParallelizeImageRegion<VDim>(
      a->GetBufferedRegion(),
      [a, b, &value, &pooling_mutex](const itk::ImageRegion<VDim> &thread_region)
      {

        // Iterator typdef
        typedef itk::ImageLinearConstIteratorWithIndex<VectorImageType> IterBase;
        typedef IteratorExtender<IterBase> Iterator;

        unsigned int line_length = thread_region.GetSize(0);
        double thread_value = 0.0;
        for(Iterator it(a, thread_region); !it.IsAtEnd(); it.NextLine())
          {
            auto *a_line = it.GetPixelPointer(a), *b_line = it.GetPixelPointer(b);
            for(unsigned int i = 0; i < line_length; i++)
              for(unsigned int k = 0; k < VDim; k++)
                thread_value += a_line[i][k] * b_line[i][k];
          }

        // Use mutex to update the u range variable
        std::lock_guard<std::mutex> guard(pooling_mutex);
        value += thread_value;
      }, nullptr);

  return value;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_scale_in_place(ImageType *img, TFloat scale)
{
  typedef itk::ShiftScaleImageFilter<ImageType, ImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetScale(scale);
  filter->SetInput(img);
  filter->GraftOutput(img);
  filter->Update();
}


template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::mimg_multiply_in_place(MatrixImageType *trg, MatrixImageType *s)
{
  typedef itk::MultiplyImageFilter<
      MatrixImageType, MatrixImageType, MatrixImageType> MultiplyFilter;
  typename MultiplyFilter::Pointer flt = MultiplyFilter::New();
  flt->SetInput1(trg);
  flt->SetInput2(s);
  flt->GraftOutput(trg);
  flt->Update();
}

template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_multiply_in_place(CompositeImageType *trg, ImageType *s)
{
  typedef itk::MultiplyImageFilter<
      CompositeImageType, ImageType, CompositeImageType> MultiplyFilter;
  typename MultiplyFilter::Pointer flt = MultiplyFilter::New();
  flt->SetInput1(trg);
  flt->SetInput2(s);
  flt->GraftOutput(trg);
  flt->Update();
}

template<class TFloat, uint VDim>
void LDDMMData<TFloat, VDim>::cimg_mask_in_place(CompositeImageType *trg, ImageType *mask, TFloat background)
{
  itkAssertOrThrowMacro(trg->GetBufferedRegion() == mask->GetBufferedRegion(),
                         "Image and mask must be same size");

  // Create a fake region to partition the entire data chunk
  unsigned int nc = trg->GetNumberOfComponentsPerPixel();
  unsigned int np = trg->GetBufferedRegion().GetNumberOfPixels();
  itk::ImageRegion<1> full_region({{0}}, {{np}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  mt->ParallelizeImageRegion<1>(
      full_region,
      [trg,mask,background,nc](const itk::ImageRegion<1> &thread_region)
      {
        TFloat *p = trg->GetBufferPointer() + nc * thread_region.GetIndex(0);
        TFloat *m = mask->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *m_end = m + thread_region.GetSize(0);
        for(; m < m_end; ++m)
          {
            if(*m == 0.0)
              for(unsigned int i = 0; i < nc; i++)
                *p++ = background;
            else
              p+=nc;
          }
      }, nullptr);

  trg->Modified();
}

template<class TFloat, uint VDim>
double
    LDDMMData<TFloat, VDim>
    ::vimg_component_abs_max(VectorImageType *v)
{
  double value = 0.0;
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  std::mutex pooling_mutex;

  mt->ParallelizeImageRegion<VDim>(
      v->GetBufferedRegion(),
      [v, &value, &pooling_mutex](const itk::ImageRegion<VDim> &thread_region)
      {

        // Iterator typdef
        typedef itk::ImageLinearConstIteratorWithIndex<VectorImageType> IterBase;
        typedef IteratorExtender<IterBase> Iterator;

        unsigned int line_length = thread_region.GetSize(0);
        TFloat thread_value = 0.0;
        for(Iterator it(v, thread_region); !it.IsAtEnd(); it.NextLine())
          {
            auto *v_line = it.GetPixelPointer(v);
            for(unsigned int i = 0; i < line_length; i++)
              for(unsigned int k = 0; k < VDim; k++)
                thread_value = std::max(thread_value, std::fabs(v_line[i][k]));
          }

        // Use mutex to update the u range variable
        std::lock_guard<std::mutex> guard(pooling_mutex);
        value = std::max((double) thread_value, value);
      }, nullptr);

  return value;
}

template<class TFloat, uint VDim>
double
    LDDMMData<TFloat, VDim>
    ::vimg_component_abs_sum(VectorImageType *v)
{
  double value = 0.0;
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  std::mutex pooling_mutex;

  mt->ParallelizeImageRegion<VDim>(
      v->GetBufferedRegion(),
      [v, &value, &pooling_mutex](const itk::ImageRegion<VDim> &thread_region)
      {

        // Iterator typdef
        typedef itk::ImageLinearConstIteratorWithIndex<VectorImageType> IterBase;
        typedef IteratorExtender<IterBase> Iterator;

        unsigned int line_length = thread_region.GetSize(0);
        double thread_value = 0.0;
        for(Iterator it(v, thread_region); !it.IsAtEnd(); it.NextLine())
          {
            auto *v_line = it.GetPixelPointer(v);
            for(unsigned int i = 0; i < line_length; i++)
              for(unsigned int k = 0; k < VDim; k++)
                thread_value += std::fabs(v_line[i][k]);
          }

        // Use mutex to update the u range variable
        std::lock_guard<std::mutex> guard(pooling_mutex);
        value += thread_value;
      }, nullptr);

  return value;
}


// Scalar math

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_add_in_place(ImageType *trg, ImageType *a)
{
  typedef itk::AddImageFilter<ImageType> AddFilter;
  typename AddFilter::Pointer flt = AddFilter::New();
  flt->SetInput(0,trg);
  flt->SetInput(1,a);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_subtract_in_place(ImageType *trg, ImageType *a)
{
  typedef itk::SubtractImageFilter<ImageType> SubtractFilter;
  typename SubtractFilter::Pointer flt = SubtractFilter::New();
  flt->SetInput(0,trg);
  flt->SetInput(1,a);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_multiply_in_place(ImageType *trg, ImageType *a)
{
  typedef itk::MultiplyImageFilter<ImageType> MultiplyFilter;
  typename MultiplyFilter::Pointer flt = MultiplyFilter::New();
  flt->SetInput(0,trg);
  flt->SetInput(1,a);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
TFloat
    LDDMMData<TFloat, VDim>
    ::vimg_euclidean_norm_sq(VectorImageType *trg)
{
  // TODO: implement by calling dot product (faster code)
  // Add all voxels in the image
  double accum = 0.0;
  typedef itk::ImageRegionIterator<VectorImageType> Iter;
  for(Iter it(trg, trg->GetBufferedRegion()); !it.IsAtEnd(); ++it)
    {
      for(uint d = 0; d < VDim; d++)
        accum += it.Value()[d] * it.Value()[d];
    }
  return (TFloat) accum;
}

template <class TFloat, uint VDim>
TFloat
    LDDMMData<TFloat, VDim>
    ::img_euclidean_norm_sq(ImageType *trg)
{
  // Add all voxels in the image
  double accum = 0.0;
  typedef itk::ImageRegionIterator<ImageType> Iter;
  for(Iter it(trg, trg->GetBufferedRegion()); !it.IsAtEnd(); ++it)
    { accum += it.Value() * it.Value(); }
  return (TFloat) accum;
}


template <class TFloat, uint VDim>
class LinearToConstRectifierFunctor
{
public:
  typedef LinearToConstRectifierFunctor<TFloat, VDim> Self;

  LinearToConstRectifierFunctor() : m_Threshold(0.0), m_Offset(0.0) {}
  LinearToConstRectifierFunctor(TFloat thresh) : m_Threshold(thresh)
  { m_Offset = log(1 + exp(thresh)); }

  TFloat operator() (const TFloat &x)
  { return m_Offset - log(1 + exp(m_Threshold - x)); }

  bool operator== (const Self &other)
  { return m_Threshold == other.m_Threshold; }

  bool operator!= (const Self &other)
  { return m_Threshold != other.m_Threshold; }

protected:
  TFloat m_Threshold, m_Offset;
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_linear_to_const_rectifier_fn(ImageType *src, ImageType *trg, TFloat thresh)
{
  typedef LinearToConstRectifierFunctor<TFloat, VDim> Functor;
  typedef itk::UnaryFunctorImageFilter<ImageType, ImageType, Functor> Filter;
  typename Filter::Pointer flt = Filter::New();

  Functor func(thresh);
  flt->SetFunctor(func);
  flt->SetInput(src);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
class LinearToConstRectifierDerivFunctor
{
public:
  typedef LinearToConstRectifierDerivFunctor<TFloat, VDim> Self;

  LinearToConstRectifierDerivFunctor() : m_Threshold(0.0) {}
  LinearToConstRectifierDerivFunctor(TFloat thresh) : m_Threshold(thresh) {}

  TFloat operator() (const TFloat &x)
  { return 1.0 / (1.0 + exp(x - m_Threshold)); }

  bool operator== (const Self &other)
  { return m_Threshold == other.m_Threshold; }

  bool operator!= (const Self &other)
  { return m_Threshold != other.m_Threshold; }

protected:
  TFloat m_Threshold;
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_linear_to_const_rectifier_deriv(ImageType *src, ImageType *trg, TFloat thresh)
{
  typedef LinearToConstRectifierDerivFunctor<TFloat, VDim> Functor;
  typedef itk::UnaryFunctorImageFilter<ImageType, ImageType, Functor> Filter;
  typename Filter::Pointer flt = Filter::New();

  Functor func(thresh);
  flt->SetFunctor(func);
  flt->SetInput(src);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
TFloat
    LDDMMData<TFloat, VDim>
    ::img_voxel_sum(ImageType *trg)
{
  // Add all voxels in the image
  double accum = 0.0;
  typedef itk::ImageRegionIterator<ImageType> Iter;
  for(Iter it(trg, trg->GetBufferedRegion()); !it.IsAtEnd(); ++it)
    accum += it.Value();
  return (TFloat) accum;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_min_max(ImageType *src, TFloat &out_min, TFloat &out_max)
{
  // Add all voxels in the image
  typedef itk::MinimumMaximumImageFilter<ImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput(src);
  filter->Update();
  out_min = filter->GetMinimum();
  out_max = filter->GetMaximum();
}


template <class TFloat, uint VDim>
class VectorScaleFunctor
{
public:
  VectorScaleFunctor() { this->Scale = 1.0; }
  typedef itk::CovariantVector<TFloat,VDim> Vec;

  Vec operator() (const Vec &x)
  { return x * Scale; }

  bool operator== (const VectorScaleFunctor<TFloat, VDim> &other)
  { return Scale == other.Scale; }

  bool operator!= (const VectorScaleFunctor<TFloat, VDim> &other)
  { return Scale != other.Scale; }

  TFloat Scale;
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_scale_in_place(VectorImageType *trg, TFloat s)
{
  typedef VectorScaleFunctor<TFloat, VDim> Functor;
  typedef itk::UnaryFunctorImageFilter<
      VectorImageType, VectorImageType, Functor> Filter;
  typename Filter::Pointer flt = Filter::New();

  Functor func;
  func.Scale = s;
  flt->SetFunctor(func);
  flt->SetInput(trg);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_scale(const VectorImageType*src, TFloat s, VectorImageType *trg)
{
  typedef VectorScaleFunctor<TFloat, VDim> Functor;
  typedef itk::UnaryFunctorImageFilter<
      VectorImageType, VectorImageType, Functor> Filter;
  typename Filter::Pointer flt = Filter::New();

  Functor func;
  func.Scale = s;
  flt->SetFunctor(func);
  flt->SetInput(src);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
class VectorScaleAddFunctor
{
public:
  VectorScaleAddFunctor() { this->Scale = 1.0; }
  typedef itk::CovariantVector<TFloat,VDim> Vec;

  Vec operator() (const Vec &x, const Vec &y)
  { return x + y * Scale; }

  bool operator== (const VectorScaleAddFunctor<TFloat, VDim> &other)
  { return Scale == other.Scale; }

  bool operator!= (const VectorScaleAddFunctor<TFloat, VDim> &other)
  { return Scale != other.Scale; }

  TFloat Scale;
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_add_scaled_in_place(VectorImageType *trg, VectorImageType *a, TFloat s)
{
  typedef VectorScaleAddFunctor<TFloat, VDim> Functor;
  typedef itk::BinaryFunctorImageFilter<
      VectorImageType, VectorImageType, VectorImageType, Functor> Filter;
  typename Filter::Pointer flt = Filter::New();

  Functor func;
  func.Scale = s;
  flt->SetFunctor(func);
  flt->SetInput1(trg);
  flt->SetInput2(a);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
class VectorDotProduct
{
public:
  VectorDotProduct() {}
  typedef itk::CovariantVector<TFloat,VDim> Vec;

  TFloat operator() (const Vec &x, const Vec &y)
  {
    TFloat dp = 0.0;
    for(uint d = 0; d < VDim; d++)
      dp += x[d] * y[d];
    return dp;
  }

  bool operator== (const VectorDotProduct<TFloat, VDim> &)
  { return true; }

  bool operator!= (const VectorDotProduct<TFloat, VDim> &)
  { return false; }
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_euclidean_inner_product(ImagePointer &trg, VectorImageType *a, VectorImageType *b)
{
  typedef VectorDotProduct<TFloat, VDim> Functor;
  typedef itk::BinaryFunctorImageFilter<
      VectorImageType, VectorImageType, ImageType, Functor> Filter;
  typename Filter::Pointer flt = Filter::New();

  Functor func;
  flt->SetFunctor(func);
  flt->SetInput1(a);
  flt->SetInput2(b);
  flt->GraftOutput(trg);
  flt->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::compute_semi_lagrangean_a()
{
  for(uint i = 0; i < nt; i++)
    {
      a[i]->FillBuffer(Vec(0.0));
      for (uint j = 0; j < 5; j++)
        {
          interp_vimg(v[i], a[i], -0.5, a[i]);
          vimg_scale_in_place(a[i], dt);
          itk::Index<VDim> x;
          x[0] = 63; x[1] = 63;
        }
    }

}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::integrate_phi_t0()
{
  // Go through and compute phi_t0 for each time point
  for(int m = 0; m < (int) nt; m++)
    if(m==0)
      {
        f[m]->FillBuffer(Vec(0.0));
      }
    else
      {
        interp_vimg(f[m-1], a[m], -1.0, f[m]);
        vimg_subtract_in_place(f[m], a[m]);
      }
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::integrate_phi_t1()
{
  for(int m = nt-1; m >= 0; m--)
    {
      if(m == (int) nt-1)
        {
          f[m]->FillBuffer(Vec(0.0));
        }
      else
        {
          interp_vimg(f[m+1], a[m], 1.0, f[m]);
          vimg_add_in_place(f[m], a[m]);
        }
    }
}

template <class TFloat, uint VDim>
class SetMatrixRowBinaryOperator
{
public:
  typedef SetMatrixRowBinaryOperator<TFloat, VDim> Self;
  typedef LDDMMData<TFloat, VDim> LDDMM;
  typedef typename LDDMM::Vec Vec;
  typedef typename LDDMM::Mat Mat;

  SetMatrixRowBinaryOperator() { m_Row = 0; }
  void SetRow(unsigned int row) { m_Row = row; }
  bool operator != (const Self &other) const { return m_Row != other.m_Row; }

  Mat operator() (const Mat &M, const Vec &V)
  {
    Mat Q;
    for(unsigned int r = 0; r < VDim; r++)
      for(unsigned int c = 0; c < VDim; c++)
        Q(r, c) = (r == m_Row) ? V[c] : M(r,c);
    return Q;
  }

protected:
  unsigned int m_Row;
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::field_jacobian(VectorImageType *vec, MatrixImageType *out)
{
  for(unsigned int a = 0; a < VDim; a++)
    {
      // Extract the a'th component of the displacement field
      typedef itk::VectorIndexSelectionCastImageFilter<VectorImageType, ImageType> CompFilterType;
      typename CompFilterType::Pointer comp = CompFilterType::New();
      comp->SetIndex(a);
      comp->SetInput(vec);

      // Compute the gradient of this component
      typedef itk::GradientImageFilter<ImageType, TFloat, TFloat> GradientFilter;
      typename GradientFilter::Pointer grad = GradientFilter::New();
      grad->SetInput(comp->GetOutput());
      grad->SetUseImageSpacingOff();
      grad->SetUseImageDirection(false);

      // Apply to the Jacobian matrix
      typedef SetMatrixRowBinaryOperator<TFloat, VDim> RowOperatorType;
      RowOperatorType rop;
      rop.SetRow(a);
      typedef itk::BinaryFunctorImageFilter<
          MatrixImageType, VectorImageType, MatrixImageType, RowOperatorType> RowFilterType;
      typename RowFilterType::Pointer rof = RowFilterType::New();
      rof->SetInput1(out);
      rof->SetInput2(grad->GetOutput());
      rof->SetFunctor(rop);
      rof->GraftOutput(out);
      rof->Update();
    }
}

/**
 * Compute the divergence of a vector field.
 * TODO: this implementation is stupid, splits vector image into components and requires
 * a working image. Write a proper divergence filter!
 */
template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::field_divergence(VectorImageType *v, ImageType *div_v, bool use_spacing)
{
  // Initialize the divergence to zero
  div_v->FillBuffer(0.0);

  // Add each component
  for(unsigned int a = 0; a < VDim; a++)
    {
      // Extract the a'th component of the displacement field
      typedef itk::VectorIndexSelectionCastImageFilter<VectorImageType, ImageType> CompFilterType;
      typename CompFilterType::Pointer comp1 = CompFilterType::New();
      comp1->SetIndex(a);
      comp1->SetInput(v);

      // Compute the gradient of this component
      typedef itk::GradientImageFilter<ImageType, TFloat, TFloat> GradientFilter;
      typename GradientFilter::Pointer grad = GradientFilter::New();
      grad->SetInput(comp1->GetOutput());
      grad->SetUseImageSpacing(use_spacing);
      grad->SetUseImageDirection(false);

      // Extract the a'th component of the gradient field
      typedef itk::VectorIndexSelectionCastImageFilter<VectorImageType, ImageType> CompFilterType;
      typename CompFilterType::Pointer comp2 = CompFilterType::New();
      comp2->SetIndex(a);
      comp2->SetInput(grad->GetOutput());
      comp2->Update();

      // Add the result
      img_add_in_place(div_v, comp2->GetOutput());
    }
}

template <class TFloat, uint VDim>
class JacobianCompisitionFunctor
{
public:
  typedef typename LDDMMData<TFloat, VDim>::Mat Mat;

  Mat operator() (const Mat &Du_wrp, const Mat &Dv)
  {
    Mat Dw = Dv + Du_wrp * Dv + Du_wrp;
    return Dw;
  }

  bool operator != (const JacobianCompisitionFunctor<TFloat, VDim> &) { return false; }
};


template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::jacobian_of_composition(
        MatrixImageType *Du, MatrixImageType *Dv, VectorImageType *v, MatrixImageType *out_Dw)
{
  // Interpolate jac_phi by psi and place it into out
  interp_mimg(Du, v, out_Dw);

  // Perform the matrix multiplication and addition
  typedef JacobianCompisitionFunctor<TFloat, VDim> Functor;
  typedef itk::BinaryFunctorImageFilter<MatrixImageType,MatrixImageType,MatrixImageType,Functor> BinaryFilter;
  typename BinaryFilter::Pointer flt = BinaryFilter::New();
  flt->SetInput1(out_Dw);
  flt->SetInput2(Dv);
  flt->GraftOutput(out_Dw);
  flt->Update();
}



template <class TFloat, uint VDim>
class MatrixPlusConstDeterminantFunctor
{
public:
  typedef typename LDDMMData<TFloat, VDim>::Mat Mat;

  TFloat operator() (const Mat &M)
  {
    Mat X = m_LambdaEye;
    X += M;
    return vnl_determinant(X.GetVnlMatrix());
  }

  void SetLambda(TFloat lambda)
  {
    m_LambdaEye.SetIdentity();
    m_LambdaEye *= lambda;
  }

  bool operator != (const MatrixPlusConstDeterminantFunctor<TFloat, VDim> &other)
  { return m_LambdaEye(0,0) != other.m_LambdaEye(0,0); }

protected:
  Mat m_LambdaEye;

};


template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::mimg_det(MatrixImageType *M, double lambda, ImageType *out_det)
{
  typedef MatrixPlusConstDeterminantFunctor<TFloat, VDim> FunctorType;
  FunctorType functor;
  functor.SetLambda(lambda);
  typedef itk::UnaryFunctorImageFilter<MatrixImageType, ImageType, FunctorType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput(M);
  filter->SetFunctor(functor);
  filter->GraftOutput(out_det);
  filter->Update();
}

/**
 * Functor to compute Ax+b
 */
template <class TFloat, uint VDim>
class MatrixVectorMultiplyAndAddVectorFunctor
{
public:
  typedef MatrixVectorMultiplyAndAddVectorFunctor<TFloat, VDim> Self;
  typedef LDDMMData<TFloat, VDim> LDDMMType;
  typedef typename LDDMMType::Mat Mat;
  typedef typename LDDMMType::Vec Vec;

  Vec operator() (const Mat &A, const Vec &x, const Vec &b)
  {
    Vec y = m_Lambda * (A * x) + m_Mu * b;
    return y;
  }

  void SetLambda(TFloat lambda) { m_Lambda = lambda; }
  void SetMu(TFloat mu) { m_Mu = mu; }

  bool operator != (const Self &other) const
  { return m_Lambda != other.m_Lambda || m_Mu != other.m_Mu; }

  bool operator == (const Self &other) const
  { return ! (*this != other); }

protected:
  TFloat m_Lambda, m_Mu;
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::mimg_vimg_product_plus_vimg(
        MatrixImageType *A, VectorImageType *x, VectorImageType *b,
        TFloat lambda, TFloat mu, VectorImageType *out)
{
  typedef MatrixVectorMultiplyAndAddVectorFunctor<TFloat, VDim> Functor;
  Functor functor;
  functor.SetLambda(lambda);
  functor.SetMu(mu);

  typedef itk::TernaryFunctorImageFilter<
      MatrixImageType, VectorImageType, VectorImageType, VectorImageType,
      Functor> FilterType;

  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput1(A);
  filter->SetInput2(x);
  filter->SetInput3(b);
  filter->SetFunctor(functor);
  filter->GraftOutput(out);
  filter->Update();
}

#include "LieBracketFilter.h"

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::lie_bracket(VectorImageType *v, VectorImageType *u, MatrixImageType *work, VectorImageType *out)
{
  // Compute Du, place it in work
  field_jacobian(v, work);

  // Multiply by v
  mimg_vimg_product_plus_vimg(work, u, out, 1.0, 0.0, out);

  // Compute Dv, place it in work
  field_jacobian(u, work);

  // Multiply by u and subtract from existing
  mimg_vimg_product_plus_vimg(work, v, out, -1.0, 1.0, out);

  // Alternative approach
  VectorImagePointer alt = new_vimg(out);

  typedef LieBracketFilter<VectorImageType, VectorImageType> LieBracketFilterType;
  typename LieBracketFilterType::Pointer fltLieBracket = LieBracketFilterType::New();
  fltLieBracket->SetFieldU(v);
  fltLieBracket->SetFieldV(u);
  fltLieBracket->GraftOutput(alt);
  fltLieBracket->Update();
}


template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::field_jacobian_det(VectorImageType *vec, ImageType *out)
{
  typedef itk::DisplacementFieldJacobianDeterminantFilter<
      VectorImageType, TFloat, ImageType> Filter;
  typename Filter::Pointer filter = Filter::New();
  filter->SetInput(vec);
  filter->SetUseImageSpacingOff();
  filter->GraftOutput(out);
  filter->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::image_gradient(ImageType *src, VectorImageType *grad, bool use_spacing)
{
  // Create a gradient image filter
  typedef itk::GradientImageFilter<ImageType, TFloat, TFloat> Filter;
  typename Filter::Pointer flt = Filter::New();
  flt->SetInput(src);
  flt->GraftOutput(grad);
  flt->SetUseImageSpacing(use_spacing);
  flt->SetUseImageDirection(false);
  flt->Update();
}

template <class TImage>
void img_smooth_dim_inplace(TImage *img, unsigned int dim, double sigma)
{
  typedef itk::RecursiveGaussianImageFilter<TImage, TImage> Filter;
  typename Filter::Pointer fltSmooth = Filter::New();
  fltSmooth->SetInput(img);
  fltSmooth->SetOrder(itk::RecursiveGaussianImageFilterEnums::GaussianOrder::ZeroOrder);
  fltSmooth->SetDirection(dim);
  fltSmooth->SetSigma(sigma);
  fltSmooth->InPlaceOn();
  fltSmooth->Update();

  img->CopyInformation(fltSmooth->GetOutput());
  img->SetRegions(fltSmooth->GetOutput()->GetBufferedRegion());
  img->SetPixelContainer(fltSmooth->GetOutput()->GetPixelContainer());
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_smooth(ImageType *src, ImageType *trg, SmoothingSigmas sigma, SmoothingMode mode)
{
  // If the source and target are not the same, copy raw data from source to target
  if(src->GetPixelContainer() != trg->GetPixelContainer())
    {
      trg->CopyInformation(src);
      trg->SetRegions(src->GetBufferedRegion());
      img_copy(src, trg);
    }

  if(mode == ITK_RECURSIVE)
    {
      Vec sigma_phys = sigma.GetSigmaInWorldUnits(src);

      // Apply smoothing in each dimension
      for(unsigned int d = 0; d < VDim; d++)
        {
          if(sigma_phys[d] > 0.0)
            img_smooth_dim_inplace(trg, d, sigma_phys[d]);
        }
    }
  else
    {
      // Masquerade as a multi-component image
      CompositeImagePointer cimg = img_as_cimg(trg);
      cimg_smooth(cimg, cimg, sigma, mode);
    }
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_smooth(CompositeImageType *src, CompositeImageType *trg, SmoothingSigmas sigma, SmoothingMode mode)
{
  // If the source and target are not the same, copy from source to target
  if(src->GetPixelContainer() != trg->GetPixelContainer())
    {
      trg->CopyInformation(src);
      trg->SetRegions(src->GetBufferedRegion());
      cimg_copy(src, trg);
    }

  if(mode == ITK_RECURSIVE)
    {
      // Handle special case of one component (common, why waste time?)
      if(trg->GetNumberOfComponentsPerPixel() == 1)
        {
          ImagePointer img = cimg_as_img(trg);
          img_smooth(img, img, sigma, mode);
        }
      else
        {
          // Apply smoothing in each dimension
          Vec sigma_phys = sigma.GetSigmaInWorldUnits(src);
          for(unsigned int d = 0; d < VDim; d++)
            {
              if(sigma_phys[d] > 0.0)
                img_smooth_dim_inplace(trg, d, sigma_phys[d]);
            }
        }
    }
  else
    {
      cimg_fast_convolution_smooth_inplace(trg, sigma, mode);
    }
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_smooth(VectorImageType *src, VectorImageType *trg, SmoothingSigmas sigma, SmoothingMode mode)
{
  // If the source and target are not the same, copy from source to target
  if(src->GetPixelContainer() != trg->GetPixelContainer())
    {
      trg->CopyInformation(src);
      trg->SetRegions(src->GetBufferedRegion());
      vimg_copy(src, trg);
    }

  // Apply smoothing in each dimension
  if(mode == ITK_RECURSIVE)
    {
      Vec sigma_phys = sigma.GetSigmaInWorldUnits(src);
      for(unsigned int d = 0; d < VDim; d++)
        {
          if(sigma_phys[d] > 0.0)
            img_smooth_dim_inplace(trg, d, sigma_phys[d]);
        }
    }
  else
    {
      // Masquerade as a multi-component image
      CompositeImagePointer cimg = vimg_as_cimg(trg);
      cimg_smooth(cimg, cimg, sigma, mode);
    }
}

template <class TFloat, unsigned int VDim>
struct VectorSquareNormFunctor
{
  template <class TVector> TFloat operator() (const TVector &v)
  {
    TFloat norm_sqr = 0.0;
    for(unsigned int i = 0; i < VDim; i++)
      norm_sqr += v[i] * v[i];
    return norm_sqr;
  }
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_norm_min_max(VectorImageType *image, ImageType *normsqr,
        TFloat &min_norm, TFloat &max_norm)
{
  // Compute the squared norm of the displacement
  typedef VectorSquareNormFunctor<TFloat, VDim> NormFunctor;
  typedef itk::UnaryFunctorImageFilter<VectorImageType, ImageType, NormFunctor> NormFilter;
  typename NormFilter::Pointer fltNorm = NormFilter::New();
  fltNorm->SetInput(image);
  fltNorm->GraftOutput(normsqr);
  fltNorm->Update();

  // Compute the maximum squared norm of the displacement
  img_min_max(normsqr, min_norm, max_norm);

  min_norm = sqrt(min_norm);
  max_norm = sqrt(max_norm);
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_normalize_to_fixed_max_length(VectorImageType *trg, ImageType *normsqr,
        double max_displacement, bool scale_down_only)
{
  // Compute the squared norm of the displacement
  typedef VectorSquareNormFunctor<TFloat, VDim> NormFunctor;
  typedef itk::UnaryFunctorImageFilter<VectorImageType, ImageType, NormFunctor> NormFilter;
  typename NormFilter::Pointer fltNorm = NormFilter::New();
  fltNorm->SetInput(trg);
  fltNorm->GraftOutput(normsqr);
  fltNorm->Update();

  // Compute the maximum squared norm of the displacement
  TFloat nsq_min, nsq_max;
  img_min_max(normsqr, nsq_min, nsq_max);

  // Compute the scale functor
  TFloat scale = max_displacement / sqrt(nsq_max);

  // Apply the scale
  if(scale_down_only && scale >= 1.0)
    return;
  else
    vimg_scale_in_place(trg, scale);
}



namespace lddmm_data_io {

template <class TInputImage, class TOutputImage>
void
write_cast(TInputImage *image, const char *filename)
{
  typedef itk::CastImageFilter<TInputImage, TOutputImage> CastType;
  typename CastType::Pointer cast = CastType::New();
  cast->SetInput(image);

  typedef itk::ImageFileWriter<TOutputImage> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(cast->GetOutput());
  writer->SetFileName(filename);
  writer->SetUseCompression(true);
  writer->Update();
}



template <class TInputImage, class TOutputComponent>
struct image_type_cast
{
};

template <class TPixel, unsigned int VDim, class TOutputComponent>
struct image_type_cast< itk::Image<TPixel, VDim>, TOutputComponent>
{
  typedef itk::Image<TOutputComponent, VDim> OutputImageType;
};

template <class TPixel, unsigned int VDim, class TOutputComponent>
struct image_type_cast< itk::Image<itk::CovariantVector<TPixel, VDim>, VDim>, TOutputComponent>
{
  typedef itk::Image<itk::CovariantVector<TOutputComponent, VDim>, VDim> OutputImageType;
};

template <class TPixel, unsigned int VDim, class TOutputComponent>
struct image_type_cast< itk::VectorImage<TPixel, VDim>, TOutputComponent>
{
  typedef itk::VectorImage<TOutputComponent, VDim> OutputImageType;
};

template <class TInputImage>
void write_cast_to_iocomp(TInputImage *image, const char *filename,
    itk::IOComponentEnum comp)
{
  switch(comp)
  {
  case itk::IOComponentEnum::UCHAR :
    write_cast<TInputImage, typename image_type_cast<TInputImage, unsigned char>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::CHAR :
    write_cast<TInputImage, typename image_type_cast<TInputImage, char>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::USHORT :
    write_cast<TInputImage, typename image_type_cast<TInputImage, unsigned short>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::SHORT :
    write_cast<TInputImage, typename image_type_cast<TInputImage, short>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::UINT :
    write_cast<TInputImage, typename image_type_cast<TInputImage, unsigned int>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::INT :
    write_cast<TInputImage, typename image_type_cast<TInputImage, int>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::ULONG :
    write_cast<TInputImage, typename image_type_cast<TInputImage, unsigned long>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::LONG :
    write_cast<TInputImage, typename image_type_cast<TInputImage, long>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::FLOAT :
    write_cast<TInputImage, typename image_type_cast<TInputImage, float>::OutputImageType>(image, filename);
    break;
  case itk::IOComponentEnum::DOUBLE :
    write_cast<TInputImage, typename image_type_cast<TInputImage, double>::OutputImageType>(image, filename);
    break;
  default:
    typedef itk::ImageFileWriter<TInputImage> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetInput(image);
    writer->SetFileName(filename);
    writer->SetUseCompression(true);
    writer->Update();
  }
}

template <class TInputImage, class TOutputImage>
bool
try_auto_cast(const TInputImage *source, itk::Object *target)
{
  TOutputImage *output = dynamic_cast<TOutputImage *>(target);
  if(output)
    {
      output->CopyInformation(source);
      output->SetRegions(source->GetBufferedRegion());
      output->Allocate();
      itk::ImageAlgorithm::Copy(source, output, source->GetBufferedRegion(), output->GetBufferedRegion());
      return true;
    }
  return false;
}

template <class TInputImage>
bool auto_cast(const TInputImage *source, itk::Object *target)
{
  return try_auto_cast<TInputImage, typename image_type_cast<TInputImage, unsigned char>::OutputImageType>(source, target)
  || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, char>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, unsigned short>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, short>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, unsigned int>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, int>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, unsigned long>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, long>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, float>::OutputImageType>(source, target)
      || try_auto_cast<TInputImage, typename image_type_cast<TInputImage, double>::OutputImageType>(source, target);
}


} // namespace


template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::IOComponentType
    LDDMMData<TFloat, VDim>
    ::img_read(const char *fn, ImagePointer &trg)
{
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(fn);
  reader->Update();
  trg = reader->GetOutput();

  return reader->GetImageIO()->GetComponentType();
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::ImagePointer
    LDDMMData<TFloat, VDim>
    ::img_read(const char *fn)
{
  ImagePointer p;
  img_read(fn, p);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_write(ImageType *src, const char *fn, IOComponentType comp)
{
  lddmm_data_io::write_cast_to_iocomp(src, fn, comp);
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_write(CompositeImageType *src, const char *fn, IOComponentType comp)
{
  lddmm_data_io::write_cast_to_iocomp(src, fn, comp);
}


template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::IOComponentType
    LDDMMData<TFloat, VDim>
    ::vimg_read(const char *fn, VectorImagePointer &trg)
{
  typedef itk::ImageFileReader<VectorImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(fn);
  reader->Update();
  trg = reader->GetOutput();

  return reader->GetImageIO()->GetComponentType();
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::VectorImagePointer
    LDDMMData<TFloat, VDim>
    ::vimg_read(const char *fn)
{
  VectorImagePointer p;
  vimg_read(fn, p);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_write(VectorImageType *src, const char *fn, IOComponentType comp)
{
  // Cast to vector image type
  typedef itk::VectorImage<TFloat, VDim> OutputImageType;
  typename OutputImageType::Pointer output = OutputImageType::New();
  output->CopyInformation(src);
  output->SetRegions(src->GetBufferedRegion());
  output->SetNumberOfComponentsPerPixel(VDim);

  // Override the data pointer
  output->GetPixelContainer()->SetImportPointer(
      (TFloat *) src->GetBufferPointer(),
      VDim * src->GetPixelContainer()->Size(), false);

  // Write
  lddmm_data_io::write_cast_to_iocomp(output.GetPointer(), fn, comp);
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::IOComponentType
    LDDMMData<TFloat, VDim>
    ::cimg_read(const char *fn, CompositeImagePointer &trg)
{
  typedef itk::ImageFileReader<CompositeImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(fn);
  reader->Update();
  trg = reader->GetOutput();

  return reader->GetImageIO()->GetComponentType();
}


template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::cimg_read(const char *fn)
{
  CompositeImagePointer p;
  cimg_read(fn, p);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vfield_read(uint nt, const char *fnpat, VelocityField &v)
{
  v.clear();
  for(uint i = 0; i < nt; i++)
    {
      char fname[1024];
      snprintf(fname, 1024, fnpat, i);
      VectorImagePointer vt;
      vimg_read(fname, vt);
      v.push_back(vt);
    }
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_copy(const VectorImageType *src, VectorImageType *trg)
{
  typedef itk::CastImageFilter<VectorImageType, VectorImageType> CastFilter;
  typename CastFilter::Pointer fltCast = CastFilter::New();
  fltCast->SetInput(src);
  fltCast->GraftOutput(trg);
  fltCast->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_copy(const CompositeImageType *src, CompositeImageType *trg)
{
  typedef itk::CastImageFilter<CompositeImageType, CompositeImageType> CastFilter;
  typename CastFilter::Pointer fltCast = CastFilter::New();
  trg->SetNumberOfComponentsPerPixel(src->GetNumberOfComponentsPerPixel());
  fltCast->SetInput(src);
  fltCast->GraftOutput(trg);
  fltCast->Update();
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::ImagePointer
    LDDMMData<TFloat, VDim>
    ::img_dup(const ImageType *src)
{
  if(!src)
    return nullptr;

  typedef itk::ImageDuplicator<ImageType> DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(src);
  duplicator->Update();
  return duplicator->GetOutput();
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::VectorImagePointer
    LDDMMData<TFloat, VDim>
    ::vimg_dup(const VectorImageType *src)
{
  if(!src)
    return nullptr;

  typedef itk::ImageDuplicator<VectorImageType> DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(src);
  duplicator->Update();
  return duplicator->GetOutput();
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::cimg_dup(const CompositeImageType *src)
{
  if(!src)
    return nullptr;

  typedef itk::ImageDuplicator<CompositeImageType> DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(src);
  duplicator->Update();
  return duplicator->GetOutput();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_copy(const ImageType *src, ImageType *trg)
{
  typedef itk::CastImageFilter<ImageType, ImageType> CastFilter;
  typename CastFilter::Pointer fltCast = CastFilter::New();
  fltCast->SetInput(src);
  fltCast->GraftOutput(trg);
  fltCast->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::mimg_copy(const MatrixImageType *src, MatrixImageType *trg)
{
  typedef itk::CastImageFilter<MatrixImageType, MatrixImageType> CastFilter;
  typename CastFilter::Pointer fltCast = CastFilter::New();
  fltCast->SetInput(src);
  fltCast->GraftOutput(trg);
  fltCast->Update();
}

template<class TFloat, uint VDim>
bool
    LDDMMData<TFloat, VDim>
    ::vimg_auto_cast(const VectorImageType *src, ImageBaseType *trg)
{
  return lddmm_data_io::auto_cast(src, trg);
}

template<class TFloat, uint VDim>
bool
    LDDMMData<TFloat, VDim>
    ::cimg_auto_cast(const CompositeImageType *src, ImageBaseType *trg)
{
  return lddmm_data_io::auto_cast(src, trg);
}

template<class TFloat, uint VDim>
bool
    LDDMMData<TFloat, VDim>
    ::img_same_space(const ImageBaseType *i1, const ImageBaseType *i2, double tol)
{
  double c_tol = std::abs(tol * i1->GetSpacing()[0]), d_tol = tol;
  return
      i1->GetBufferedRegion() == i2->GetBufferedRegion() &&
      i1->GetSpacing().GetVnlVector().is_equal(i2->GetSpacing().GetVnlVector(), c_tol) &&
      i1->GetOrigin().GetVnlVector().is_equal(i2->GetOrigin().GetVnlVector(), c_tol) &&
      i1->GetDirection().GetVnlMatrix().as_ref().is_equal(i2->GetDirection().GetVnlMatrix().as_ref(), d_tol);
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::cimg_concat(const std::vector<CompositeImagePointer> &img)
{
  // Simple case where there is nothing to pack
  if(img.size() == 0) return nullptr;
  else if(img.size() == 1) return img[0];

  // Otherwise figure the number of output components
  unsigned int ncomp = 0;
  for(unsigned int j = 0; j < img.size(); j++)
    ncomp += img[j]->GetNumberOfComponentsPerPixel();

  // Create the output image
  typedef LDDMMData<TFloat, VDim> LDDMMType;
  CompositeImagePointer pack = LDDMMType::new_cimg(img[0], ncomp);

  // For each input image, copy its data into the output
  // Use the new parallelism code
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  mt->ParallelizeImageRegion<VDim>(
      pack->GetBufferedRegion(),
      [pack,img,&ncomp](const itk::ImageRegion<VDim> &region)
      {
        unsigned int offset = 0;
        for(auto src : img)
          {
            // Components being packed
            unsigned int nc_j = src->GetNumberOfComponentsPerPixel();

            // Iterator typdef
            typedef itk::ImageLinearConstIteratorWithIndex<CompositeImageType> IterBase;
            typedef IteratorExtender<IterBase> Iterator;
            unsigned int line_length = region.GetSize(0);
            for(Iterator it(pack, region); !it.IsAtEnd(); it.NextLine())
              {
                TFloat *pack_line = it.GetPixelPointer(pack.GetPointer()) + offset;
                const TFloat *pack_line_end = pack_line + ncomp * line_length;
                const TFloat *src_line = it.GetPixelPointer(src.GetPointer());
                for(; pack_line < pack_line_end; pack_line += ncomp)
                  for(unsigned int j = 0; j < nc_j; j++)
                    pack_line[j] = *src_line++;
              }

            // Update the offset
            offset += nc_j;
          }
      }, nullptr);

  return pack;
}

template<class TFloat, uint VDim>
unsigned int
    LDDMMData<TFloat, VDim>
    ::cimg_nancount(const CompositeImageType *img)
{
  // Create a fake region to partition the entire data chunk
  unsigned int npix = img->GetPixelContainer()->Size();
  itk::ImageRegion<1> full_region({{0}}, {{npix}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  std::atomic<unsigned int> nan_count(0);
  mt->ParallelizeImageRegion<1>(
      full_region,
      [img,&nan_count](const itk::ImageRegion<1> &thread_region)
      {
        unsigned int thread_nan_count = 0;
        const TFloat *p = img->GetBufferPointer() + thread_region.GetIndex(0);
        const TFloat *p_end = p + thread_region.GetSize(0);
        for(; p < p_end; ++p)
          if(std::isnan(*p))
            thread_nan_count++;

        nan_count += thread_nan_count;
      }, nullptr);

  return nan_count;
}

template<class TFloat, uint VDim>
void LDDMMData<TFloat, VDim>::cimg_mask_smooth_adjust_in_place(
    CompositeImageType *img, ImageType *mask, TFloat thresh)
{
  // Create a fake region to partition the entire data chunk
  unsigned int nc = img->GetNumberOfComponentsPerPixel();
  unsigned int np = img->GetBufferedRegion().GetNumberOfPixels();
  itk::ImageRegion<1> full_region({{0}}, {{np}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  mt->ParallelizeImageRegion<1>(
      full_region,
      [img,mask,thresh,nc](const itk::ImageRegion<1> &thread_region)
      {
        TFloat *p = img->GetBufferPointer() + nc * thread_region.GetIndex(0);
        TFloat *m = mask->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *m_end = m + thread_region.GetSize(0);
        for(; m < m_end; ++m)
          {
            if(*m < thresh)
              {
                for(unsigned int i = 0; i < nc; i++)
                  *p++ = 0.0;
                *m = 0.0;
              }
            else
              {
                for(unsigned int i = 0; i < nc; i++)
                  *p++ /= *m;
                *m = 1.0;
              }
          }
      }, nullptr);

  img->Modified();
  mask->Modified();
}

template<class TFloat, uint VDim>
bool
    LDDMMData<TFloat, VDim>
    ::img_auto_cast(const ImageType *src, ImageBaseType *trg)
{
  return lddmm_data_io::auto_cast(src, trg);
}


template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_shrink(ImageType *src, ImageType *trg, int factor)
{
  typedef itk::ShrinkImageFilter<ImageType, ImageType> Filter;
  typename Filter::Pointer filter = Filter::New();
  filter->SetInput(src);
  filter->SetShrinkFactors(factor);
  filter->GraftOutput(trg);
  filter->Update();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_resample_identity(ImageType *src, ImageBaseType *ref, ImageType *trg)
{
  typedef itk::ResampleImageFilter<ImageType, ImageType, TFloat> ResampleFilter;
  typedef itk::IdentityTransform<TFloat, VDim> TranType;
  typedef itk::LinearInterpolateImageFunction<ImageType, TFloat> InterpType;

  typename ResampleFilter::Pointer filter = ResampleFilter::New();
  typename TranType::Pointer tran = TranType::New();
  typename InterpType::Pointer func = InterpType::New();

  filter->SetInput(src);
  filter->SetTransform(tran);
  filter->SetInterpolator(func);
  filter->UseReferenceImageOn();
  filter->SetReferenceImage(ref);
  filter->GraftOutput(trg);
  filter->Update();
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::ImageBasePointer
    LDDMMData<TFloat, VDim>
    ::create_reference_space_for_downsample(ImageBaseType *src, Vec factors)
{
  // Compute the size and index of the new image
  typename ImageBaseType::SizeType sz_pre = src->GetBufferedRegion().GetSize(), sz_post;
  typename ImageBaseType::IndexType idx_pre = src->GetBufferedRegion().GetIndex(), idx_post;
  typename ImageBaseType::SpacingType spc_pre = src->GetSpacing(), spc_post;
  typename ImageBaseType::PointType origin_pre = src->GetOrigin(), origin_post;

  // Compute size, index and spacing
  for(unsigned int i = 0; i < VDim; i++)
    {
      // Size gets rounded up (since it doesn't have to be exact, we adjust based on spacing
      sz_post[i] = (unsigned long) std::ceil(sz_pre[i] / factors[i]);

      // Index gets rounded to closest int
      idx_post[i] = (long) std::floor(idx_pre[i] / factors[i] + 0.5);

      // Compute the spacing (to keep the bounding box exactly the same)
      spc_post[i] = spc_pre[i] * sz_pre[i] * 1.0 / sz_post[i];
    }

  // Compute the direction * spacing vectors
  typename ImageBaseType::SpacingType ds_pre = (src->GetDirection() * spc_pre);
  typename ImageBaseType::SpacingType ds_post = (src->GetDirection() * spc_post);

  // Recalculate the origin. The origin describes the center of voxel 0,0,0
  // so that as the voxel size changes, the origin will change as well.
  for(unsigned int i = 0; i < VDim; i++)
    {
      origin_post[i] = origin_pre[i] + ds_pre[i] * (idx_pre[i] - 0.5) - ds_post[i] * (idx_post[i] - 0.5);
    }

  // Weird - have to allocate the output image?
  ImagePointer ref = ImageType::New();
  ref->SetRegions(typename ImageBaseType::RegionType(idx_post, sz_post));
  ref->SetOrigin(origin_post);
  ref->SetSpacing(spc_post);
  ref->SetDirection(src->GetDirection());

  return ImageBasePointer(ref.GetPointer());
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::cimg_downsample(CompositeImageType *img, Vec factors)
{
  // First smooth the image to avoid aliasing
  Vec sigma;
  for(unsigned int d = 0; d < VDim; d++)
    sigma[d] = (factors[d] > 1) ? 0.5 * factors[d] * img->GetSpacing()[d] : 0.0;

  CompositeImagePointer smoothed = CompositeImageType::New();
  cimg_smooth(img, smoothed, sigma);

  // Create a target space
  ImageBasePointer ref_space = create_reference_space_for_downsample(img, factors);

  // Use the fast resampler to resample the image to target resolution
  typedef FastWarpCompositeImageFilter<CompositeImageType, CompositeImageType, VectorImageType> WF;
  typename WF::Pointer wf = WF::New();
  wf->SetReferenceSpace(ref_space);
  wf->SetMovingImage(smoothed);
  wf->SetUseNearestNeighbor(false);
  wf->SetUsePhysicalSpace(true);
  wf->SetOutsideValue(0);
  wf->Update();

  // Just return the output
  return wf->GetOutput();
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::ImagePointer
    LDDMMData<TFloat, VDim>
    ::img_downsample(ImageType *img, Vec factors)
{
  // First smooth the image to avoid aliasing
  Vec sigma;
  for(unsigned int d = 0; d < VDim; d++)
    sigma[d] = (factors[d] > 1) ? 0.5 * factors[d] * img->GetSpacing()[d] : 0.0;

  ImagePointer smoothed = ImageType::New();
  img_smooth(img, smoothed, sigma);

  // Create a target space
  ImageBasePointer ref_space = create_reference_space_for_downsample(img, factors);

  // Use the fast resampler to resample the image to target resolution
  typedef FastWarpCompositeImageFilter<ImageType, ImageType, VectorImageType> WF;
  typename WF::Pointer wf = WF::New();
  wf->SetReferenceSpace(ref_space);
  wf->SetMovingImage(smoothed);
  wf->SetUseNearestNeighbor(false);
  wf->SetUsePhysicalSpace(true);
  wf->SetOutsideValue(0);
  wf->Update();

  // Just return the output
  return wf->GetOutput();
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_resample_identity(VectorImageType *src, ImageBaseType *ref, VectorImageType *trg)
{
  typedef itk::ResampleImageFilter<VectorImageType, VectorImageType, TFloat> ResampleFilter;
  typedef itk::IdentityTransform<TFloat, VDim> TranType;
  typedef itk::LinearInterpolateImageFunction<VectorImageType, TFloat> InterpType;

  typename ResampleFilter::Pointer filter = ResampleFilter::New();
  typename TranType::Pointer tran = TranType::New();
  typename InterpType::Pointer func = InterpType::New();

  filter->SetInput(src);
  filter->SetTransform(tran);
  filter->SetInterpolator(func.GetPointer());
  filter->SetSize(ref->GetBufferedRegion().GetSize());
  filter->SetOutputSpacing(ref->GetSpacing());
  filter->SetOutputOrigin(ref->GetOrigin());
  filter->SetOutputDirection(ref->GetDirection());
  filter->SetOutputStartIndex(ref->GetBufferedRegion().GetIndex());
  filter->GraftOutput(trg);
  filter->Update();
}

template <class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::VectorImagePointer
    LDDMMData<TFloat, VDim>
    ::vimg_resample_identity(VectorImageType *src, ImageBaseType *ref)
{
  VectorImagePointer p = VectorImageType::New();
  vimg_resample_identity(src, ref, p);
  return p;
}

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_threshold_in_place(ImageType *src, double lt, double ut, double fore, double back)
{
  typedef itk::BinaryThresholdImageFilter<ImageType, ImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput(src);
  filter->GraftOutput(src);
  filter->SetLowerThreshold(lt);
  filter->SetUpperThreshold(ut);
  filter->SetInsideValue(fore);
  filter->SetOutsideValue(back);
  filter->Update();
}

template <class TImage>
class MaskNaNFunctor
{
public:
  typedef typename TImage::PixelType PixelType;
  PixelType operator() (const PixelType &x)
  {
    return std::isnan(x) ? 1 : 0;
  }
};



template <class TImage>
class FilterNaNFunctor
{
public:
  typedef typename TImage::PixelType PixelType;
  PixelType operator() (const PixelType &x)
  {
    return std::isnan(x) ? 0 : x;
  }
};

template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_filter_nans_in_place(ImageType *src, ImageType *nan_mask)
{
  typedef MaskNaNFunctor<ImageType> MaskFunctor;
  typedef itk::UnaryFunctorImageFilter<ImageType, ImageType, MaskFunctor> MaskFilterType;
  typename MaskFilterType::Pointer mask = MaskFilterType::New();
  mask->SetInput(src);
  mask->GraftOutput(nan_mask);
  mask->Update();

  typedef FilterNaNFunctor<ImageType> RemoveFunctor;
  typedef itk::UnaryFunctorImageFilter<ImageType, ImageType, RemoveFunctor> RemoveFilterType;
  typename RemoveFilterType::Pointer remove = RemoveFilterType::New();
  remove->SetInput(src);
  remove->GraftOutput(src);
  remove->Update();
}

template <class TImage>
class ReconstituteNaNFunctor
{
public:
  typedef typename TImage::PixelType PixelType;
  PixelType operator() (const PixelType &x, const PixelType &m)
  {
    return m > 0 ? nan("") : x;
  }
};


template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::img_reconstitute_nans_in_place(ImageType *src, ImageType *nan_mask)
{
  typedef ReconstituteNaNFunctor<ImageType> Functor;
  typedef itk::BinaryFunctorImageFilter<ImageType, ImageType, ImageType, Functor> FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput1(src);
  filter->SetInput2(nan_mask);
  filter->GraftOutput(src);
  filter->Update();
}

template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_add_in_place(CompositeImageType *trg, CompositeImageType *a)
{
  // The regions of the two images must be the same
  struct AddFunctor {
    static TFloat apply (TFloat a, TFloat b) { return a+b; }
  };

  cimg_apply_binary_functor_in_place<AddFunctor>(trg, a);
}

template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_scale_in_place(CompositeImageType *trg, TFloat scale)
{
  // Create a fake region to partition the entire data chunk
  unsigned int npix = trg->GetPixelContainer()->Size();
  itk::ImageRegion<1> full_region({{0}}, {{npix}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  mt->ParallelizeImageRegion<1>(
      full_region,
      [trg, scale](const itk::ImageRegion<1> &thread_region)
      {
        TFloat *pt = trg->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *pt_end = pt + thread_region.GetSize(0);
        for(; pt < pt_end; ++pt)
          *pt *= scale;
      }, nullptr);
}

template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_threshold_in_place(CompositeImageType *trg, double lt, double up, double fore, double back)
{
  // Create a fake region to partition the entire data chunk
  unsigned int npix = trg->GetPixelContainer()->Size();
  itk::ImageRegion<1> full_region({{0}}, {{npix}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  mt->ParallelizeImageRegion<1>(
      full_region,
      [trg, lt, up, fore, back](const itk::ImageRegion<1> &thread_region)
      {
        TFloat *pt = trg->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *pt_end = pt + thread_region.GetSize(0);
        for(; pt < pt_end; ++pt)
          *pt = (*pt >= lt && *pt <= up) ? fore : back;
      }, nullptr);
}

#include <chrono>

// A simple hashing function
inline size_t mix_hash(size_t idx, size_t array_size) {
  idx ^= (idx >> 33);
  idx *= 0xff51afd7ed558ccd;
  idx ^= (idx >> 33);
  idx *= 0xc4ceb9fe1a85ec53;
  idx ^= (idx >> 33);
  return idx % array_size;
}

template<class TFloat, uint VDim>
void LDDMMData<TFloat, VDim>
    ::cimg_add_gaussian_noise_in_place(
        CompositeImageType *img, const std::vector<double> &sigma, std::mt19937 &rnd)
{
  // Create a fake region to partition the entire data chunk
  unsigned int npix = img->GetBufferedRegion().GetNumberOfPixels();
  itk::ImageRegion<1> full_region({{0}}, {{npix}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  // Compute an array of randomly sampled normal values
  constexpr int normal_sample_size = 10000;
  std::normal_distribution<TFloat> ndist(0., 1.);
  TFloat normal_sample[normal_sample_size];
  for(unsigned int i = 0; i < normal_sample_size; i++)
    normal_sample[i] = ndist(rnd);

  // We need a way to hash an offset
  mt->ParallelizeImageRegion<1>(
      full_region,
      [img, &sigma, normal_sample, normal_sample_size](const itk::ImageRegion<1> &thread_region)
      {
        unsigned int nc = img->GetNumberOfComponentsPerPixel();

        TFloat *p_buffer = img->GetBufferPointer();
        TFloat *p_start = p_buffer + thread_region.GetIndex(0) * nc;
        TFloat *p_end = p_start + thread_region.GetSize(0) * nc;

        // Just generate random numbers, no reuse
        for(TFloat *p = p_start; p < p_end; p += nc)
          {
            for(unsigned int j = 0; j < nc; j++)
              {
                size_t n_index = mix_hash(j + p - p_buffer, normal_sample_size);
                p[j] += normal_sample[n_index] * sigma[j];
              }
          }
      }, nullptr);

  // Mark image as modified
  img->Modified();
}

template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::vimg_add_gaussian_noise_in_place(
        VectorImageType *img, double sigma, std::mt19937 &rnd)
{
  // Use the cimg method by masquerading
  CompositeImagePointer cimg = vimg_as_cimg(img);
  std::vector<double> sigma_vec(VDim, sigma);
  cimg_add_gaussian_noise_in_place(cimg, sigma_vec, rnd);
  img->Modified();
}

template <class TImage>
struct VoxelToPhysicalFunctor
{

  typedef typename TImage::PixelType PixelType;
  typedef itk::ImageBase<TImage::ImageDimension> ImageBaseType;

  PixelType operator() (const PixelType &x)
  {
    typedef itk::ContinuousIndex<double, TImage::ImageDimension> CIType;
    typedef typename TImage::PointType PtType;
    CIType ci0, ci;
    PtType p0, p;
    for(unsigned int i = 0; i < TImage::ImageDimension; i++)
      {
        ci[i] = x[i];
        ci0[i] = 0.0;
      }

    m_Image->TransformContinuousIndexToPhysicalPoint(ci, p);
    m_Image->TransformContinuousIndexToPhysicalPoint(ci0, p0);
    PixelType y;

    for(unsigned int i = 0; i < TImage::ImageDimension; i++)
      {
        y[i] = p[i] - p0[i];
      }

    return y;
  }

  bool operator != (const VoxelToPhysicalFunctor<TImage> &other)
  { return other.m_Image != m_Image; }

  ImageBaseType *m_Image;
};

template <class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::warp_voxel_to_physical(VectorImageType *src, ImageBaseType *ref_space, VectorImageType *trg)
{
  // Set up functor
  typedef VoxelToPhysicalFunctor<VectorImageType> FunctorType;
  FunctorType fnk;
  fnk.m_Image = ref_space;

  // Set up filter
  typedef itk::UnaryFunctorImageFilter<VectorImageType, VectorImageType, FunctorType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput(src);
  filter->GraftOutput(trg);
  filter->SetFunctor(fnk);
  filter->Update();
}



/* =============================== */

#ifdef _LDDMM_FFT_


template <class TFloat, uint VDim>
LDDMMFFTInterface<TFloat, VDim>
    ::LDDMMFFTInterface(ImageType *ref)
{
  // Work out the data dimensions (large enough, and multiple of four)
  m_Size = ref->GetBufferedRegion().GetSize();
  m_Alloc = m_Size;
  m_Alloc[VDim-1] = 2 * (m_Size[VDim-1] / 2 + 1);

  // Size for calling the plan routines
  int n[VDim];

  // Get the data dimensions
  m_AllocSize = 1; m_DataSize = 1;
  for(uint i = 0; i < VDim; i++)
    {
      m_AllocSize *= m_Alloc[i];
      m_DataSize *= m_Size[i];
      n[i] = m_Size[i];
    }

  // Allocate the complex data (real data is packed in the complex data)
  m_Data = (double *) fftw_malloc(m_AllocSize * sizeof(double));

  // Create plans for forward and inverse transforms
  m_Plan = fftw_plan_dft_r2c(VDim, n, m_Data, (fftw_complex *) m_Data, FFTW_MEASURE);
  m_InvPlan = fftw_plan_dft_c2r(VDim, n, (fftw_complex *) m_Data, m_Data, FFTW_MEASURE);
}

template <class TFloat, uint VDim>
void
    LDDMMFFTInterface<TFloat, VDim>
    ::convolution_fft(
        VectorImageType *img, ImageType *kernel_ft, bool inv_kernel,
        VectorImageType *out)
{
  // Pack the data into m_Data. This requires us to skip a few bytes at
  // the end of each row of data
  uint nskip = m_Alloc[VDim-1] - m_Size[VDim-1];
  uint ncopy = m_Size[VDim-1];
  uint nout = m_Alloc[VDim-1] / 2;
  uint noutskip = kernel_ft->GetBufferedRegion().GetSize()[VDim-1] - nout;
  uint nstrides = m_AllocSize / m_Alloc[VDim-1];

  for(uint d = 0; d < VDim; d++)
    {
      const Vec *src = img->GetBufferPointer();
      double *dst = m_Data;

      // Funky loop
      for(double *end = dst + m_AllocSize; dst < end; dst+=nskip)
        for(double *rowend = dst + ncopy; dst < rowend; dst++, src++)
          *dst = (double) (*src)[d];

      // Execute the plan
      fftw_execute(m_Plan);

      // Multiply or divide the complex values in m_Data by the kernel array
      fftw_complex *c = (fftw_complex *) m_Data;

      // Scaling factor for producing final result
      double scale = 1.0 / m_DataSize;

      /*
      // Precision weirdness (results differ from MATLAB fft in 6th, 7th decimal digit)
      uint tp = (m_Alloc[VDim-1] / 2) * 6 + 8;
      printf("Before scaling, value at %d is %12.12lf, %12.12lf\n",
      tp, c[tp][0], c[tp][1]);
      printf("Kernel value at %d is %12.12lf\n", 128*6+8, kp[128*6+8]);
      */

      // Another such loop
      TFloat *kp = kernel_ft->GetBufferPointer();
      if(inv_kernel)
        {
          for(uint i = 0; i < nstrides; i++)
            {
              for(uint j = 0; j < nout; j++)
                {
                  (*c)[0] /= (*kp);
                  (*c)[1] /= (*kp);
                  c++; kp++;
                }
              kp += noutskip;
            }
        }
      else
        {
          for(uint i = 0; i < nstrides; i++)
            {
              for(uint j = 0; j < nout; j++)
                {
                  (*c)[0] *= (*kp);
                  (*c)[1] *= (*kp);
                  c++; kp++;
                }
              kp += noutskip;
            }
        }

      /*
      fftw_complex *ctest = (fftw_complex *) m_Data;
      printf("After scaling, value at %d is %12.12lf, %12.12lf\n",
      tp, ctest[tp][0], ctest[tp][1]);
      */

      // Inverse transform
      fftw_execute(m_InvPlan);

      // Copy the results to the output image
      const double *osrc = m_Data;
      Vec *odst = out->GetBufferPointer();
      for(uint i = 0; i < nstrides; i++, osrc+=nskip)
        for(uint j = 0; j < ncopy; j++, odst++, osrc++)
          (*odst)[d] = (TFloat) ((*osrc) * scale);

      /*
      odst = out->GetBufferPointer();
      printf("Result %12.12lf\n",  odst[128*6+8][0]);
      */
    }

}

template <class TFloat, uint VDim>
LDDMMFFTInterface<TFloat, VDim>
    ::~LDDMMFFTInterface()
{
  fftw_destroy_plan(m_Plan);
  fftw_destroy_plan(m_InvPlan);
  fftw_free(m_Data);
}

#endif // _LDDMM_FFT_


template <class TFloat, uint VDim>
LDDMMImageMatchingObjective<TFloat, VDim>
    ::LDDMMImageMatchingObjective(LDDMM &p)
    : fft(p.fix)
{
  // Allocate intermediate datasets
  Jt0 = LDDMM::new_img(p.fix);
  Jt1 = LDDMM::new_img(p.fix);
  DetPhit1 = LDDMM::new_img(p.fix);
  GradJt0 = LDDMM::new_vimg(p.fix);
}

template <class TFloat, uint VDim>
TFloat
    LDDMMImageMatchingObjective<TFloat, VDim>
    ::compute_objective_and_gradient(LDDMM &p)
{
  // Compute the regularization energy of v. We can use a[0] for temporary storage
  // e_field = lddmm_vector_field_dot_product(vx, vy, vx, vy, p);
  TFloat e_field = 0.0;
  for(uint m = 0; m < p.nt; m++)
    {
      // a[0] = Lv[m] .* v[m]
      fft.convolution_fft(p.v[m], p.f_kernel_sq, false, p.a[0]);

      // We're sticking the inner product in Jt0
      LDDMM::vimg_euclidean_inner_product(Jt0, p.a[0], p.v[m]);
      e_field += LDDMM::img_voxel_sum(Jt0) / p.nt;
    }

  // Compute the 'a' array (for semilagrangean scheme)
  p.compute_semi_lagrangean_a();

  // Go through and compute phi_t1 for each time point
  p.integrate_phi_t1();

  // Compute the update for v at each time step
  for(uint m = 0; m < p.nt; m++)
    {
      // Currently, f[m] holds phi_t1[m]. Use it for whatever we need
      // and then replace with phi_t0[m]

      // TODO: for ft00 and ft11, don't waste time on interpolation

      // Jt1 = lddmm_warp_scalar_field(p.I1, ft1x(:,:,it), ft1y(:,:,it), p);
      LDDMM::interp_img(p.mov, p.f[m], Jt1);

      // detjac_phi_t1 = lddmm_jacobian_determinant(ft1x(:,:,it), ft1y(:,:,it), p);
      LDDMM::field_jacobian_det(p.f[m], DetPhit1);

      // Place phi_t0 into the f array
      if(m==0)
        {
          p.f[m]->FillBuffer(typename LDDMM::Vec(0.0));
        }
      else
        {
          LDDMM::interp_vimg(p.f[m-1], p.a[m], -1.0, p.f[m]);
          LDDMM::vimg_subtract_in_place(p.f[m], p.a[m]);
        }

      // Jt0 = lddmm_warp_scalar_field(p.I0, ft0x(:,:,it), ft0y(:,:,it), p);
      LDDMM::interp_img(p.fix, p.f[m], Jt0);

      // [grad_Jt0_x grad_Jt0_y] = gradient(Jt0);
      LDDMM::image_gradient(Jt0, GradJt0, false);

      // pde_rhs_x = detjac_phi_t1 .* (Jt0 - Jt1) .* grad_Jt0_x;
      // pde_rhs_y = detjac_phi_t1 .* (Jt0 - Jt1) .* grad_Jt0_y;

      // Here we do some small tricks. We want to retain Jt0 because it's the warped
      // template image, and we want to retain the difference Jt0-Jt1 = (J0-I1) for
      // calculating the objective at the end.
      LDDMM::img_subtract_in_place(Jt1, Jt0);           // 'Jt1' stores Jt1 - Jt0
      LDDMM::img_multiply_in_place(DetPhit1, Jt1);      // 'DetPhit1' stores (det Phi_t1)(Jt1-Jt0)
      LDDMM::vimg_multiply_in_place(GradJt0, DetPhit1); // 'GradJt0' stores  GradJt0 * (det Phi_t1)(Jt1-Jt0)

      // Solve PDE via FFT convolution
      // pde_soln_x = ifft2(fft2(pde_rhs_x) ./ p.f_kernel_sq,'symmetric');
      // pde_soln_y = ifft2(fft2(pde_rhs_y) ./ p.f_kernel_sq,'symmetric');
      fft.convolution_fft(GradJt0, p.f_kernel_sq, true, GradJt0); // 'GradJt0' stores K[ GradJt0 * (det Phi_t1)(Jt1-Jt0) ]

      // dedvx(:,:,it) = dedvx(:,:,it) - 2 * pde_soln_x / p.sigma^2;
      // dedvy(:,:,it) = dedvy(:,:,it) - 2 * pde_soln_y / p.sigma^2;

      // Store the update in a[m]
      LDDMM::vimg_scale_in_place(GradJt0, 1.0 / p.sigma_sq); // 'GradJt0' stores 1 / sigma^2 K[ GradJt0 * (det Phi_t1)(Jt1-Jt0) ]
      LDDMM::vimg_add_in_place(GradJt0, p.v[m]); // 'GradJt0' stores v + 1 / sigma^2 K[ GradJt0 * (det Phi_t1)(Jt1-Jt0) ]
      LDDMM::vimg_scale(GradJt0, 2.0, p.a[m]); // p.a[m] holds 2 v + 2 / sigma^2 K[ GradJt0 * (det Phi_t1)(Jt1-Jt0) ]
    }

  // Ok, Jt1 currently contains (Jt1-Jt0), we just need to square it.
  TFloat e_image = LDDMM::img_euclidean_norm_sq(Jt1) / p.sigma_sq;

  // Return the energy
  printf("  Energy components: %lf, %lf\n", e_field, e_image);
  return e_field + e_image;
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::as_cimg(ImageBaseType *src)
{
  // Cast to different types
  if(auto *cimg = dynamic_cast<CompositeImageType *>(src))
    return cimg;
  else if(auto *img = dynamic_cast<ImageType *>(src))
    return img_as_cimg(img);
  else if(auto *vimg = dynamic_cast<VectorImageType *>(src))
    return vimg_as_cimg(vimg);
  else
    return nullptr;
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::vimg_as_cimg(VectorImageType *src)
{
  CompositeImagePointer cimg = CompositeImageType::New();
  cimg->CopyInformation(src);
  cimg->SetNumberOfComponentsPerPixel(VDim);
  cimg->SetRegions(src->GetBufferedRegion());
  cimg->GetPixelContainer()->SetImportPointer(
      reinterpret_cast<TFloat *>(src->GetBufferPointer()),
      src->GetBufferedRegion().GetNumberOfPixels() * VDim,
      false);
  return cimg;
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::ImagePointer
    LDDMMData<TFloat, VDim>
    ::cimg_as_img(CompositeImageType *src)
{
  itkAssertOrThrowMacro(src->GetNumberOfComponentsPerPixel() == 1,
                         "Multicomponent image passed to cimg_as_img");

  ImagePointer img = ImageType::New();
  img->CopyInformation(src);
  img->SetRegions(src->GetBufferedRegion());
  img->SetPixelContainer(src->GetPixelContainer());
  return img;
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::CompositeImagePointer
    LDDMMData<TFloat, VDim>
    ::img_as_cimg(ImageType *src)
{
  CompositeImagePointer img = CompositeImageType::New();
  img->CopyInformation(src);
  img->SetNumberOfComponentsPerPixel(1);
  img->SetRegions(src->GetBufferedRegion());
  img->SetPixelContainer(src->GetPixelContainer());
  return img;
}

template<class TFloat, uint VDim>
void LDDMMData<TFloat, VDim>::cimg_fast_convolution_smooth_inplace(CompositeImageType *img, SmoothingSigmas sigma, SmoothingMode mode)
{
  itkAssertOrThrowMacro(mode == FAST_ZEROPAD || mode == FAST_REFLECT, "Mode must be FAST_ZEROPAD or FAST_REFLECT");

  // Map the sigmas into voxel units
  Vec sigma_voxel = sigma.GetSigmaInVoxelUnits(img);

  // Leverage separability of the filter, i.e., repeat for every dimension
  for(int d = 0; d < VDim; d++)
    {
      // Zero sigma means no smoothing
      if(sigma_voxel[d] == 0.0)
        continue;

      // Generate the half-kernel for this dimension
      int m = std::max(2, 1 + (int) std::ceil(sigma_voxel[d] * sigma.cutoff_in_units_of_sigma));
      TFloat *kernel = new TFloat[m];
      for(int i = 0; i < m; i++)
        kernel[i] = std::exp(-0.5 * (i * i) / (sigma_voxel[d] * sigma_voxel[d])) / (2.5066282746 * sigma_voxel[d]);

      // Get the line length
      int n = img->GetBufferedRegion().GetSize()[d];
      int k = img->GetNumberOfComponentsPerPixel();
      int nk = n * k;

      // Generate the sample array - this stores for every target location the list of locations
      // in the kernel*line matrix that contribute to that location
      // Sample arrays are generated based on the mode
      int *sample_k = new int[n];
      int *sample_offset;

      if(mode == FAST_ZEROPAD)
        {
          sample_offset = new int[nk * (2 * m-1) - m * (m - 1)];

          // Compute the sample offsets
          for(int i = 0, q = 0; i < n; i++)
            {
              // How many pixels do I have to the left and right of me
              int k_left = std::min(i, m-1), k_right = std::min((n-1) - i, m-1);
              sample_k[i] = 1 + k_left + k_right;

              // Iterate over the components
              for(int c = 0; c < k; c++)
                {
                  // Now record the offsets
                  for(int j = -k_left; j <= k_right; j++)
                    {
                      int row = std::abs(j), col = i + j;
                      sample_offset[q++] = row * nk + col * k + c;
                    }
                }
            }
        }
      else if(mode == FAST_REFLECT)
        {
          sample_offset = new int[nk * (2*m-1)];

          // Compute the sample offsets
          for(int i = 0, q = 0; i < n; i++)
            {
              sample_k[i] = 2 * m - 1;

              // Iterate over the components
              for(int c = 0; c < k; c++)
                {
                  // Now record the offsets
                  for(int j = -(m-1); j <= (m-1); j++)
                    {
                      int row = std::abs(j);
                      int col = (i + j);
                      if(col < 0 || col >= n)
                        col = i - j;
                      sample_offset[q++] = row * nk + col * k + c;
                    }
                }
            }
        }

      // Parallelize over the image region
      itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
      mt->ParallelizeImageRegionRestrictDirection<VDim>(
          d, img->GetBufferedRegion(),
          [img, kernel, d, k, m, n, nk, &sample_k, &sample_offset](const itk::ImageRegion<VDim> &thread_region)
          {
            // Create a line iterator over the thread region
            typedef itk::ImageLinearIteratorWithIndex<CompositeImageType> IterBase;
            typedef IteratorExtenderWithOffset<IterBase> IterType;
            IterType it(img, thread_region);
            it.SetDirection(d);
            int jump = it.GetOffset(d) * k;

            // Allocate a kernel * line matrix
            TFloat *klm = new TFloat[m * nk];

            // Iterate over the lines in the threaded region
            for(it.GoToBegin(); !it.IsAtEnd(); it.NextLine())
              {
                int i, j, c, q;

                // Get a pointer to the beginning of the line
                TFloat *p = it.GetPixelPointer(img);

                // Compute the kernel products -- these are all the required multiplications
                for(i = 0, q = 0; i < m; i++)
                  {
                    const TFloat *pi = p;
                    for(j = 0; j < n; j++, pi+=jump)
                      for(c = 0; c < k; c++)
                        klm[q++] = kernel[i] * pi[c];
                  }

                // Now compute all the sums and send to the output
                for(j = 0, q = 0; j < n; j++, p+=jump)
                  {
                    for(c = 0; c < k; c++)
                      {
                        p[c] = 0;
                        for(i = 0; i < sample_k[j]; i++)
                          {
                            auto v = klm[sample_offset[q++]];
                            p[c] += v;
                          }
                      }
                  }
              }

            delete [] klm;
          }, nullptr);

      delete [] kernel;
      delete [] sample_k;
      delete [] sample_offset;
    }
}


template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_extract_component(CompositeImageType *src, ImageType *trg, unsigned int c)
{
  itkAssertOrThrowMacro(
      trg->GetBufferedRegion() == src->GetBufferedRegion(),
      "Source and target image regions are different in cimg_extract_component");

  // Create a fake region to partition the entire data chunk
  unsigned int nc = src->GetNumberOfComponentsPerPixel();
  unsigned int np = src->GetBufferedRegion().GetNumberOfPixels();
  itk::ImageRegion<1> full_region({{0}}, {{np}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  mt->ParallelizeImageRegion<1>(
      full_region,
      [src,trg, nc, c](const itk::ImageRegion<1> &thread_region)
      {
        TFloat *p = src->GetBufferPointer() + nc * thread_region.GetIndex(0) + c;
        TFloat *m = trg->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *m_end = m + thread_region.GetSize(0);
        for(; m < m_end; ++m, p+=nc)
          *m = *p;
      }, nullptr);

  trg->Modified();
}

template<class TFloat, uint VDim>
void
    LDDMMData<TFloat, VDim>
    ::cimg_update_component(CompositeImageType *cimg, ImageType *comp, unsigned int c)
{
  itkAssertOrThrowMacro(
      cimg->GetBufferedRegion() == comp->GetBufferedRegion(),
      "Source and target image regions are different in cimg_extract_component");

  // Create a fake region to partition the entire data chunk
  unsigned int nc = cimg->GetNumberOfComponentsPerPixel();
  unsigned int np = cimg->GetBufferedRegion().GetNumberOfPixels();
  itk::ImageRegion<1> full_region({{0}}, {{np}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  mt->ParallelizeImageRegion<1>(
      full_region,
      [cimg,comp, nc, c](const itk::ImageRegion<1> &thread_region)
      {
        TFloat *p = cimg->GetBufferPointer() + nc * thread_region.GetIndex(0) + c;
        TFloat *m = comp->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *m_end = m + thread_region.GetSize(0);
        for(; m < m_end; ++m, p+=nc)
          *p = *m;
      }, nullptr);

  cimg->Modified();
}



template<class TFloat, uint VDim>
template<class TFunctor>
void
    LDDMMData<TFloat, VDim>
    ::cimg_apply_binary_functor_in_place(CompositeImageType *trg, CompositeImageType *a)
{
  // Regions must match
  itkAssertOrThrowMacro(trg->GetBufferedRegion() == a->GetBufferedRegion(),
                         "Image region mismatch in binary composite image operation");

  // Create a fake region to partition the entire data chunk
  unsigned int npix = trg->GetPixelContainer()->Size();
  itk::ImageRegion<1> full_region({{0}}, {{npix}});
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  mt->ParallelizeImageRegion<1>(
      full_region,
      [trg, a](const itk::ImageRegion<1> &thread_region)
      {
        TFloat *pt = trg->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *pa = a->GetBufferPointer() + thread_region.GetIndex(0);
        TFloat *pt_end = pt + thread_region.GetSize(0);
        for(; pt < pt_end; ++pt, ++pa)
          *pt = TFunctor::apply(*pt, *pa);
      }, nullptr);
}

template<class TFloat, uint VDim>
LDDMMData<TFloat, VDim>::SmoothingSigmas::SmoothingSigmas(const Vec &sigma, bool world_units, TFloat cutoff_in_units_of_sigma)
{
  this->sigma = sigma;
  this->world_units = world_units;
  this->cutoff_in_units_of_sigma = cutoff_in_units_of_sigma;
}

template<class TFloat, uint VDim>
LDDMMData<TFloat, VDim>::SmoothingSigmas::SmoothingSigmas(TFloat sigma, bool world_units, TFloat cutoff_in_units_of_sigma)
{
  this->sigma.Fill(sigma);
  this->world_units = world_units;
  this->cutoff_in_units_of_sigma = cutoff_in_units_of_sigma;
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::Vec
LDDMMData<TFloat, VDim>::SmoothingSigmas::GetSigmaInWorldUnits(const ImageBaseType *img) const
{
  if(this->world_units)
    return this->sigma;
  Vec result;
  for(unsigned int d = 0; d < VDim; d++)
    result[d] = this->sigma[d] * img->GetSpacing()[d];
  return result;
}

template<class TFloat, uint VDim>
typename LDDMMData<TFloat, VDim>::Vec
LDDMMData<TFloat, VDim>::SmoothingSigmas::GetSigmaInVoxelUnits(const ImageBaseType *img) const
{
  if(!this->world_units)
    return this->sigma;
  Vec result;
  for(unsigned int d = 0; d < VDim; d++)
    result[d] = this->sigma[d] / img->GetSpacing()[d];
  return result;
}


template class LDDMMData<float, 2>;
template class LDDMMData<float, 3>;
template class LDDMMData<float, 4>;

template class LDDMMData<double, 2>;
template class LDDMMData<double, 3>;
template class LDDMMData<double, 4>;

#ifdef _LDDMM_FFT_
template class LDDMMFFTInterface<double, 2>;
template class LDDMMFFTInterface<double, 3>;
template class LDDMMFFTInterface<double, 4>;
#endif

template class LDDMMImageMatchingObjective<myreal, 2>;
template class LDDMMImageMatchingObjective<myreal, 3>;



