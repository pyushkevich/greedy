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
#ifndef _LDDMM_DATA_H_
#define _LDDMM_DATA_H_

#include <lddmm_common.h>
#include <itkNumericTraits.h>
#include <itkImage.h>
#include <itkVectorImage.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkCovariantVector.h>
#include <vnl/vnl_math.h>
#include <vector>

#include <itkImageIOBase.h>

#include <fftw3.h>

template<class TFloat, uint VDim>
class LDDMMData
{
public:
  // Image data
  typedef itk::ImageBase<VDim> ImageBaseType;
  typedef itk::Image<TFloat, VDim> ImageType;
  typedef typename ImageType::Pointer ImagePointer;
  typedef itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator;

  // Vector fields, etc
  typedef itk::CovariantVector<TFloat, VDim> Vec;
  typedef itk::Image<Vec, VDim> VectorImageType;
  typedef typename VectorImageType::Pointer VectorImagePointer;
  typedef std::vector<VectorImagePointer> VelocityField;

  // Composite images (variable number of components)
  typedef itk::VectorImage<TFloat, VDim> CompositeImageType;
  typedef typename CompositeImageType::Pointer CompositeImagePointer;

  // Regions, etc
  typedef itk::ImageRegion<VDim> RegionType;

  // Pointers to the fixed and moving images
  ImagePointer fix, mov;

  // Fourier space kernels
  ImagePointer f_kernel, f_kernel_sq;

  // Velocity field pointers (v, phi, a used for semi-lagrange scheme)
  VelocityField v, f, a;

  // Region for the velocity fields
  RegionType r;

  // Parameters
  double alpha, sigma, gamma, dt, sigma_sq;
  
  // Dimensions
  uint n[VDim];

  // Number of timesteps, number of voxels
  uint nt, nv;

  // Allocate a velocity field
  static void alloc_vf(VelocityField &vf, uint nt, ImageBaseType *ref);
  static void alloc_img(ImagePointer &img, ImageBaseType *ref);
  static void alloc_vimg(VectorImagePointer &vimg, ImageBaseType *ref);
  static void alloc_cimg(CompositeImagePointer &img, ImageBaseType *ref, int n_comp);

  // Initialize LDDMM data 
  static void init(LDDMMData<TFloat, VDim> &, 
    ImageType *fix, ImageType *mov, 
    uint nt, double alpha, double gamma, double sigma);

  // Apply deformation to data
  static void interp_vimg(
    VectorImageType *data, VectorImageType *field, 
    TFloat def_scale, VectorImageType *out,
    bool use_nn = false, bool phys_space = false);

  // Apply deformation to data
  static void interp_img(ImageType *data, VectorImageType *field, ImageType *out,
                         bool use_nn = false, bool phys_space = false);

  // Apply deformation to data
  static void interp_cimg(CompositeImageType *data, VectorImageType *field, CompositeImageType *out,
                          bool use_nn = false, bool phys_space = false);

  // Take Jacobian of deformation field
  static void field_jacobian_det(VectorImageType *vec, ImageType *out);

  // Smooth an image in-place
  static void img_smooth(ImageType *src, ImageType *out, double sigma);
  static void vimg_smooth(VectorImageType *src, VectorImageType *out, double sigma);
  static void vimg_smooth(VectorImageType *src, VectorImageType *out, Vec sigmas);

  // Smooth a displacement field with a border of zeros around it
  static void vimg_smooth_withborder(VectorImageType *src, VectorImageType *trg, Vec sigma, int border_size);

  // Take gradient of an image
  static void image_gradient(ImageType *src, VectorImageType *grad);

  // Basic math
  static void vimg_add_in_place(VectorImageType *trg, VectorImageType *a);
  static void vimg_subtract_in_place(VectorImageType *trg, VectorImageType *a);
  static void vimg_scale_in_place(VectorImageType *trg, TFloat s);

  // compute trg = trg + s * a
  static void vimg_add_scaled_in_place(VectorImageType *trg, VectorImageType *a, TFloat s);

  static void vimg_scale(VectorImageType *src, TFloat s, VectorImageType *trg);
  static void vimg_multiply_in_place(VectorImageType *trg, ImageType *s);
  static void vimg_euclidean_inner_product(ImagePointer &trg, VectorImageType *a, VectorImageType *b);
  static TFloat vimg_euclidean_norm_sq(VectorImageType *trg);

  // Compute the range of the norm of a vector field
  static void vimg_norm_min_max(VectorImageType *image, ImagePointer &normsqr,
                      TFloat &min_norm, TFloat &max_norm);

  // Update a vector image to make its maxumum length equal to given value. The
  // second parameter is a working image that will return unnormalized lengths squared
  static void vimg_normalize_to_fixed_max_length(VectorImageType *trg,
                                                 ImagePointer &normsqr,
                                                 double max_displacement,
                                                 bool scale_down_only);

  // Scalar math
  static void img_subtract_in_place(ImagePointer &trg, ImageType *a);
  static void img_multiply_in_place(ImagePointer &trg, ImageType *a);
  static TFloat img_euclidean_norm_sq(ImageType *trg);
  static TFloat img_voxel_sum(ImageType *trg);
  static void img_min_max(ImageType *src, TFloat &out_min, TFloat &out_max);

  // Generate a kernel image for navier-stokes operator
  static void compute_navier_stokes_kernel(ImageType *kernel, double alpha, double gamma);

  // Downsample and upsample images (includes smoothing, use sparingly)
  static void img_downsample(ImageType *src, ImageType *trg, double factor);
  static void img_shrink(ImageType *src, ImageType *trg, int factor);
  static void img_resample_identity(ImageType *src, ImageBaseType *ref, ImageType *trg);
  static void vimg_resample_identity(VectorImageType *src, ImageBaseType *ref, VectorImageType *trg);

  // Threshold image
  static void img_threshold_in_place(ImageType *src, double lt, double up, double fore, double back);

  // Convert voxel-space warp to a physical space warp
  static void warp_voxel_to_physical(VectorImageType *src, ImageBaseType *ref_space, VectorImageType *trg);
  
  // Some IO methods
  typedef itk::ImageIOBase::IOComponentType IOComponentType;

  static IOComponentType img_read(const char *fn, ImagePointer &trg);
  static IOComponentType vimg_read(const char *fn, VectorImagePointer &trg);
  static IOComponentType cimg_read(const char *fn, CompositeImagePointer &trg);

  // Write scalar image, with optional output format specification
  static void img_write(ImageType *src, const char *fn,
                        IOComponentType comp = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE);

  // Write vector image, with optional output format specification
  static void vimg_write(VectorImageType *src, const char *fn,
                         IOComponentType comp = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE);

  // Write composite image, with optional output format specification
  static void cimg_write(CompositeImageType *src, const char *fn,
                         IOComponentType comp = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE);

  static void vfield_read(uint nt, const char *fnpat, VelocityField &v);

  static void vimg_copy(const VectorImageType *src, VectorImageType *trg);
  static void img_copy(const ImageType *src, ImageType *trg);

  // Compute a array from v
  void compute_semi_lagrangean_a();

  // Integrate forward tranform (phi_0_t)
  void integrate_phi_t0();
  void integrate_phi_t1();

protected:

  // A vector image for in-place interpolation operations
  VectorImagePointer vtmp;

};

template <class TFloat, uint VDim>
class LDDMMFFTInterface
{
public:
  typedef typename LDDMMData<TFloat, VDim>::ImageType ImageType;
  typedef typename LDDMMData<TFloat, VDim>::VectorImageType VectorImageType;
  typedef typename LDDMMData<TFloat, VDim>::Vec Vec;

  LDDMMFFTInterface(ImageType *ref);
  ~LDDMMFFTInterface();

  void convolution_fft(
    VectorImageType *img, ImageType *kernel_ft, bool inv_kernel, 
    VectorImageType *out);

private:

  // Size of the input array and allocated array (bigger, for in-place math)
  itk::Size<VDim> m_Size, m_Alloc;
  uint m_AllocSize, m_DataSize;

  // In-place data array
  double *m_Data;

  // FFT plan
  fftw_plan m_Plan, m_InvPlan;

};

// Class for iteratively computing the objective function
template<class TFloat, uint VDim>
class LDDMMImageMatchingObjective 
{
public:
  typedef LDDMMData<TFloat, VDim> LDDMM;
  typedef LDDMMFFTInterface<TFloat, VDim> FFT;

  LDDMMImageMatchingObjective(LDDMM &p);
  TFloat compute_objective_and_gradient(LDDMM &p);

  typename LDDMM::ImagePointer Jt0, Jt1, DetPhit1;
  typename LDDMM::VectorImagePointer GradJt0;
  FFT fft;
};




#endif
