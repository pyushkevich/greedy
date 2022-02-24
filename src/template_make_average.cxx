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

#include "GreedyAPI.h"
#include "MultiImageRegistrationHelper.h"
#include "CommandLineHelper.h"
#include "MultiComponentQuantileBasedNormalizationFilter.h"
#include "itkLaplacianSharpeningImageFilter.h"
#include <itkImageIOFactory.h>

using namespace std;

extern const char *GreedyVersionInfo;

int usage()
{
  printf(
        "greedy_template_average: tool for image averaging during template construction\n"
        "usage:\n"
        "  greedy_template_average [options]\n"
        "required options:\n"
        "  -d <2|3>                 : Image dimension\n"
        "  -i <inputs> <output>     : Input images and output intensity average. If a single\n"
        "                             input is supplied, only shape averaging will be done\n"
        "additional options:\n"
        "  -m <inputs> <output>     : Input mask and output average mask. Number of masks should\n"
        "                             match the number of images specified with '-i'\n"
        "  -w <transforms>          : A list of root-warps (greedy's -oroot) or affine\n"
        "                             transformations between the template and each\n"
        "                             input image used to push the tempate towards the average shape\n"
        "  -N <lp> <up> <lv> <uv>   : Apply normalization to input images when performing\n"
        "                             intensity averaging. Intensity quantiles lp to up will\n"
        "                             be normalized to range lv to uv. E.g., (0, .98, 0, 1.)\n"
        "                             will map 98th percentile of intensity to 1.0\n"
        "  -r <image>               : Reference space for the output image/mask. Normally, the inputs\n"
        "                             will all be in the same space, but if not or if you want to pad\n"
        "                             the average, this reference space can be supplied\n"
        "  -U <N>                   : Apply a sharpening operation to the template image (N times)\n"
        "  -T <lt> <ut> <iv> <ov>   : Threshold the result (pixels between lt and ut map to iv, others ov)\n"
        "  -M <value>               : Threshold the mask at value (default is 0.5)\n"
  );

  return -1;
}

struct Parameters {
  int dimension = 3;
  vector<string> fn_in_image, fn_in_mask, fn_in_transform;
  string fn_out_image, fn_out_mask, fn_reference;
  bool flag_normalize = false;
  double norm_param[4] = {0., .98, 0., 1.0};
  int warp_exponent = GreedyParameters().warp_exponent;
  double mask_threshold = 0.5;

  bool flag_threshold_image = false;
  double image_threshold_param[4] = {0., 0., 0., 0.};
  unsigned int n_sharpen_iter = 0;
};

template <typename TReal, unsigned int VDim>
class MakeAverage
{
public:
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef typename GreedyAPI::OFHelperType OFHelperType;
  typedef typename LDDMMType::CompositeImageType CompositeImageType;
  typedef typename LDDMMType::CompositeImagePointer CompositeImagePointer;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::ImagePointer ImagePointer;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::VectorImagePointer VectorImagePointer;
  typedef typename LDDMMType::ImageBaseType ImageBaseType;
  typedef typename LDDMMType::ImageBasePointer ImageBasePointer;

  /** Common code to read an image and optional mask */
  static CompositeImagePointer read_image(
      const string &fn, VectorImageType *phi_resample,
      bool normalize, const double norm_param[4])
  {
    // Read image
    CompositeImagePointer img_raw = LDDMMType::cimg_read(fn.c_str());

    // Perform normalization - this has to be done component by component
    if(normalize)
      {
      typedef MultiComponentQuantileBasedNormalizationFilter<
          CompositeImageType, CompositeImageType> NormalizeFilter;
      typename NormalizeFilter::Pointer fixed_binner = NormalizeFilter::New();
      fixed_binner->SetInput(img_raw);
      fixed_binner->SetLowerQuantile(norm_param[0]);
      fixed_binner->SetUpperQuantile(norm_param[1]);
      fixed_binner->SetLowerQuantileOutputValue(norm_param[2]);
      fixed_binner->SetUpperQuantileOutputValue(norm_param[3]);
      fixed_binner->Update();
      img_raw = fixed_binner->GetOutput();
      }

    // Transform to reference space
    if(phi_resample)
      {
      CompositeImagePointer img = LDDMMType::new_cimg(phi_resample, img_raw->GetNumberOfComponentsPerPixel());
      LDDMMType::interp_cimg(img_raw, phi_resample, img, false, true);
      return img;
      }
    else
      return img_raw;
  }

  static ImagePointer read_mask(const string &fn, VectorImageType *phi_resample)
  {
    // Read image
    ImagePointer img_raw = LDDMMType::img_read(fn.c_str());

    // Transform to reference space if needed
    if(phi_resample)
      {
      ImagePointer img = LDDMMType::new_img(phi_resample);
      LDDMMType::interp_img(img_raw, phi_resample, img, false, true);
      return img;
      }
    else
      return img_raw;
  }

  static VectorImagePointer read_svf(const string &fn, VectorImageType *phi_resample)
  {
    // Read image
    VectorImagePointer svf_raw = LDDMMType::vimg_read(fn.c_str());

    // Transform to reference space if needed
    if(phi_resample)
      {
      VectorImagePointer svf = LDDMMType::new_vimg(phi_resample);
      LDDMMType::interp_vimg(svf_raw, phi_resample, 0.0, svf, false, true);
      return svf;
      }
    else
      return svf_raw;
  }


  /** Main method */
  static void make_average(const Parameters &p)
  {
    // Are we doing masks?
    bool use_masks = p.fn_in_mask.size() > 0;

    // Load the optional reference image (stored as a transform to use interp_cimg)
    VectorImagePointer ref_phi;
    if(p.fn_reference.size())
      {
      ImagePointer ref = LDDMMType::img_read(p.fn_reference.c_str());
      ref_phi = LDDMMType::new_vimg(ref);
      }

    // Initialize the composite average with the first image
    CompositeImagePointer img_avg =
        read_image(p.fn_in_image.front(), ref_phi, p.flag_normalize, p.norm_param);

    // Optionally initialize the first mask
    ImagePointer msk_avg;

    // Set the reference space to the first image
    if(!ref_phi)
      ref_phi = LDDMMType::new_vimg(img_avg);

    // Read each of the remaining images and masks and add to the current
    for(unsigned int i = 1; i < p.fn_in_image.size(); i++)
      {
      CompositeImagePointer img_i =
          read_image(p.fn_in_image[i], ref_phi, p.flag_normalize, p.norm_param);
      LDDMMType::cimg_add_in_place(img_avg, img_i);
      }

    // Scale the image
    LDDMMType::cimg_scale_in_place(img_avg, 1.0 / p.fn_in_image.size());

    // Handle the masks likewise
    if(use_masks)
      {
      msk_avg = read_mask(p.fn_in_mask.front(), ref_phi);
      for(unsigned int i = 1; i < p.fn_in_image.size(); i++)
        {
        ImagePointer msk_i = read_mask(p.fn_in_mask[i], ref_phi);
        LDDMMType::img_add_in_place(msk_avg, msk_i);
        }
      LDDMMType::img_scale_in_place(msk_avg, 1.0 / p.fn_in_image.size());
      }

    // At this stage, we have created an average image and average mask, and we
    // are ready to apply an unwarping operation to them
    if(p.fn_in_transform.size())
      {
      VectorImagePointer shape_unwarp;

      // Read the first one to determine if it's a stationary velocity field or a matrix
      bool is_warp = itk::ImageIOFactory::CreateImageIO(
                       p.fn_in_transform.front().c_str(), itk::IOFileModeEnum::ReadMode);
      if(is_warp)
        {
        // Compute the average stationary velocity field
        VectorImagePointer svf_avg = read_svf(p.fn_in_transform.front(), ref_phi);
        for(unsigned int i = 1; i < p.fn_in_transform.size(); i++)
          {
          VectorImagePointer svf_i = read_svf(p.fn_in_transform[i], ref_phi);
          LDDMMType::vimg_add_in_place(svf_avg, svf_i);
          }
        LDDMMType::vimg_scale_in_place(svf_avg, 1.0 / p.fn_in_transform.size());

        // Exponentiate the average field
        VectorImagePointer phi = LDDMMType::new_vimg(svf_avg);
        VectorImagePointer exp_work = LDDMMType::new_vimg(svf_avg);

        OFHelperType::PhysicalWarpToVoxelWarp(svf_avg, svf_avg, svf_avg);
        LDDMMType::vimg_exp(svf_avg, phi, exp_work, p.warp_exponent, -1.0);
        OFHelperType::VoxelWarpToPhysicalWarp(phi, svf_avg, svf_avg);

        // Apply the field to the image and mask
        shape_unwarp = svf_avg;
        }
      else
        {
        // The transforms are matrices and should be read, square rooted a bunch
        // of times, averaged, inverted, and then exponentiated
        GreedyAPI api;
        TransformSpec ts(p.fn_in_transform.front().c_str(), -pow(2.0, p.warp_exponent));
        vnl_matrix<double> mat_avg = api.ReadAffineMatrixViaCache(ts);

        for(unsigned int i = 1; i < p.fn_in_transform.size(); i++)
          {
          TransformSpec ts(p.fn_in_transform[i].c_str(), -pow(2.0, p.warp_exponent));
          vnl_matrix<double> mat_i = api.ReadAffineMatrixViaCache(ts);
          cout << "mat " << i << mat_i << endl;
          mat_avg += mat_i;
          }

        // This is the average root-matrix
        mat_avg *= 1.0 / p.fn_in_transform.size();

        // This is the inversion/exponentiation
        mat_avg = vnl_matrix_inverse<double>(mat_avg).as_matrix();
        for(int j = 0; j < p.warp_exponent; j++)
          mat_avg = mat_avg * mat_avg;

        cout << "Matrix is " << mat_avg << endl;

        // Finally, we need to apply this transformation to the image and mask
        shape_unwarp = LDDMMType::new_vimg(ref_phi);
        GreedyAPI::MapRASAffineToPhysicalWarp(mat_avg, shape_unwarp);
        }

      // Apply the unwarp transformation to the image and mask
      CompositeImagePointer img_unwarp = LDDMMType::new_cimg(shape_unwarp, img_avg->GetNumberOfComponentsPerPixel());
      LDDMMType::interp_cimg(img_avg, shape_unwarp, img_unwarp, false, true);
      img_avg = img_unwarp;

      if(use_masks)
        {
        ImagePointer msk_unwarp = LDDMMType::new_img(shape_unwarp);
        LDDMMType::interp_img(msk_avg, shape_unwarp, msk_unwarp, false, true);
        msk_avg = msk_unwarp;
        }
      }

    // Final transformations applied to the template. The template can be thresholded
    // or it can be sharpened. The mask should be thresholded
    if(p.n_sharpen_iter > 0)
      {
      ImagePointer i_comp = LDDMMType::new_img(img_avg);
      for(unsigned int c = 0; c < img_avg->GetNumberOfComponentsPerPixel(); c++)
        {
        LDDMMType::cimg_extract_component(img_avg, i_comp, c);
        for(unsigned int k = 0; k < p.n_sharpen_iter; k++)
          {
          typedef itk::LaplacianSharpeningImageFilter<ImageType, ImageType> SharpFilter;
          typename SharpFilter::Pointer sharp = SharpFilter::New();
          sharp->SetInput(i_comp);
          sharp->Update();
          i_comp = sharp->GetOutput();
          }
        LDDMMType::cimg_update_component(img_avg, i_comp, c);
        }

      }

    // Apply thresholding if requested
    if(p.flag_threshold_image)
      {
      LDDMMType::cimg_threshold_in_place(
            img_avg,
            p.image_threshold_param[0], p.image_threshold_param[1],
            p.image_threshold_param[2], p.image_threshold_param[3]);
      }

    // Save the image
    LDDMMType::cimg_write(img_avg, p.fn_out_image.c_str());

    // Threshold the mask
    if(use_masks)
      {
      LDDMMType::img_threshold_in_place(
            msk_avg, p.mask_threshold, numeric_limits<TReal>::infinity(), 1.0, 0.0);

      LDDMMType::img_write(msk_avg, p.fn_out_mask.c_str());
      }

  }
};




int main(int argc, char *argv[])
{
  if(argc < 2)
    return usage();

  try
  {
    Parameters p;
    CommandLineHelper cl(argc, argv);
    while(!cl.is_at_end())
      {
      // Read the next command
      std::string cmd = cl.read_command();

      // Parse generic commands
      if(cmd == "-version")
        {
        std::cout << GreedyVersionInfo << std::endl;
        return 0;
        }
      else if(cmd == "-h" || cmd == "-help" || cmd == "--help")
        {
        return usage();
        }
      else if(cmd == "-d")
        {
        p.dimension = cl.read_integer();
        }
      else if(cmd == "-i")
        {
        unsigned int n = cl.command_arg_count();
        for(unsigned int i = 0; i < n-1; i++)
          p.fn_in_image.push_back(cl.read_existing_filename());
        p.fn_out_image = cl.read_output_filename();
        }
      else if(cmd == "-m")
        {
        unsigned int n = cl.command_arg_count();
        for(unsigned int i = 0; i < n-1; i++)
          p.fn_in_mask.push_back(cl.read_existing_filename());
        p.fn_out_mask = cl.read_output_filename();
        }
      else if(cmd == "-w")
        {
        unsigned int n = cl.command_arg_count();
        for(unsigned int i = 0; i < n; i++)
          p.fn_in_transform.push_back(cl.read_existing_filename());
        }
      else if(cmd == "-w")
        {
        p.fn_reference = cl.read_existing_filename();
        }
      else if(cmd == "-N")
        {
        p.flag_normalize = true;
        if(cl.command_arg_count() == 4)
          for(unsigned int i = 0; i < 4; i++)
            p.norm_param[i] = cl.read_double();
        }
      else if(cmd == "-U")
        {
        p.n_sharpen_iter = cl.read_integer();
        }
      else if(cmd == "-T")
        {
        p.flag_threshold_image = true;
        for(unsigned int i = 0; i < 4; i++)
          p.image_threshold_param[i] = cl.read_double();
        }
      else if(cmd == "-M")
        {
        p.mask_threshold = cl.read_double();
        }
      else
        {
        throw GreedyException("Unrecognized option %s", cmd.c_str());
        }
      }

    // Check parameter integrity
    unsigned int n_img = p.fn_in_image.size();
    if(n_img == 0)
      throw GreedyException("No input images were specified");
    if(p.fn_in_mask.size() && p.fn_in_mask.size() != n_img)
      throw GreedyException("Number of input masks (-m) does not match number of input images");
    if(p.fn_reference.size() && n_img > 1 && p.fn_reference.size() != n_img)
      throw GreedyException("Number of transforms (-w) does not match number of input images");

    if(p.dimension == 2)
      MakeAverage<double, 2>::make_average(p);
    else if(p.dimension == 3)
      MakeAverage<double, 3>::make_average(p);
    else
      throw GreedyException("Dimension parameter (-d) must be 2 or 3");

    return 0;
  }
  catch (GreedyException &exc)
  {
    cerr << "Error: " << exc.what() << endl;
    return -1;
  }
}
