#include "MultiImageRegistrationHelper.h"
#include <string>
#include <fstream>
#include <vector>

#include "lddmm_data.h"

#include "CommandLineHelper.h"

using namespace std;

struct MACFParameters
{
  string fnReference, fnPsiPattern, fnWeightPattern, fnIds, fnOutPhiPattern;
  string fnGrayPattern, fnOutIterTemplatePattern;
  string fnTransportedWeightsPattern, fnOutTransportedWeightsPattern;
  string fnGlobalMask, fnInitialRootPhiInvPattern;
  string fnOutIterDeltaSq;

  int exponent;
  double sigma1, sigma2;
  double epsilon;
  int dumpfreq;

  // Number of iterations per level
  vector<int> n_iter;


  MACFParameters()
    {
    exponent = 6;
    sigma1 = sqrt(3.0);
    sigma2 = sqrt(0.5);
    epsilon = 0.25;
    dumpfreq = 10;

    n_iter.push_back(100);
    n_iter.push_back(100);
    }
};

int usage()
{
  MACFParameters p;

  printf("macf_optimize: optimization routine for CVPR 2012 MACF paper\n");
  printf("usage:\n");
  printf("  macf_optimize [options]\n");
  printf("required options:\n");
  printf("  -d <2|3>             : number of dimensions\n");
  printf("  -ids <file>          : text file of ids\n");
  printf("  -ref <image>         : reference image\n");
  printf("  -psi <pattern_2s>    : pattern of root warps of psi\n");
  printf("  -wgt <pattern_2s>    : pattern of weight images\n");
  printf("  -o <pattern_1s>      : output phi pattern\n");
  printf("additional options:\n");
  printf("  -n <value>           : number of iterations (def = 100x100)\n");
  printf("  -eps <value>         : step size (def = %f)\n",p.epsilon);
  printf("  -exp <value>         : exponent for scaling and squaring (def = %d)\n",p.exponent);
  printf("  -s <sigma1> <sigma2> : smoothing in voxels (def = %f, %f)\n",p.sigma1,p.sigma2);
  printf("  -gm <image>          : global mask image *** BROKEN ***\n");
  printf("  -phi <pattern_1s>    : initial root-phi-inv (starting point for optimization)\n");
  printf("  -img <pattern_1s>    : pattern of grayscale images (for visualizing registration)\n");
  printf("  -otemp <pattern_2d>  : pattern for saving templates at each iteration\n");
  printf("  -odelta <pattern_2d> : pattern for saving delta^2 at each iteration\n");
  printf("  -freq <value>        : frequency with which per-iteration images are saved (def = %d)\n", p.dumpfreq);
  printf("transported weights:\n");
  printf("  -owtm <pattern_2s>   : write the weights transported to moving space to files\n");
  printf("  -wtm <pattern_2s>    : read transported weights (saves a lot of time upfront)\n");
  printf("patterns:\n");
  printf("  pattern_1s           : of form blah_%%1_blah.nii.gz\n");
  printf("  pattern_2s           : of form blah_%%1_blah_%%2_blah.nii.gz (1:fixed, 2:moving)\n");
  printf("  pattern_1d           : of form blah_%%03d_blah.nii.gz\n");
  return -1;
}

string string_replace(const string &str, const string &pattern, const string &replacement)
{
  string result = str;
  string::size_type pos = 0u;
  while((pos = result.find(pattern, pos)) != std::string::npos)
    {
     result.replace(pos, pattern.length(), replacement);
     pos += replacement.length();
    }
  return result;
}

string exp_pattern_1(const string &pattern, const string &id)
{
  return string_replace(pattern, "%1", id);
}

string exp_pattern_2(const string &pattern, const string &id_fixed, const string &id_moving)
{
  return string_replace(string_replace(pattern, "%1", id_fixed), "%2", id_moving);
}

template <typename TFloat, unsigned int VDim>
class MACFWorker
{
public:

  typedef LDDMMData<TFloat, VDim> LDDMMType;
  typedef MultiImageOpticalFlowHelper<TFloat, VDim> OFHelperType;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::MatrixImageType MatrixImageType;
  typedef typename VectorImageType::Pointer VectorImagePointer;
  typedef typename ImageType::Pointer ImagePointer;
  typedef typename MatrixImageType::Pointer MatrixImagePointer;

  void ReadImages()
    {
    // Read the list of ids
    ifstream iff(m_Param.fnIds.c_str());
    string id;
    while(iff >> id)
      m_Ids.push_back(id);

    // Allocate the main storage
    m_Size = m_Ids.size();

    // Create the levels
    m_Levels.resize(m_Param.n_iter.size());

    // Initialize the images for each level
    typename vector<LevelData>::reverse_iterator lev, uplev;
    int factor = 1;
    for(lev= m_Levels.rbegin(); lev!= m_Levels.rend(); uplev = lev, ++lev, factor *= 2)
      {
      // Read or downsample the reference image
      if(lev == m_Levels.rbegin())
        lev->reference = LDDMMType::img_read(m_Param.fnReference.c_str());
      else
        lev->reference = LDDMMType::img_downsample(uplev->reference, 2);

      // Some working images
      lev->work = LDDMMType::alloc_vimg(lev->reference);
      lev->lev_psi_root = LDDMMType::alloc_vimg(lev->reference);
      lev->scalar_work = LDDMMType::alloc_img(lev->reference);
      lev->factor = factor;

      // Create all the image data
      lev->img_data.resize(m_Size);
      for(int i = 0; i < m_Size; i++)
        {
        lev->img_data[i].u = LDDMMType::alloc_vimg(lev->reference);
        lev->img_data[i].u_root = LDDMMType::alloc_vimg(lev->reference);
        lev->img_data[i].grad_u = LDDMMType::alloc_vimg(lev->reference);
        lev->img_data[i].delta = LDDMMType::alloc_vimg(lev->reference);
        lev->img_data[i].pair_data.resize(m_Size);
        }
      }

    // Iterate over all the pairwise registrations
    MatrixImagePointer jac = LDDMMType::alloc_mimg(m_Levels.back().reference);
    MatrixImagePointer jac_work = LDDMMType::alloc_mimg(m_Levels.back().reference);

    // Create all the pair data
    for(int i = 0; i < m_Size; i++)
      {
      for(int j = 0; j < m_Size; j++)
        {
        if(i != j)
          {
          // Start with the last leve
          lev = m_Levels.rbegin();

          // Reference the current pair data
          PairData &pd = lev->img_data[i].pair_data[j];

          // Read the psi root image
          string fn = exp_pattern_2(m_Param.fnPsiPattern, m_Ids[i], m_Ids[j]);
          VectorImagePointer psi_root = LDDMMType::vimg_read(fn.c_str());
          OFHelperType::PhysicalWarpToVoxelWarp(psi_root, psi_root, psi_root);

          // Load the weight image
          fn = exp_pattern_2(m_Param.fnWeightPattern, m_Ids[i], m_Ids[j]);
          pd.wgt_fixed = LDDMMType::img_read(fn.c_str());

          // Integrate the psi image forward
          pd.psi_forward = LDDMMType::alloc_vimg(lev->reference);
          LDDMMType::vimg_exp(psi_root, pd.psi_forward, lev->work, m_Param.exponent, 1.0);

          // Did the user supply the transported weights?
          if(m_Param.fnTransportedWeightsPattern.size())
            {
            // Read the transported weights
            fn = exp_pattern_2(m_Param.fnTransportedWeightsPattern, m_Ids[i], m_Ids[j]);
            pd.wgt_moving = LDDMMType::img_read(fn.c_str());

            // Simply integrate the velocity backwards to get inverse psi
            pd.psi_inverse = LDDMMType::alloc_vimg(lev->reference);
            LDDMMType::vimg_exp(psi_root, pd.psi_inverse, lev->work, m_Param.exponent, -1.0);
            }
          else
            {
            // Integrate the psi image backward with jacobian 
            pd.psi_inverse = LDDMMType::alloc_vimg(lev->reference);
            LDDMMType::vimg_exp_with_jacobian(
              psi_root, pd.psi_inverse, lev->work, jac, jac_work, m_Param.exponent, -1.0);

            // Compute Jacobian determinant
            LDDMMType::mimg_det(jac, 1.0, lev->scalar_work);

            // Warp the weight by the inverse psi and scale by the determinant
            pd.wgt_moving = LDDMMType::alloc_img(lev->reference);
            LDDMMType::interp_img( pd.wgt_fixed, pd.psi_inverse, pd.wgt_moving, false, false, 0);
            LDDMMType::img_multiply_in_place(pd.wgt_moving, lev->scalar_work);

            // Save the transported weights if requested
            if(m_Param.fnOutTransportedWeightsPattern.size())
              {
              fn = exp_pattern_2(m_Param.fnOutTransportedWeightsPattern, m_Ids[i], m_Ids[j]);
              LDDMMType::img_write(pd.wgt_moving, fn.c_str());
              }
            }

          // Downsample to the other levels
          uplev = lev; lev++;
          for(; lev != m_Levels.rend(); uplev = lev, ++lev)
            {
            // Reference the current pair data
            PairData &pd = lev->img_data[i].pair_data[j];
            PairData &up_pd = uplev->img_data[i].pair_data[j];

            // Downsample the psi root
            LDDMMType::vimg_resample_identity(psi_root, lev->reference, lev->lev_psi_root);

            // Exponentiate forward
            pd.psi_forward = LDDMMType::alloc_vimg(lev->reference);
            LDDMMType::vimg_exp(lev->lev_psi_root, pd.psi_forward, lev->work, m_Param.exponent, 1.0 / lev->factor);

            // Exponentiate backward
            pd.psi_inverse = LDDMMType::alloc_vimg(lev->reference);
            LDDMMType::vimg_exp(lev->lev_psi_root, pd.psi_inverse, lev->work, m_Param.exponent, -1.0 / lev->factor);

            // Downsample the weight images from previous level
            pd.wgt_fixed = LDDMMType::img_downsample(up_pd.wgt_fixed, 2);
            pd.wgt_moving = LDDMMType::img_downsample(up_pd.wgt_moving, 2);
            }

          cout << "." << flush;
          }
        }

      // Read the optional grayscale images
      if(m_Param.fnGrayPattern.size())
        {
        // Start with the last level
        lev = m_Levels.rbegin();

        string fn = exp_pattern_1(m_Param.fnOutTransportedWeightsPattern, m_Ids[i]);
        lev->img_data[i].img_gray = LDDMMType::img_read(fn.c_str());

        // Downsample to the other levels
        uplev = lev; lev++;
        for(; lev != m_Levels.rend(); uplev = lev, ++lev)
          lev->img_data[i].img_gray = LDDMMType::img_downsample(uplev->img_data[i].img_gray, 2);
        }

      // Read the optional initial fields
      if(m_Param.fnInitialRootPhiInvPattern.size())
        {
        // We can go straight to the terminal level
        LevelData &first_lev = m_Levels.front();
        ImageData &id = first_lev.img_data[i];

        string fn = exp_pattern_1(m_Param.fnInitialRootPhiInvPattern, m_Ids[i]);
        VectorImagePointer u_root_full = LDDMMType::vimg_read(fn.c_str());

        // Downsample to the current level and exponentate
        OFHelperType::PhysicalWarpToVoxelWarp(u_root_full, u_root_full, u_root_full);
        LDDMMType::vimg_resample_identity(u_root_full, first_lev.reference, id.u_root);
        LDDMMType::vimg_scale_in_place(id.u_root, 1.0 / first_lev.factor);
        LDDMMType::vimg_exp(id.u_root, id.u, first_lev.work, m_Param.exponent, 1.0);
        }

      cout << "." << endl;
      }

    // Read the global masks
    if(m_Param.fnGlobalMask.size())
      {
      // Start with the last level
      lev = m_Levels.rbegin();
      lev->global_mask = LDDMMType::img_read(m_Param.fnGlobalMask.c_str());

      // Downsample to the other levels
      uplev = lev; lev++;
      for(; lev != m_Levels.rend(); uplev = lev, ++lev)
        lev->global_mask = LDDMMType::img_downsample(uplev->global_mask, 2);
      }
    }

  double ComputeDeltasAndObjective(int level)
    {
    double total_error = 0;

    // Reference to the level data
    LevelData &lev = m_Levels[level];

    // Compute the deltas and the objective
    for(int i = 0; i < m_Size; i++)
      {
      // Get a reference to the i-th image data
      ImageData &id = lev.img_data[i];

      // Set the delta to the current u_i
      LDDMMType::vimg_copy(id.u, id.delta);

      // Add all the differences
      for(int j = 0; j < m_Size; j++)
        {
        if(j != i)
          {
          PairData &pd = id.pair_data[j];
          LDDMMType::interp_vimg(lev.img_data[j].u, pd.psi_forward, 1.0, lev.work);
          LDDMMType::vimg_add_in_place(lev.work, pd.psi_forward);
          LDDMMType::vimg_multiply_in_place(lev.work, pd.wgt_fixed);
          LDDMMType::vimg_subtract_in_place(id.delta, lev.work);
          }
        }

      // Compute the norm of the delta
      if(lev.global_mask)
        {
        LDDMMType::vimg_euclidean_inner_product(lev.scalar_work, id.delta, id.delta);
        LDDMMType::img_multiply_in_place(lev.scalar_work, lev.global_mask);
        id.norm_delta = LDDMMType::img_voxel_sum(lev.scalar_work);
        }
      else
        {
        id.norm_delta = LDDMMType::vimg_euclidean_norm_sq(id.delta);
        }

      // Add to the total error
      total_error += id.norm_delta;
      }

    // Extract the average error per pixel per image
    total_error /= m_Size * lev.reference->GetBufferedRegion().GetNumberOfPixels();

    return total_error;
    }

  void BuildTemplate(int level, int iter)
    {
    // Reference to the level data
    LevelData &lev = m_Levels[level];

    // We need a couple of images
    ImagePointer templ = LDDMMType::alloc_img(lev.reference);
    VectorImagePointer phi_exp = LDDMMType::alloc_vimg(lev.reference);

    // Iterate over the images
    for(int i = 0; i < m_Size; i++)
      {
      // Get a reference to the i-th image data
      ImageData &id = lev.img_data[i];

      // Compute the deformation that warps i-th image into template space
      LDDMMType::vimg_exp(id.u_root, phi_exp, lev.work, m_Param.exponent, -1.0);

      // Apply that warp to the gray image
      LDDMMType::interp_img(id.img_gray, phi_exp, lev.scalar_work, false, false, 0);

      // Add the image to the template
      LDDMMType::img_add_in_place(templ, lev.scalar_work);
      }

    // Scale the template by the number of images
    LDDMMType::img_scale_in_place(templ, 1.0 / m_Size);

    // Write the template
    char fn[1024];
    sprintf(fn, m_Param.fnOutIterTemplatePattern.c_str(), level, iter);
    LDDMMType::img_write(templ, fn);
    }

  void BuildErrorMap(int level, int iter)
    {
    // Reference to the level data
    LevelData &lev = m_Levels[level];

    // We need a couple of images
    ImagePointer delta_i = LDDMMType::alloc_img(lev.reference);
    ImagePointer mean_delta = LDDMMType::alloc_img(lev.reference);
    VectorImagePointer phi_exp = LDDMMType::alloc_vimg(lev.reference);

    // Iterate over the images
    for(int i = 0; i < m_Size; i++)
      {
      // Get a reference to the i-th image data
      ImageData &id = lev.img_data[i];

      // Get the squared norm of delta
      LDDMMType::vimg_euclidean_inner_product(delta_i, id.delta, id.delta);

      // Compute the deformation that warps i-th image into template space
      LDDMMType::vimg_exp(id.u_root, phi_exp, lev.work, m_Param.exponent, -1.0);

      // Apply that warp to the gray image
      LDDMMType::interp_img(delta_i, phi_exp, lev.scalar_work, false, false, 0);

      // Add the image to the template
      LDDMMType::img_add_in_place(mean_delta, lev.scalar_work);
      }

    // Scale the template by the number of images
    LDDMMType::img_scale_in_place(mean_delta, 1.0 / m_Size);

    // Write the template
    char fn[1024];
    sprintf(fn, m_Param.fnOutIterDeltaSq.c_str(), level, iter);
    LDDMMType::img_write(mean_delta, fn);
    }

  void ComputeGradientAndUpdate(int level)
    {
    // Reference to the level data
    LevelData &lev = m_Levels[level];

    double global_max_norm = 0.0;

    // Compute gradients and their norms
    for(int m = 0; m < m_Size; m++)
      {
      // Get a reference to the i-th image data
      ImageData &id_m = lev.img_data[m];

      // Start by adding the delta
      LDDMMType::vimg_copy(id_m.delta, id_m.grad_u);

      // Subtract each of the deltas warped into moving space
      for(int j = 0; j < m_Size; j++)
        {
        if(m != j)
          {
          // Get a reference to the i-th image data
          ImageData &id_j = lev.img_data[j];

          PairData &pd = id_j.pair_data[m];
          LDDMMType::interp_vimg(id_j.delta, pd.psi_inverse, 1.0, lev.work);
          LDDMMType::vimg_multiply_in_place(lev.work, pd.wgt_moving);
          LDDMMType::vimg_subtract_in_place(id_m.grad_u, lev.work);
          }
        }

      // Multiply by the mask
      if(lev.global_mask)
        LDDMMType::vimg_multiply_in_place(id_m.grad_u, lev.global_mask);

      // Smooth the gradient 
      LDDMMType::vimg_smooth_withborder(id_m.grad_u, lev.work, m_Param.sigma1, 1);

      // Compute the norm of the gradient
      TFloat norm_min, norm_max;
      LDDMMType::vimg_norm_min_max(id_m.grad_u, lev.scalar_work, norm_min, norm_max);
      if(norm_max > global_max_norm)
        global_max_norm = norm_max;
      }

    // Compute the scaling factor
    double scale = 1.0 / (2 << m_Param.exponent);
    if(global_max_norm > m_Param.epsilon)
      scale = scale * m_Param.epsilon / global_max_norm;

    // Scale everything down by the max norm and smooth again
    for(int m = 0; m < m_Size; m++)
      {
      // Get a reference to the i-th image data
      ImageData &id_m = lev.img_data[m];

      // Compute the updated root warp
      LDDMMType::vimg_copy(id_m.u_root, lev.work);
      LDDMMType::vimg_add_scaled_in_place(lev.work, id_m.grad_u, -scale);
      LDDMMType::vimg_smooth_withborder(lev.work, id_m.u_root, m_Param.sigma2, 1);

      // Exponentiate the root warps
      LDDMMType::vimg_exp(id_m.u_root, id_m.u, lev.work, m_Param.exponent, 1.0);
      }
    }

  void WriteResults()
    {
    // Get the last level
    LevelData &lev = m_Levels.back();

    for(int i = 0; i < m_Size; i++)
      {
      // Map the warp back into physical units
      OFHelperType::VoxelWarpToPhysicalWarp(lev.img_data[i].u_root, lev.reference, lev.work);
      string fn = exp_pattern_1(m_Param.fnOutPhiPattern, m_Ids[i]);
      LDDMMType::vimg_write(lev.work, fn.c_str()); 
      }
    }

  void UpsampleWarps(int level)
    {
    LevelData &src_lev = m_Levels[level-1];
    LevelData &trg_lev = m_Levels[level];

    for(int i = 0; i < m_Size; i++)
      {
      ImageData &id_src = src_lev.img_data[i];
      ImageData &id_trg = trg_lev.img_data[i];
      LDDMMType::vimg_resample_identity(id_src.u_root, trg_lev.reference, id_trg.u_root);
      LDDMMType::vimg_scale_in_place(id_trg.u_root, 2.0);
      LDDMMType::vimg_exp(id_trg.u_root, id_trg.u, trg_lev.work, m_Param.exponent, 1.0);
      }
    }

  bool DumpThisIter(int level, int iter)
    {
    return (iter % m_Param.dumpfreq) == 0 || (iter == m_Param.n_iter.size() - 1);
    }

  void Run()
    {
    // Read the images into the datastructure
    ReadImages();
    printf("Read images for %d ids\n", m_Size);

    // Iterate
    for(int ilev = 0; ilev < m_Param.n_iter.size(); ilev++)
      {
      // Upsample warps from previous level
      if(ilev > 0)
        UpsampleWarps(ilev);

      // Gradient descent for this level
      for(int iter = 0; iter < m_Param.n_iter[ilev]; iter++)
        {
        // Compute the objective and deltas
        double total_error = ComputeDeltasAndObjective(ilev);
        printf("Level %d, Iter %04d:   Total Error: %12.4f\n", ilev, iter, total_error);

        // Write the iteration template
        if(m_Param.fnGrayPattern.size() && m_Param.fnOutIterTemplatePattern.size() && DumpThisIter(ilev, iter))
          BuildTemplate(ilev, iter);

        // Write the iteration template
        if(m_Param.fnOutIterDeltaSq.size() && DumpThisIter(ilev, iter))
          BuildErrorMap(ilev, iter);

        // Compute the gradients 
        ComputeGradientAndUpdate(ilev);
        }
      }

    // Write out final warps
    WriteResults();
    }

  MACFWorker(const MACFParameters &param) 
    : m_Param(param) {}

protected:

  struct PairData 
    {
    VectorImagePointer psi_forward, psi_inverse;
    ImagePointer wgt_fixed, wgt_moving;
    };

  struct ImageData
    {
    vector<PairData> pair_data;
    VectorImagePointer u, u_root, grad_u;
    VectorImagePointer delta;
    ImagePointer img_gray;
    double norm_delta;
    };

  struct LevelData
    {
    vector<ImageData> img_data;
    VectorImagePointer work, lev_psi_root;
    ImagePointer scalar_work;
    ImagePointer reference;
    ImagePointer global_mask;
    int factor;
    };

  vector<LevelData> m_Levels;
  MACFParameters m_Param;
  vector<string> m_Ids;
  int m_Size;
};

int main(int argc, char *argv[])
{
  MACFParameters param;
  int dim = 2;

  if(argc < 2)
    return usage();

  CommandLineHelper cl(argc, argv);
  while(!cl.is_at_end())
    {
    // Read the next command
    std::string arg = cl.read_command();

    if(arg == "-d")
      {
      dim = cl.read_integer();
      }
    else if(arg == "-ids")
      {
      param.fnIds = cl.read_existing_filename();
      }
    else if(arg == "-ref")
      {
      param.fnReference = cl.read_existing_filename();
      }
    else if(arg == "-psi")
      {
      param.fnPsiPattern = cl.read_string();
      }
    else if(arg == "-wgt")
      {
      param.fnWeightPattern = cl.read_string();
      }
    else if(arg == "-o")
      {
      param.fnOutPhiPattern = cl.read_string();
      }
    else if(arg == "-img")
      {
      param.fnGrayPattern = cl.read_string();
      }
    else if(arg == "-wtm")
      {
      param.fnTransportedWeightsPattern = cl.read_string();
      }
    else if(arg == "-owtm")
      {
      param.fnOutTransportedWeightsPattern = cl.read_string();
      }
    else if(arg == "-gm")
      {
      param.fnGlobalMask = cl.read_existing_filename();
      }
    else if(arg == "-phi")
      {
      param.fnInitialRootPhiInvPattern = cl.read_string();
      }
    else if(arg == "-exp")
      {
      param.exponent = cl.read_integer();
      }
    else if(arg == "-eps")
      {
      param.epsilon = cl.read_double();
      }
    else if(arg == "-odelta")
      {
      param.fnOutIterDeltaSq = cl.read_string();
      }
    else if(arg == "-freq")
      {
      param.dumpfreq = cl.read_integer();
      }
    else if(arg == "-s")
      {
      param.sigma1 = cl.read_double();
      param.sigma2 = cl.read_double();
      }
    else if(arg == "-n")
      {
      param.n_iter = cl.read_int_vector();
      }
    else if(arg == "-otemp")
      {
      param.fnOutIterTemplatePattern = cl.read_string();
      }
    else
      {
      printf("Unknown parameter %s\n", arg.c_str());
      return -1;
      }
    }

  if(dim == 2)
    {
    MACFWorker<float, 2> worker(param);
    worker.Run();
    }
  else if(dim == 3)
    {
    MACFWorker<float, 3> worker(param);
    worker.Run();
    }
}
