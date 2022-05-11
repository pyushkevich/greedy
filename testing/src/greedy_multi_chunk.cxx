#include <GreedyAPI.h>
#include <CommandLineHelper.h>
#include <itksys/SystemTools.hxx>
#include <itkMatrixOffsetTransformBase.h>
#include <GreedyException.h>
#include <lddmm_data.h>
#include <itkImageFileReader.h>
#include <itkBinaryThresholdImageFilter.hxx>
#include <MultiImageRegistrationHelper.h>
#include <AffineCostFunctions.h>
#include <AffineTransformUtilities.h>
#include <vnl/algo/vnl_lbfgs.h>


struct ChunkGreedyParameters
{
  std::string fn_chunk_mask;
  std::string fn_output_pattern, fn_output_inv_pattern, fn_output_root_pattern;
  std::string fn_init_tran_pattern;
  double reg_weight = 0.01;
};

int usage()
{
  printf("chunk_greedy: Multi-chunk registration with GreedyReg\n");
  printf("usage: \n");
  printf("  chunk_greedy <options> <greedy_options> \n");
  printf("options for chunk_greedy: \n");
  printf("  -cm <file>      : Chunk mask, each chunk assigned a different label\n");
  printf("  -wreg <value>   : Regularization term weight (default: 0.01)\n");
  printf("  -o <pattern>    : Like -o in greedy, but a printf pattern \n");
  printf("  -it <pattern>   : Like -it in greedy, but a printf pattern\n");
  printf("  -oinv <pattern> : Like -oinv in greedy, but a printf pattern\n");
  printf("main greedy options accepted: \n");
  printf("  -d, -i, -m, -n, -a, -dof, -bg, -ia, -wncc-mask-dilate, -search");
  printf("greedy options modified to accept printf-like pattern (e.g., test%%02d.mat): \n");
  printf("  -o, -oinv, -oroot, -it");
  return -1;
}

template <typename TImage>
std::vector<typename TImage::PixelType>
get_unique_labels(TImage *label_image, unsigned int max_allowed = 31, unsigned int max_value = 31)
{
  // Scan the unique labels in the image
  typedef typename TImage::PixelType PixelType;
  std::set<PixelType> label_set;
  PixelType *labels = label_image->GetBufferPointer();
  int n_pixels = label_image->GetPixelContainer()->Size();

  // Get the list of unique pixels
  PixelType last_pixel = 0;
  for(int j = 0; j < n_pixels; j++)
    {
    PixelType pixel = labels[j];
    if(last_pixel != pixel || j == 0)
      {
      label_set.insert(pixel);
      last_pixel = pixel;
      if(label_set.size() > max_allowed)
        throw GreedyException("Chunk mask has too many labels");
      if(pixel > max_value)
        throw GreedyException("Chunk mask has label greater than allowed maximum ", max_value);
      }
    }

  // Turn this set into an array
  std::vector<PixelType> label_array(label_set.begin(), label_set.end());
  return label_array;
}

// An 'assembly' for each registration problem
template <unsigned int VDim, typename TReal=double>
struct MultiChunkAffineAssembly
{
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCF;

  GreedyAPI api;
  typename GreedyAPI::OFHelperType helper;
  GreedyParameters gp;
  AbstractAffineCF *acf = nullptr;
  typename GreedyAPI::LinearTransformType::Pointer tLevel;
  typename GreedyAPI::ImagePointer mask;
  vnl_matrix<double> Q_physical;
};

std::string ssprintf(const char *format, ...)
{
  if(format && strlen(format))
    {
    char buffer[4096];
    va_list args;
    va_start (args, format);
    vsprintf (buffer, format, args);
    va_end (args);
    return std::string(buffer);
    }
  else
    return std::string();
}

template <unsigned int VDim, typename TReal=double>
void initialize_assemblies(ChunkGreedyParameters cgp, GreedyParameters gp,
                           std::map<short, MultiChunkAffineAssembly<VDim, TReal> > &label_data,
                           std::vector<short> &chunk_labels,
                           typename itk::Image<short, VDim>::Pointer &chunk_mask)
{
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef MultiChunkAffineAssembly<VDim, TReal> Assembly;

  // Read the chunked mask as an image of shorts
  typedef itk::Image<short, VDim> ChunkMaskImageType;
  typedef itk::ImageFileReader<ChunkMaskImageType> ChunkMaskReader;
  typename ChunkMaskReader::Pointer cmreader = ChunkMaskReader::New();
  cmreader->SetFileName(cgp.fn_chunk_mask);
  cmreader->Update();
  chunk_mask = cmreader->GetOutput();

  // Split the chunked mask into discrete regions
  chunk_labels = get_unique_labels(chunk_mask.GetPointer());

  // Threshold each chunk
  for(short label : chunk_labels)
    {
    if(label > 0)
      {
      // Apply binary threshold
      typedef itk::BinaryThresholdImageFilter<ChunkMaskImageType, typename LDDMMType::ImageType> TFilter;
      typename TFilter::Pointer thresh = TFilter::New();
      thresh->SetInput(chunk_mask);
      thresh->SetLowerThreshold(label);
      thresh->SetUpperThreshold(label);
      thresh->SetInsideValue(1.0);
      thresh->SetOutsideValue(0.0);
      thresh->Update();

      // Set up registration problem for this
      label_data[label] = Assembly();
      label_data[label].gp = gp;
      label_data[label].mask = thresh->GetOutput();

      // Pass the mask to each registration problem
      for(auto &ig : label_data[label].gp.input_groups)
        ig.fixed_mask = "label_mask";
      label_data[label].api.AddCachedInputObject("label_mask", label_data[label].mask);

      // Set the output filename
      label_data[label].gp.output = ssprintf(cgp.fn_output_pattern.c_str(), label);
      }
    }
}

#include "itkFastMarchingImageFilter.h"

template <unsigned int VDim, typename TReal=double>
class AffineRegularizer
{
public:
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCF;
  typedef MultiChunkAffineAssembly<VDim, TReal> Assembly;
  typedef std::map<short, Assembly> AssemblyMap;
  typedef itk::Point<TReal, VDim> Point;
  typedef itk::ContinuousIndex<TReal, VDim> CIndex;
  typedef vnl_vector_fixed<double, VDim> Vec;
  typedef vnl_matrix_fixed<double, VDim, VDim> Mat;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::ImagePointer ImagePointer;
  typedef itk::Image<short, VDim> ChunkMask;
  typedef typename ChunkMask::Pointer ChunkMaskPointer;

  struct OverlapData
  {
    // The labels involved
    unsigned int l1, l2;

    // List of coordinates
    std::vector<Point> x_phys;
    std::vector<CIndex> x_img_level;
  };

  struct ChunkData
  {
    Mat A, dA;
    Vec b, db;
  };

  AffineRegularizer(ChunkMask *chunk_mask, double border_radius_mm)
    {
    // Binarize the chunk mask to create a speed image for fast marching
    typedef itk::BinaryThresholdImageFilter<ChunkMask, ImageType> ThreshFilter;
    typename ThreshFilter::Pointer tf = ThreshFilter::New();
    tf->SetInput(chunk_mask);
    tf->SetLowerThreshold(1);
    tf->SetUpperThreshold(0x7fff);
    tf->SetInsideValue(1.0);
    tf->SetOutsideValue(1.0);

    // Get the list of unique values
    auto labels = get_unique_labels(chunk_mask);

    // For each label, we expand the label outward into the mask (this should be done using fast
    // marching). We can do this at full resolution to always keep using the same set of sample
    // points at all resolution levels
    // This image keeps the bit mask for overlapping regions
    ImagePointer ovl = LDDMMType::new_img(chunk_mask);
    for(auto label : labels)
      {
      // Create a fast marching filter
      typedef itk::FastMarchingImageFilter<ImageType, ImageType> FastMarchingType;
      typename FastMarchingType::Pointer fm = FastMarchingType::New();

      // Assign speed
      fm->SetInput(tf->GetOutput());

      // Assign trial nodes
      typename FastMarchingType::NodeContainer::Pointer trial = FastMarchingType::NodeContainer::New();
      unsigned int count = 0;
      itk::ImageRegionIteratorWithIndex<ChunkMask> it(chunk_mask, chunk_mask->GetBufferedRegion());
      for(; !it.IsAtEnd(); ++it)
        {
        if(it.Value() == label)
          {
          typename FastMarchingType::NodeType node;
          node.SetIndex(it.GetIndex());
          trial->InsertElement(count++, node);
          }
        }
      fm->SetTrialPoints(trial);

      // Set how far to run filter
      fm->SetStoppingValue(border_radius_mm);

      // Perform fast marching
      fm->Update();

      // Threshold the fast marching output and append to the overall map
      ImagePointer dil = fm->GetOutput();
      LDDMMType::img_threshold_in_place(dil, 0.0, border_radius_mm, 1, 0);
      LDDMMType::img_scale_in_place(dil, 1 << label);
      LDDMMType::img_add_in_place(ovl, dil);
      }

    // Calculate all adjacent pairs present
    float last_val = -1;
    std::set<unsigned int> unique_val;
    for(typename LDDMMType::ImageIterator it(ovl, ovl->GetBufferedRegion()); !it.IsAtEnd(); ++it)
      {
      if(it.Value() != last_val)
        {
        unique_val.insert((unsigned int) (it.Value() + 0.5));
        last_val = it.Value();
        }
      }

    LDDMMType::img_write(ovl, "/tmp/ovl.nii.gz");

    ovl_data.clear();
    total_pairs = 0;
    for(auto v : unique_val)
      {
      // Ignore single values
      if(ceil(log2(v)) == floor(log2(v)))
        continue;

      // Generate a list of all coordinates or this value
      std::vector<Point> pts;
      for(typename LDDMMType::ImageIterator it(ovl, ovl->GetBufferedRegion()); !it.IsAtEnd(); ++it)
        {
        if(v == (unsigned int) (it.Value() + 0.5))
          {
          // Record the point in physical coordinates
          Point p;
          ovl->TransformIndexToPhysicalPoint(it.GetIndex(), p);
          pts.push_back(p);
          }
        }

      // Get all labels involved
      for(unsigned int a = 1; a < 32; a++)
        {
        for(unsigned int b = a+1; b < 32; b++)
          {
          unsigned int mask = (1 << a) | (1 << b);
          if((v & (1 << a)) && (v & (1 << b)))
            {
            if(ovl_data.find(mask) == ovl_data.end())
              {
              ovl_data[mask].l1 = a;
              ovl_data[mask].l2 = b;
              }

            // Append coordinate list to this pair
            ovl_data[mask].x_phys.insert(ovl_data[mask].x_phys.end(), pts.begin(), pts.end());
            total_pairs += pts.size();
            }
          }
        }
      }

    // Binarize the masks
    for(auto it : ovl_data)
      std::cout << "Pair " << it.second.l1 << " " << it.second.l2 << " : " << it.second.x_phys.size() << std::endl;
    }

  void InitializeLevel(unsigned int level, AssemblyMap label_data)
    {
    this->label_data = label_data;
    this->level = level;
    ImagePointer ref = this->label_data.begin()->second.helper.GetFixedMask(0, level);
    for(auto &it : ovl_data)
      {
      it.second.x_img_level.clear();
      for(auto &p : it.second.x_phys)
        {
        itk::ContinuousIndex<double, VDim> ci;
        ref->TransformPhysicalPointToContinuousIndex(p, ci);
        it.second.x_img_level.push_back(ci);
        }
      }
    }

  unsigned int get_total_pairs() const { return total_pairs; }

  virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g)
    {
    // Get voxel-level transform associated with each chunk
    std::map<unsigned int, ChunkData> cd;
    unsigned int pos = 0;
    *f = 0;
    vnl_vector<double> zeros(VDim * (VDim+1), 0.0);
    for(auto &item : label_data)
      {
      auto &ld = item.second;
      int nunk = ld.acf->get_number_of_unknowns();

      // Initialize and compute the transform/gradient
      typename GreedyAPI::LinearTransformType::Pointer tran = GreedyAPI::LinearTransformType::New();
      ld.acf->GetTransform(x.extract(nunk, pos), tran, true);
      cd[item.first].A = tran->GetMatrix().GetVnlMatrix();
      cd[item.first].b = tran->GetOffset().GetVnlVector();
      cd[item.first].dA.fill(0.0);
      cd[item.first].db.fill(0.0);

      pos+=nunk;
      }

    // Compute disageement measure
    double total_dsq = 0.0;
    unsigned int n_pairs = 0;
    for(auto it : ovl_data)
      {
      // Get the transform corresponding to the first ACF
      auto &cd1 = cd[it.second.l1];
      auto &cd2 = cd[it.second.l2];
      for(auto &ci : it.second.x_img_level)
        {
        Vec x = ci.GetVnlVector();
        Vec y1 = cd1.A * x + cd1.b;
        Vec y2 = cd2.A * x + cd2.b;

        Vec dy = y2 - y1;

        // Compute the squared difference between the points
        double dsq = 0.5 * dy.squared_magnitude();
        total_dsq += dsq;
        n_pairs++;

        // if(dsq > 100.0)
        //  printf("!");

        // Compute the contributions to the gradient
        for(unsigned int a = 0; a < VDim; a++)
          {
          for(unsigned int b = 0; b < VDim; b++)
            {
            cd1.dA(a,b) -= dy[a] * x[b];
            cd2.dA(a,b) += dy[a] * x[b];
            }
          cd1.db(a) -= dy[a];
          cd2.db(a) += dy[a];
          }
        }
      }

    // Compute the RMS distance in pixels
    double d_rms = sqrt(total_dsq / n_pairs + 1.e-4);

    // Record the function
    if(f)
      *f = d_rms;

    // Backpropagate gradients
    if(g)
      {
      pos = 0;
      for(auto &item : label_data)
        {
        auto &ld = item.second;
        int nunk = ld.acf->get_number_of_unknowns();

        // Initialize and compute the transform/gradient
        typename GreedyAPI::LinearTransformType::Pointer d_tran = GreedyAPI::LinearTransformType::New();
        set_affine_transform(cd[item.first].dA, cd[item.first].db, d_tran.GetPointer());
        vnl_vector<double> d_x = ld.acf->BackPropTransform(d_tran);

        d_x *= 1.0 / (2. * n_pairs * d_rms);

        g->update(d_x, pos);

        pos+=nunk;
        }
      }
    }

private:
  AssemblyMap label_data;
  unsigned int level;
  std::map<unsigned int, OverlapData> ovl_data;
  unsigned int total_pairs;
};



template <unsigned int VDim, typename TReal = double>
class MultiChunkAffineCostFunction : public vnl_cost_function
{
public:
  typedef MultiChunkAffineAssembly<VDim, TReal> Assembly;
  typedef AffineRegularizer<VDim, TReal> Regularizer;
  typedef std::map<short, Assembly> LabelData;
  MultiChunkAffineCostFunction(
      int n_unknowns, LabelData &in_ld,
      unsigned int in_level, Regularizer *in_reg, double in_reg_weight)
    : vnl_cost_function(n_unknowns), label_data(in_ld), reg(in_reg), reg_weight(in_reg_weight)
    {
    // Scale the weight by the number of pairs - so that we are computing average error per pair
    reg_weight *= (1 << in_level);
    }

  virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g)
  {
    // Pass the affine parameters to individual functions
    unsigned int pos = 0;
    *f = 0;

    printf("NCC: ");
    for(auto &item : label_data)
      {
      auto &ld = item.second;
      int nunk = ld.acf->get_number_of_unknowns();
      double f_chunk;
      vnl_vector<double> g_chunk(nunk, 0.0);
      ld.acf->compute(x.extract(nunk, pos), &f_chunk, g ? &g_chunk : nullptr);

      *f += f_chunk * 0.0001;
      if(g)
        g->update(g_chunk * 0.0001, pos);

      printf("%4.3f ", f_chunk * 0.0001);
      pos+=nunk;
      }

    // Compute the regularization penalty
    double f_reg;
    vnl_vector<double> g_reg(x.size(), 0.0);
    reg->compute(x, &f_reg, g ? &g_reg : nullptr);

    // Add the regularizer
    *f += reg_weight * f_reg;
    if(g)
      (*g) += reg_weight * g_reg;

    // Report
    printf("Reg: %5.3f  Tot: %5.3f\n", f_reg, *f);
  }

protected:
  double reg_weight;
  LabelData &label_data;
  Regularizer *reg;
};


template <unsigned int VDim, typename TReal=double>
int run_affine(ChunkGreedyParameters cgp, GreedyParameters gp)
{
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCF;
  typedef MultiChunkAffineAssembly<VDim, TReal> Assembly;

  // Perform the common initialization (mask split, etc)
  std::map<short, Assembly> label_data;
  std::vector<short> chunk_labels;
  typename itk::Image<short, VDim>::Pointer chunk_mask;
  initialize_assemblies(cgp, gp, label_data, chunk_labels, chunk_mask);

  // Peform affine-specific initialization for each label
  for(auto &item : label_data)
    {
    // Read the data
    // Set the scaling factors for multi-resolution
    item.second.helper.SetDefaultPyramidFactors(gp.iter_per_level.size());

    // Read the image pairs to register - this will also build the composite pyramids
    // In affine mode, we do not force resampling of moving image to fixed image space
    item.second.api.ReadImages(item.second.gp, item.second.helper, false);

    // Add random sampling jitter for affine stability at voxel edges
    item.second.helper.SetJitterSigma(gp.affine_jitter);
    }

  // The number of resolution levels
  unsigned nlevels = gp.iter_per_level.size();

  // Create the affine regularizer
  AffineRegularizer<VDim, TReal> reg(chunk_mask, 0.5);

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    unsigned total_unk = 0;
    for(auto &item : label_data)
      {
      // Create an affine problem for each level
      auto &ld = item.second;
      ld.acf = item.second.api.CreateAffineCostFunction(ld.gp, ld.helper, level);
      total_unk += ld.acf->get_number_of_unknowns();

      // Current transform
      ld.tLevel = GreedyAPI::LinearTransformType::New();

      // Set up the initial transform
      if(level == 0)
        ld.api.InitializeAffineTransform(ld.gp, ld.helper, ld.acf, ld.tLevel);
      else
        ld.api.MapPhysicalRASSpaceToAffine(ld.helper, 0, level, ld.Q_physical, ld.tLevel);
      }

    // Run optimization
    if(gp.iter_per_level[level] > 0)
      {
      // Combine the coefficients into a single vector
      vnl_vector<double> xLevel(total_unk, 0.0);
      int pos = 0;
      for(auto &item : label_data)
        {
        auto x = item.second.acf->GetCoefficients(item.second.tLevel.GetPointer());
        xLevel.update(x, pos);
        pos += x.size();
        }

      // Set up a custom objective function that combines similarity term and a regularizer on the multiple
      // rigid transformations

      // Create a regularizer
      reg.InitializeLevel(level, label_data);

      // Test the derivatives of the regularizer
      vnl_vector<double> xGradN(xLevel.size(), 0.0), xGradA(xLevel.size(), 0.0);
      double f, f1, f2, eps=1.0e-4;
      reg.compute(xLevel, &f, &xGradA);
      for(int i = 0; i < xLevel.size(); i++)
        {
        double x0 = xLevel[i];
        xLevel[i] = x0 - eps; reg.compute(xLevel, &f1, nullptr);
        xLevel[i] = x0 + eps; reg.compute(xLevel, &f2, nullptr);
        xLevel[i] = x0;
        xGradN[i] = (f2 - f1) / (2 * eps);
        }
      printf("Derivative check:\n");
      for(int i = 0; i < xLevel.size(); i++)
        printf("%d: %8.4f  %8.4f\n", i, xGradA[i], xGradN[i]);

      std::cout << "Level: " << level << " XLevel: " << xLevel << std::endl;
      MultiChunkAffineCostFunction<VDim, TReal> cfun(total_unk, label_data, level, &reg, cgp.reg_weight);

      // Set up the optimizer
      vnl_lbfgs *optimizer = new vnl_lbfgs(cfun);

      // Using defaults from scipy
      double ftol = (gp.lbfgs_param.ftol == 0.0) ? 2.220446049250313e-9 : gp.lbfgs_param.ftol;
      double gtol = (gp.lbfgs_param.gtol == 0.0) ? 1e-05 : gp.lbfgs_param.gtol;

      optimizer->set_f_tolerance(ftol);
      optimizer->set_g_tolerance(gtol);
      if(gp.lbfgs_param.memory > 0)
        optimizer->memory = gp.lbfgs_param.memory;

      optimizer->set_trace(gp.verbosity > GreedyParameters::VERB_NONE);
      optimizer->set_verbose(gp.verbosity > GreedyParameters::VERB_DEFAULT);
      optimizer->set_max_function_evals(gp.iter_per_level[level]);

      optimizer->minimize(xLevel);
      delete optimizer;

      pos = 0;
      for(auto &item : label_data)
        {
        auto &ld = item.second;

        // Get the final transform
        auto x = xLevel.extract(ld.acf->get_number_of_unknowns(), pos);
        pos += ld.acf->get_number_of_unknowns();

        auto tFinal = GreedyAPI::LinearTransformType::New();
        ld.acf->GetTransform(x, tFinal.GetPointer(), false);

        // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
        ld.Q_physical = ld.api.MapAffineToPhysicalRASSpace(ld.helper, 0, level, tFinal);
        }
      }
    }

  // Save the separate matrices
  for(auto &item : label_data)
    {
    // Create an affine problem for each level
    auto &ld = item.second;
    ld.api.WriteAffineMatrixViaCache(ld.gp.output, ld.Q_physical);
    }

  return 0;
}



template <unsigned int VDim, typename TReal=double>
int run_deform(ChunkGreedyParameters cgp, GreedyParameters gp)
{
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCF;
  typedef MultiChunkAffineAssembly<VDim, TReal> Assembly;

  // Perform the common initialization (mask split, etc)
  std::map<short, Assembly> label_data;
  std::vector<short> chunk_labels;
  typename itk::Image<short, VDim>::Pointer chunk_mask;
  initialize_assemblies(cgp, gp, label_data, chunk_labels, chunk_mask);

  // Peform affine-specific initialization for each label
  for(auto &item : label_data)
    {
    // Map accepted patterns to item-specific parameters
    if(cgp.fn_init_tran_pattern.size())
      {
      std::string fn_tran = ssprintf(cgp.fn_init_tran_pattern.c_str(), item.first);
      item.second.gp.input_groups[0].moving_pre_transforms.push_back(TransformSpec(fn_tran, 1.0));
      }

    item.second.gp.inverse_warp = ssprintf(cgp.fn_output_inv_pattern.c_str(), item.first);
    item.second.gp.root_warp = ssprintf(cgp.fn_output_root_pattern.c_str(), item.first);

    // Add random sampling jitter for affine stability at voxel edges
    item.second.api.RunDeformable(item.second.gp);
    }

  return 0;
}



template <unsigned int VDim>
int run(ChunkGreedyParameters cgp, GreedyParameters gp)
{
  if(gp.mode == GreedyParameters::AFFINE)
    run_affine<VDim>(cgp, gp);
  else if(gp.mode == GreedyParameters::GREEDY)
    run_deform<VDim>(cgp, gp);
  else
    throw GreedyException("Only affine mode is implemented");
  return 0;
}

int main(int argc, char *argv[])
{
  if(argc < 2)
    return usage();

  // List of greedy commands that are recognized by this mode
  std::set<std::string> greedy_cmd {
    "-threads", "-d", "-m", "-i", "-n", "-a", "-dof", "-bg", "-ia", "-wncc-mask-dilate", "-search", "-dump-pyramid", "-dump-metric",
    "-it", "-sv", "-s"
  };

  CommandLineHelper cl(argc, argv);
  ChunkGreedyParameters cg_param;
  GreedyParameters gp;
  std::string arg;
  while(cl.read_command(arg))
    {
    if(arg == "-o")
      {
      cg_param.fn_output_pattern = cl.read_string();
      }
    else if(arg == "-oinv")
      {
      cg_param.fn_output_inv_pattern = cl.read_string();
      }
    else if(arg == "-oroot")
      {
      cg_param.fn_output_root_pattern = cl.read_string();
      }
    else if(arg == "-it")
      {
      cg_param.fn_init_tran_pattern = cl.read_string();
      }
    else if(arg == "-cm")
      {
      cg_param.fn_chunk_mask = cl.read_existing_filename();
      }
    else if(arg == "-wreg")
      {
      cg_param.reg_weight = cl.read_double();
      }
    else if(greedy_cmd.find(arg) != greedy_cmd.end())
      {
      gp.ParseCommandLine(arg, cl);
      }
    else
      throw GreedyException("Unknown parameter to 'splat': %s", arg.c_str());
    }

  // Check the dimension
  if(gp.dim == 2)
    run<2>(cg_param, gp);
  else if(gp.dim == 3)
    run<3>(cg_param, gp);
}
