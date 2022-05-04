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
#include <vnl/algo/vnl_lbfgs.h>


struct ChunkGreedyParameters
{
  std::string fn_chunk_mask;
  std::string fn_output_pattern;
};

int usage()
{
  printf("chunk_greedy: Multi-chunk registration with GreedyReg\n");
  printf("usage: \n");
  printf("  chunk_greedy <options> <greedy_options> \n");
  printf("options for chunk_greedy: \n");
  printf("  -cm: <file>      : Chunk mask, each chunk assigned a different label\n");
  printf("  -op: <pattern>   : Output pattern (printf format), equivalent to -o in greedy\n");
  printf("main greedy options accepted: \n");
  printf("  -d, -i, -m, -n, -a, -dof, -bg, -ia, -wncc-mask-dilate");
  return -1;
}

template <typename TImage>
std::vector<typename TImage::PixelType>
get_unique_labels(TImage *label_image, unsigned int max_allowed = 100)
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
  vnl_matrix<double> Q_physical;
};

template <unsigned int VDim, typename TReal = double>
class MultiChunkAffineCostFunction : public vnl_cost_function
{
public:
  typedef MultiChunkAffineAssembly<VDim, TReal> Assembly;
  typedef std::map<short, Assembly> LabelData;
  MultiChunkAffineCostFunction(int n_unknowns, LabelData &in_ld) : vnl_cost_function(n_unknowns), label_data(in_ld) {}
  virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g)
  {
    // Pass the affine parameters to individual functions
    unsigned int pos = 0;
    *f = 0;
    for(auto &item : label_data)
      {
      auto &ld = item.second;
      int nunk = ld.acf->get_number_of_unknowns();
      double f_chunk;
      if(g)
        {
        vnl_vector<double> g_chunk(nunk, 0.0);
        ld.acf->compute(x.extract(nunk, pos), &f_chunk, &g_chunk);
        g->update(g_chunk, pos);
        }
      else
        {
        ld.acf->compute(x.extract(nunk, pos), &f_chunk, nullptr);
        }
      pos+=nunk;
      *f += f_chunk;
      }
  }

protected:
  LabelData &label_data;
};



template <unsigned int VDim, typename TReal=double>
int run_affine(ChunkGreedyParameters cgp, GreedyParameters gp)
{
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCF;

  // Read the chunked mask as an image of shorts
  typedef itk::Image<short, VDim> ChunkMaskImageType;
  typedef itk::ImageFileReader<ChunkMaskImageType> ChunkMaskReader;
  typename ChunkMaskReader::Pointer cmreader = ChunkMaskReader::New();
  cmreader->SetFileName(cgp.fn_chunk_mask);
  cmreader->Update();
  typename ChunkMaskImageType::Pointer cm = cmreader->GetOutput();

  // Split the chunked mask into discrete regions
  std::vector<short> chunk_labels = get_unique_labels(cm.GetPointer());

  // Threshold each chunk
  typedef MultiChunkAffineAssembly<VDim, TReal> Assembly;
  std::map<short, Assembly> label_data;
  for(short label : chunk_labels)
    {
    if(label > 0)
      {
      // Apply binary threshold
      typedef itk::BinaryThresholdImageFilter<ChunkMaskImageType, typename LDDMMType::ImageType> TFilter;
      typename TFilter::Pointer thresh = TFilter::New();
      thresh->SetInput(cm);
      thresh->SetLowerThreshold(label);
      thresh->SetUpperThreshold(label);
      thresh->SetInsideValue(1.0);
      thresh->SetOutsideValue(0.0);
      thresh->Update();

      // Set up registration problem for this
      label_data[label] = Assembly();
      label_data[label].gp = gp;

      // Pass the mask to each registration problem
      for(auto &ig : label_data[label].gp.input_groups)
        ig.fixed_mask = "label_mask";
      label_data[label].api.AddCachedInputObject("label_mask", thresh->GetOutput());

      // Read the data
      // Set the scaling factors for multi-resolution
      label_data[label].helper.SetDefaultPyramidFactors(gp.iter_per_level.size());

      // Add random sampling jitter for affine stability at voxel edges
      label_data[label].helper.SetJitterSigma(gp.affine_jitter);

      // Read the image pairs to register - this will also build the composite pyramids
      // In affine mode, we do not force resampling of moving image to fixed image space
      label_data[label].api.ReadImages(label_data[label].gp, label_data[label].helper, false);

      // Set the output filename
      char buffer[1024];
      sprintf(buffer, cgp.fn_output_pattern.c_str(), label);
      label_data[label].gp.output = buffer;
      }
    }

  // The number of resolution levels
  unsigned nlevels = gp.iter_per_level.size();

  // Iterate over the resolution levels
  unsigned total_unk = 0;
  for(unsigned int level = 0; level < nlevels; ++level)
    {
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
      MultiChunkAffineCostFunction<VDim, TReal> cfun(total_unk, label_data);

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
        ld.acf->GetTransform(x, tFinal.GetPointer());

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


template <unsigned int VDim>
int run(ChunkGreedyParameters cgp, GreedyParameters gp)
{
  if(gp.mode == GreedyParameters::AFFINE)
    run_affine<VDim>(cgp, gp);
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
    "-threads", "-d", "-m", "-i", "-n", "-a", "-dof", "-bg", "-ia", "-wncc-mask-dilate"
  };

  CommandLineHelper cl(argc, argv);
  ChunkGreedyParameters cg_param;
  GreedyParameters gp;
  std::string arg;
  while(cl.read_command(arg))
    {
    if(arg == "-op")
      {
      cg_param.fn_output_pattern = cl.read_string();
      }
    else if(arg == "-cm")
      {
      cg_param.fn_chunk_mask = cl.read_existing_filename();
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
