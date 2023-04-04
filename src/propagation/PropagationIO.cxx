#include "PropagationIO.h"
#include "PropagationTools.h"
#include "PropagationCommon.h"


using namespace propagation;

//==================================================
// PropagationInput Definitions
//==================================================

template<typename TReal>
PropagationInput<TReal>
::PropagationInput()
{
  m_Data = std::make_shared<PropagationData<TReal>>();
}

template<typename TReal>
PropagationInput<TReal>
::~PropagationInput()
{

}

template<typename TReal>
void
PropagationInput<TReal>
::SetDefaultGreedyParameters()
{

}

template<typename TReal>
void
PropagationInput<TReal>
::SetDefaultPropagationParameters()
{
  if (m_PropagationParam.fnsegout_pattern.size() == 0)
    {
    std::cout << "-- [Propagation] segmentation output filename pattern (-sps-op) has not been set. "
                 "Setting to default value \"segmentation_%02d_resliced.nii.gz\"" << std::endl;
    m_PropagationParam.fnsegout_pattern = "segmentation_%02d_resliced.nii.gz";
    }

  if (m_PropagationParam.fnmeshout_pattern.size() == 0)
    {
    std::cout << "-- [Propagation] segmentation mesh output filename pattern (-sps-mop) has not been set. "
                 "Setting to default value \"segmentation_mesh_%02d_resliced.vtk\"" << std::endl;
    m_PropagationParam.fnmeshout_pattern = "segmentation_mesh_%02d_resliced.vtk";
    }
}

template<typename TReal>
void
PropagationInput<TReal>
::SetGreedyParameters(const GreedyParameters &gParam)
{
  this->m_GreedyParam = gParam;

  // dim = 3 is mandatory and is not configured by users
  if (m_GreedyParam.dim != 3)
    m_GreedyParam.dim = 3;

  SetDefaultGreedyParameters();
  ValidateGreedyParameters();
}

template<typename TReal>
void
PropagationInput<TReal>
::SetPropagationParameters(const PropagationParameters &pParam)
{
  this->m_PropagationParam = pParam;
  SetDefaultPropagationParameters();
  ValidatePropagationParameters();
}

template<typename TReal>
void
PropagationInput<TReal>
::ValidateGreedyParameters() const
{

}

template<typename TReal>
void
PropagationInput<TReal>
::ValidatePropagationParameters() const
{
  if (m_PropagationParam.writeOutputToDisk && m_PropagationParam.outdir.size() == 0)
    throw GreedyException("Output directory (-spo) not provided!");
}

template<typename TReal>
void
PropagationInput<TReal>
::SetDataForAPIRun(std::shared_ptr<PropagationData<TReal>> data)
{
  m_Data = data;
}


//==================================================
// PropagationOutput Definitions
//==================================================

template<typename TReal>
PropagationOutput<TReal>
::PropagationOutput()
{
  m_Data = nullptr;
}

template<typename TReal>
PropagationOutput<TReal>
::~PropagationOutput()
{

}

template<typename TReal>
void
PropagationOutput<TReal>
::Initialize(std::shared_ptr<PropagationData<TReal>> data)
{
  m_Data = data;
}

template<typename TReal>
typename PropagationOutput<TReal>::TLabelImage3D::Pointer
PropagationOutput<TReal>
::GetSegmentation3D(unsigned int tp)
{
  if (m_Data->tp_data.count(tp))
    return m_Data->tp_data[tp].seg;
  else
    throw GreedyException(
        "PropagationOutput::GetSegmentation3D, timepoint [%d] does not exist in target tp list", tp);

}

template<typename TReal>
typename PropagationOutput<TReal>::TLabelImage4D::Pointer
PropagationOutput<TReal>
::GetSegmentation4D()
{
  return m_Data->seg4d_out;
}

template<typename TReal>
typename PropagationOutput<TReal>
::TMeshSeries
PropagationOutput<TReal>
::GetMeshSeries()
{
  TMeshSeries ret;
  for (auto &kv : m_Data->tp_data)
    {
    ret[kv.first] = kv.second.seg_mesh;
    }

  return ret;
}

template<typename TReal>
typename PropagationOutput<TReal>
::TMeshSeries
PropagationOutput<TReal>
::GetExtraMeshSeries(std::string &tag)
{
  TMeshSeries ret;
  for (auto &kv : m_Data->tp_data)
    {
    ret[kv.first] = kv.second.extra_meshes[tag];
    }

  return ret;
}

template<typename TReal>
typename PropagationOutput<TReal>
::TMeshSeriesMap
PropagationOutput<TReal>
::GetAllExtraMeshSeries()
{
  TMeshSeriesMap ret;
  auto firstTP = m_Data->tp_data.begin()->second;
  std::vector<std::string> tags;

  // get all the tags from the first time point
  for (auto kv : firstTP.extra_meshes)
    {
    tags.push_back(kv.first);
    }

  // populate return map
  for (auto tag : tags)
    {
    ret[tag] = this->GetExtraMeshSeries(tag);
    }

  return ret;
}

//==================================================
// PropagationInputBuilder Definitions
//==================================================

template<typename TReal>
PropagationInputBuilder<TReal>
::PropagationInputBuilder()
{
  m_Data = std::make_shared<PropagationData<TReal>>();
  m_PParam.writeOutputToDisk = false;
}

template<typename TReal>
PropagationInputBuilder<TReal>
::~PropagationInputBuilder()
{

}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::Reset()
{
  m_Data = nullptr;
  m_Data = std::make_shared<PropagationData<TReal>>();
  m_GParam = GreedyParameters();
  m_PParam = PropagationParameters();
}

template<typename TReal>
std::shared_ptr<PropagationInput<TReal>>
PropagationInputBuilder<TReal>
::BuildInputForCommandLineRun(const PropagationParameters &pParam, const GreedyParameters &gParam)
{
  std::shared_ptr<PropagationInput<TReal>> pInput = std::make_shared<PropagationInput<TReal>>();

  // Copy the parameter
  pInput->SetGreedyParameters(gParam);
  pInput->SetPropagationParameters(pParam);

  // Read Reference Image4D from the paramter
  pInput->m_Data->img4d = PropagationTools<TReal>::template ReadImage<TImage4D>(pParam.fn_img4d);

  // Read Segmentation Image from the parameter
  if (pParam.use4DSegInput)
    {
    pInput->m_Data->seg4d_in = PropagationTools<TReal>
        ::template ReadImage<TLabelImage4D>(pParam.fn_seg4d);
    }
  else
    {
    pInput->m_Data->seg_ref = PropagationTools<TReal>
        ::template ReadImage<TLabelImage3D>(pParam.fn_seg3d);
    }


  // Todo: Read 4D segmentation image and extract the 3D time point from it
  // Set output directory
  pInput->m_Data->outdir = pParam.outdir;

  return pInput;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetImage4D(TImage4D *img4d)
{
  m_Data->img4d = img4d;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetReferenceSegmentationIn3D(TLabelImage3D *seg3d)
{
  m_Data->seg_ref = seg3d;
  m_PParam.use4DSegInput = false;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetReferenceSegmentationIn4D(TLabelImage4D *seg4d)
{
  m_Data->seg4d_in = seg4d;
  m_PParam.use4DSegInput = true;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetReferenceTimePoint(unsigned int refTP)
{
  m_PParam.refTP = refTP;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetTargetTimePoints(const std::vector<unsigned int> &targetTPs)
{
  m_PParam.targetTPs = targetTPs;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetResliceMetric(const InterpSpec metric)
{
  m_PParam.reslice_spec = metric;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetResliceMetricToNearestNeighbor()
{
  m_PParam.reslice_spec = InterpSpec(InterpSpec::NEAREST);
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetResliceMetricToLinear()
{
  m_PParam.reslice_spec = InterpSpec(InterpSpec::LINEAR);
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetResliceMetricToLabel(double sigma, bool is_physical_unit)
{
  m_PParam.reslice_spec = InterpSpec(InterpSpec::LABELWISE, sigma, is_physical_unit);
}

template<typename TReal>
InterpSpec
PropagationInputBuilder<TReal>
::GetResliceMetric() const
{
  return m_PParam.reslice_spec;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetDebugOn(std::string &debug_dir)
{
  m_PParam.debug = true;
  m_PParam.debug_dir = debug_dir;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetRegistrationMetric(GreedyParameters::MetricType metric, std::vector<int> metric_radius)
{
  m_GParam.metric = metric;
  if (metric == GreedyParameters::NCC || metric == GreedyParameters::WNCC)
    {
    if (metric_radius.size() == 0)
      throw GreedyException("PropagationInputBuilder::SetRegistrationMetric: metric_radius is required but not givien");

    m_GParam.metric_radius = metric_radius;
    }
}

template<typename TReal>
GreedyParameters::MetricType
PropagationInputBuilder<TReal>
::GetRegistrationMetric() const
{
  return m_GParam.metric;
}

template<typename TReal>
std::vector<int>
PropagationInputBuilder<TReal>
::GetRegistrationMetricRadius() const
{
  return m_GParam.metric_radius;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetMultiResolutionSchedule(std::vector<int> iter_per_level)
{
  m_GParam.iter_per_level = iter_per_level;
}

template<typename TReal>
std::vector<int>
PropagationInputBuilder<TReal>
::GetMultiResolutionSchedule() const
{
  return m_GParam.iter_per_level;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetAffineDOF(GreedyParameters::AffineDOF dof)
{
  m_GParam.affine_dof = dof;
}

template<typename TReal>
GreedyParameters::AffineDOF
PropagationInputBuilder<TReal>
::GetAffineDOF() const
{
  return m_GParam.affine_dof;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::AddExtraMeshToWarp(std::string &fnmesh, std::string &outpattern)
{
  MeshSpec meshspec;
  meshspec.fn_mesh = fnmesh;
  meshspec.fnout_pattern = outpattern;
  m_PParam.extra_mesh_list.push_back(meshspec);
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::AddExtraMeshToWarp(TPropagationMeshPointer mesh, std::string &tag)
{
  MeshSpec meshspec;
  meshspec.fn_mesh = "cached";
  meshspec.fnout_pattern = tag;
  m_PParam.extra_mesh_list.push_back(meshspec);
  m_Data->extra_mesh_cache[tag] = mesh;
}

template<typename TReal>
std::vector<MeshSpec>
PropagationInputBuilder<TReal>
::GetExtraMeshesToWarp() const
{
  return m_PParam.extra_mesh_list;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetGreedyVerbosity(GreedyParameters::Verbosity v)
{
  m_GParam.verbosity = v;
}

template<typename TReal>
GreedyParameters::Verbosity
PropagationInputBuilder<TReal>
::GetGreedyVerbosity() const
{
  return m_GParam.verbosity;
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetPropagationVerbosity(PropagationParameters::Verbosity v)
{
  m_PParam.verbosity = v;
}


template<typename TReal>
PropagationParameters::Verbosity
PropagationInputBuilder<TReal>
::GetPropagationVerbosity() const
{
  return m_PParam.verbosity;
}

template<typename TReal>
std::shared_ptr<PropagationInput<TReal>>
PropagationInputBuilder<TReal>
::BuildPropagationInput()
{
  std::shared_ptr<PropagationInput<TReal>> pInput = std::make_shared<PropagationInput<TReal>>();
  pInput->SetGreedyParameters(m_GParam);
  pInput->SetPropagationParameters(m_PParam);
  pInput->SetDataForAPIRun(m_Data);
  return pInput;
}



namespace propagation
{
  template class PropagationInput<float>;
  template class PropagationInput<double>;
  template class PropagationOutput<float>;
  template class PropagationOutput<double>;
  template class PropagationInputBuilder<float>;
  template class PropagationInputBuilder<double>;
}
