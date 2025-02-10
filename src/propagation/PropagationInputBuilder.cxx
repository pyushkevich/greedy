#include "PropagationInputBuilder.h"
#include "GreedyException.h"
#include "GreedyMeshIO.h"
#include "PropagationTools.h"

using namespace propagation;

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
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::SetReferenceSegmentationIn4D(TLabelImage4D *seg4d)
{
  m_Data->seg4d_in = seg4d;
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
::AddExtraMeshToWarp(std::string fnmesh, std::string outpattern)
{
  MeshSpec meshspec;
  meshspec.fn_mesh = fnmesh;
  meshspec.fnout_pattern = outpattern;
  m_PParam.extra_mesh_list.push_back(meshspec);
}

template<typename TReal>
void
PropagationInputBuilder<TReal>
::AddExtraMeshToWarp(TPropagationMeshPointer mesh, std::string tag)
{
  MeshSpec meshspec;
  meshspec.fn_mesh = tag; // useless
  meshspec.fnout_pattern = tag;
  meshspec.cached = true;
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
void
PropagationInputBuilder<TReal>
::ConfigForCLI(const PropagationParameters &pParam, const GreedyParameters &gParam)
{
  // Copy the parameters, this will cover all settings not related to data reading
  this->SetPropagationParameters(pParam);
  this->SetGreedyParameters(gParam);

  // Read the data
  this->SetImage4D(PropagationTools<TReal>::template ReadImage<TImage4D>(pParam.fn_img4d));

  // Read Segmentation Image from the parameter
  if (pParam.use4DSegInput)
    {
    this->SetReferenceSegmentationIn4D(PropagationTools<TReal>
        ::template ReadImage<TLabelImage4D>(pParam.fn_seg4d));
    }
  else
    {
    this->SetReferenceSegmentationIn3D(PropagationTools<TReal>
        ::template ReadImage<TLabelImage3D>(pParam.fn_seg3d));
    }

  // Read extra meshes
  for (auto &meshspec : pParam.extra_mesh_list)
    {
    auto meshData = ReadMesh(meshspec.fn_mesh.c_str());
    auto *pd = dynamic_cast<vtkPolyData *>(meshData.GetPointer());
    if (!pd)
      throw GreedyException("Propagation: Extra mesh %s is not a vtkPolyData", meshspec.fn_mesh.c_str());
    
    this->AddExtraMeshToWarp(pd, meshspec.fnout_pattern);
    }
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

  if (m_PParam.outdir.size() > 0)
    pInput->SetOutputDirectory(m_PParam.outdir);

  return pInput;
}

namespace propagation
{
  template class PropagationInputBuilder<float>;
  template class PropagationInputBuilder<double>;
}