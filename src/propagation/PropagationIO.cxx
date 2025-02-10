#include "PropagationIO.h"
#include "PropagationTools.h"
#include "PropagationCommon.hxx"


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
typename PropagationOutput<TReal>::TImage3D::Pointer
PropagationOutput<TReal>
::GetImage3D(unsigned int tp)
{
  return m_Data->tp_data.at(tp).img;
}

template<typename TReal>
typename PropagationOutput<TReal>::TLabelImage3D::Pointer
PropagationOutput<TReal>
::GetSegmentation3D(unsigned int tp)
{
  return m_Data->tp_data.at(tp).seg;
}

template<typename TReal>
typename PropagationOutput<TReal>
::TSegmentation3DSeries
PropagationOutput<TReal>
::GetSegmentation3DSeries()
{
  TSegmentation3DSeries ret;
  for (auto kv : m_Data->tp_data)
    {
    ret[kv.first] = kv.second.seg;
    }

  return ret;
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
::TPropagationMeshPointer
PropagationOutput<TReal>
::GetMesh(unsigned int tp)
{
  return m_Data->tp_data.at(tp).seg_mesh;
}

template<typename TReal>
typename PropagationOutput<TReal>
::TMeshSeries
PropagationOutput<TReal>
::GetExtraMeshSeries(std::string tag)
{
  TMeshSeries ret;
  for (auto &kv : m_Data->tp_data)
    {
    ret[kv.first] = kv.second.GetExtraMesh(tag);
    }

  return ret;
}

template<typename TReal>
typename PropagationOutput<TReal>
::TPropagationMeshPointer
PropagationOutput<TReal>
::GetExtraMesh(std::string tag, unsigned int tp)
{
  return m_Data->tp_data.at(tp).GetExtraMesh(tag);
}

template<typename TReal>
typename PropagationOutput<TReal>
::TMeshSeriesMap
PropagationOutput<TReal>
::GetAllExtraMeshSeries()
{
  TMeshSeriesMap ret;
  auto firstTP = m_Data->tp_data.begin()->second;

  // populate return map
  for (auto tag : firstTP.GetExtraMeshTags())
    {
    ret[tag] = this->GetExtraMeshSeries(tag);
    }

  return ret;
}

template <typename TReal>
std::vector<unsigned int>
PropagationOutput<TReal>
::GetTimePointList()
{
  std::vector<unsigned int> ret;
  for (auto kv : m_Data->tp_data)
    {
    ret.push_back(kv.first);
    }

  return ret;
}

template <typename TReal>
size_t
PropagationOutput<TReal>
::GetNumberOfTimePoints()
{
  return m_Data->tp_data.size();
}



namespace propagation
{
  template class PropagationInput<float>;
  template class PropagationInput<double>;
  template class PropagationOutput<float>;
  template class PropagationOutput<double>;
}
