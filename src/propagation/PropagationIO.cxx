#include "PropagationIO.h"
#include "PropagationTools.h"


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
  if (m_GreedyParam.dim != 3)
    m_GreedyParam.dim = 3;
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
  // if fails, check SetDefaultGreedyParameters
  assert(m_GreedyParam.dim == 3);
}

template<typename TReal>
void
PropagationInput<TReal>
::ValidatePropagationParameters() const
{
  if (m_PropagationParam.outdir.size() == 0)
    throw GreedyException("Output directory (-spo) not provided!");
}


//==================================================
// PropagationOutput Definitions
//==================================================

template<typename TReal>
PropagationOutput<TReal>
::PropagationOutput()
{

}

template<typename TReal>
PropagationOutput<TReal>
::~PropagationOutput()
{

}


template<typename TReal>
PropagationInputBuilder<TReal>
::PropagationInputBuilder()
{
  m_Input = std::make_shared<PropagationInput<TReal>>();
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
  m_Input = nullptr;
  m_Input = std::make_shared<PropagationInput<TReal>>();
}

template<typename TReal>
std::shared_ptr<PropagationInput<TReal>>
PropagationInputBuilder<TReal>
::CreateInputForCommandLineRun(const PropagationParameters &pParam, const GreedyParameters &gParam)
{
  std::shared_ptr<PropagationInput<TReal>> pInput = std::make_shared<PropagationInput<TReal>>();

  // Copy the parameter
  pInput->SetGreedyParameters(gParam);
  pInput->SetPropagationParameters(pParam);

  // Read Reference Image4D from the paramter
  pInput->m_Data->img4d = PropagationTools<TReal>::
      template ReadImage<TImage4D>(pParam.fn_img4d);

  // Read Segmentation Image from the parameter
  pInput->m_Data->seg_ref = PropagationTools<TReal>
      ::template ReadImage<TLabelImage3D>(pParam.segspec.fn_seg);

  // Todo: Read 4D segmentation image and extract the 3D time point from it
  // Set output directory
  pInput->m_Data->outdir = pParam.outdir;

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
