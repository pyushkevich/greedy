#include "PropagationIO.h"
#include "PropagationTools.h"

using namespace propagation;

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

  // Read Reference Image4D from the paramter
  pInput->m_Data->img4d = PropagationTools<TReal>::
      template ReadImage<TImage4D>(pParam.img4d);

  // Read Segmentation Image from the parameter
  pInput->m_Data->seg_ref = PropagationTools<TReal>
      ::template ReadImage<TLabelImage3D>(pParam.segspec.refseg);

  // Todo: Read 4D segmentation image and extract the 3D time point from it
  // Set output directory
  pInput->m_Data->outdir = pParam.segspec.outsegdir;

  // Copy the parameter
  pInput->m_GreedyParam = gParam;
  pInput->m_PropagationParam = pParam;
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
