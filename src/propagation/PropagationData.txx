#include "PropagationData.h"

namespace propagation
{

template<typename TReal>
PropagationData<TReal>
::PropagationData()
{

}

template <typename TReal>
size_t
PropagationData<TReal>
::GetNumberOfTimePoints()
{
  return this->img4d->GetBufferedRegion().GetSize()[3];
}

template<typename TReal>
TimePointData<TReal>
::TimePointData()
{
	affine_to_prev = TTransform::New();
}

template<typename TReal>
TimePointData<TReal>
::~TimePointData()
{

}

}
