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

//template<typename TReal>
//TimePointData<TReal>
//::TimePointData(const TimePointData &other)
//{
//  this->img = other.img;
//  this->img_srs = other.img_srs;
//  this->seg = other.seg;
//  this->seg_srs = other.seg_srs;
//  this->full_res_mask = other.full_res_mask;
//  this->affine_to_prev = other.affine_to_prev;
//  this->deform_to_prev = other.deform_to_prev;
//  this->deform_to_ref = other.deform_to_ref;
//  this->deform_from_prev = other.deform_from_prev;
//  this->deform_from_ref = other.deform_from_ref;

//  std::copy(other.transform_specs.begin(), other.transform_specs.end(), this->transform_specs.begin());
//  std::copy(other.full_res_label_trans_specs.begin(), other.full_res_label_trans_specs.end(),
//            this->full_res_label_trans_specs.begin());
//}

//template<typename TReal>
//TimePointData<TReal>&
//TimePointData<TReal>
//::operator=(const TimePointData &other)
//{
//  this->img = other.img;
//  this->img_srs = other.img_srs;
//  this->seg = other.seg;
//  this->seg_srs = other.seg_srs;
//  this->full_res_mask = other.full_res_mask;
//  this->affine_to_prev = other.affine_to_prev;
//  this->deform_to_prev = other.deform_to_prev;
//  this->deform_to_ref = other.deform_to_ref;
//  this->deform_from_prev = other.deform_from_prev;
//  this->deform_from_ref = other.deform_from_ref;

//  std::copy(other.transform_specs.begin(), other.transform_specs.end(), this->transform_specs.begin());
//  std::copy(other.full_res_label_trans_specs.begin(), other.full_res_label_trans_specs.end(),
//            this->full_res_label_trans_specs.begin());

//  return *this;
//}
}


