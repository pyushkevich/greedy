#ifndef PROPAGATIONDATA_H
#define PROPAGATIONDATA_H

#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <vtkSmartPointer.h>
#include <itkMatrixOffsetTransformBase.h>
#include <vtkPolyData.h>
#include "lddmm_data.h"
#include "PropagationAPI.h"
#include "PropagationCommon.h"


namespace propagation
{

template<typename TReal>
class PropagationMeshGroup
{
	PROPAGATION_DATA_TYPEDEFS
};

template<typename TReal>
class TimePointTransformSpec
{
public:
	PROPAGATION_DATA_TYPEDEFS

	TimePointTransformSpec(typename TTransform::Pointer _affine, typename TVectorImage3D::Pointer _deform,
												 unsigned int _crntTP)
		: affine(_affine), deform (_deform), currentTP(_crntTP) {}

  TimePointTransformSpec(const TimePointTransformSpec &other) = default;
  TimePointTransformSpec & operator=(const TimePointTransformSpec &other) = default;

	unsigned int currentTP;
	typename TTransform::Pointer affine;
	typename TVectorImage3D::Pointer deform;
};

template<typename TReal>
class TimePointData
{
public:
	PROPAGATION_DATA_TYPEDEFS

	TimePointData();
	~TimePointData();
  TimePointData(const TimePointData &other) = default;
  TimePointData &operator=(const TimePointData &other) = default;

	typename TImage3D::Pointer img;
	typename TImage3D::Pointer img_srs;
	typename TLabelImage3D::Pointer seg;
	typename TLabelImage3D::Pointer seg_srs;
	typename TLabelImage3D::Pointer full_res_mask;
	typename TTransform::Pointer affine_to_prev;
	typename TVectorImage3D::Pointer deform_to_prev;
	typename TVectorImage3D::Pointer deform_to_ref;
	typename TVectorImage3D::Pointer deform_from_prev;
	typename TVectorImage3D::Pointer deform_from_ref;
	std::vector<TimePointTransformSpec<TReal>> transform_specs;
	std::vector<TimePointTransformSpec<TReal>> full_res_label_trans_specs;
};

template<typename TReal>
class PropagationData
{
public:
	PROPAGATION_DATA_TYPEDEFS

	PropagationData();

  size_t GetNumberOfTimePoints();

  std::map<unsigned int, TimePointData<TReal>> tp_data;
	typename TImage4D::Pointer img4d;
  typename TLabelImage3D::Pointer seg_ref;
  std::string outdir;
	typename TImage3D::Pointer full_res_ref_space;
};

} // end of namespace propagation

#include "PropagationData.txx"

#endif // PROPAGATIONDATA_H
