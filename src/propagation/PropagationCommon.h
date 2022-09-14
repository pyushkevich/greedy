#ifndef PROPAGATIONCOMMON_H
#define PROPAGATIONCOMMON_H

#define PROPAGATION_DATA_TYPEDEFS \
	using TImage4D = typename PropagationAPI<TReal>::TImage4D; \
	using TImage3D = typename PropagationAPI<TReal>::TImage3D; \
	using TLabelImage4D = typename PropagationAPI<TReal>::TLabelImage4D; \
	using TLabelImage3D = typename PropagationAPI<TReal>::TLabelImage3D; \
	using TLDDMM3D = typename PropagationAPI<TReal>::TLDDMM3D; \
	using TVectorImage3D = typename PropagationAPI<TReal>::TVectorImage3D; \
	using TCompositeImage3D = typename PropagationAPI<TReal>::TCompositeImage3D; \
	using TTransform = typename PropagationAPI<TReal>::TTransform; \
	using TMeshPointer = typename PropagationAPI<TReal>::TMeshPointer; \
	using ResampleInterpolationMode = typename PropagationAPI<TReal>::ResampleInterpolationMode;

#endif // PROPAGATIONCOMMON_H
