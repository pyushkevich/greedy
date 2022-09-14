#ifndef PROPAGATIONTOOLS_H
#define PROPAGATIONTOOLS_H

#include "PropagationCommon.h"
#include "PropagationAPI.h"
#include <itkImageRegion.h>
#include <itkIndex.h>
#include <itkVTKImageExport.h>
#include <vtkImageImport.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageToImageFilter.h>

namespace propagation
{

template <typename TReal>
class PropagationTools
{
public:
	PROPAGATION_DATA_TYPEDEFS

	PropagationTools();
	~PropagationTools();

	static typename TImage3D::Pointer CastLabelToRealImage(TLabelImage3D *input);

	static typename TLabelImage3D::Pointer ResliceLabelImageWithIdentityMatrix(
			TImage3D *ref, TLabelImage3D *src);

	static TMeshPointer GetMeshFromLabelImage(TLabelImage3D *img);

	static typename TLabelImage3D::Pointer TrimLabelImage(TLabelImage3D *input, double vox);

	static void ExpandRegion(itk::ImageRegion<3> &region, const itk::Index<3> &idx);

	static void ConnectITKToVTK(itk::VTKImageExport<TLabelImage3D> *fltExport,vtkImageImport *fltImport);

	/**
	 * This static function constructs a NIFTI matrix from the ITK direction
	 * cosines matrix and Spacing and Origin vectors
	 */
	static vnl_matrix_fixed<double,4,4> ConstructNiftiSform(vnl_matrix<double> m_dir,
																													vnl_vector<double> v_origin,
																													vnl_vector<double> v_spacing);

	static vnl_matrix_fixed<double,4,4> ConstructVTKtoNiftiTransform(vnl_matrix<double> m_dir,
																																	 vnl_vector<double> v_origin,
																																	 vnl_vector<double> v_spacing);

	template<class TImage>
	static itk::SmartPointer<TImage> ReadImage(const std::string &filename);

	template<class TImage>
	static void WriteImage(TImage *img, const std::string &filename,
									itk::IOComponentEnum comp = itk::IOComponentEnum::UNKNOWNCOMPONENTTYPE);

	template<class TTimepointImage, class TFullImage>
	static typename TTimepointImage::Pointer ExtractTimePointImage(TFullImage *full_img, unsigned int tp);

	template<class TImage>
	static itk::SmartPointer<TImage> Resample3DImage(TImage *input, double factor,
																									 ResampleInterpolationMode intpMode, double smooth_sigma = 0);

	template<class TInputImage, class TOutputImage>
	static typename TOutputImage::Pointer
	ThresholdImage(TInputImage *img, typename TInputImage::PixelType lower, typename TInputImage::PixelType upper,
										typename TOutputImage::PixelType value_in, typename TOutputImage::PixelType value_out);

	template<class TInputImage, class TOutputImage>
	static typename TOutputImage::Pointer
	DilateImage(TInputImage *img, size_t radius, typename TInputImage::PixelType value);

	template <class TInputImage, class TIntensityMapping>
	static typename TCompositeImage3D::Pointer
	CastToCompositeImage(TInputImage *img);

	static typename TCompositeImage3D::Pointer CastImageToCompositeImage(TImage3D *img);

	inline static char GetPathSeparator()
	{
	#ifdef _WIN32
			return '\\';
	#else
			return '/';
	#endif
	}

};

template <class TInputImage, class TOutputImage, class TFunctor>
class UnaryFunctorImageToSingleComponentVectorImageFilter
		: public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
	typedef UnaryFunctorImageToSingleComponentVectorImageFilter<TInputImage, TOutputImage, TFunctor> Self;
	typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer< const Self >  ConstPointer;

	typedef TInputImage InputImageType;
	typedef TOutputImage OutputImageType;
	typedef TFunctor FunctorType;

	typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

	/** Run-time type information (and related methods). */
	itkTypeMacro(UnaryFunctorImageToSingleComponentVectorImageFilter, ImageToImageFilter)
	itkNewMacro(Self)

	/** ImageDimension constants */
	itkStaticConstMacro(InputImageDimension, unsigned int,
											TInputImage::ImageDimension);
	itkStaticConstMacro(OutputImageDimension, unsigned int,
											TOutputImage::ImageDimension);

	void SetFunctor(const FunctorType &functor)
	{
		if(m_Functor != functor)
			{
			m_Functor = functor;
			this->Modified();
			}
	}

	itkGetConstReferenceMacro(Functor, FunctorType)

	void DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;


protected:

	UnaryFunctorImageToSingleComponentVectorImageFilter() {}
	virtual ~UnaryFunctorImageToSingleComponentVectorImageFilter() {}

	FunctorType m_Functor;

};

template <typename TReal>
class LinearIntensityMapping
{
public:
	typedef LinearIntensityMapping Self;

	double operator() (TReal g) const
		{ return MapInternalToNative(g); }

	double MapInternalToNative(TReal internal) const
		{ return internal * scale + shift; }

	double MapNativeToInternal(TReal native) const
		{ return (native - shift) / scale; }

	LinearIntensityMapping() : scale(1.0), shift(0.0) {}
	LinearIntensityMapping(TReal a, TReal b) : scale(a), shift(b) {}

	bool operator != (const Self &other) const
		{ return scale != other.scale || shift != other.shift; }

protected:
	TReal scale;
	TReal shift;
};

template <typename TReal>
class IdentityIntensityMapping
{
public:

	TReal operator() (TReal g) const
		{ return g; }

	TReal MapGradientMagnitudeToNative(TReal internalGM) const
		{ return internalGM; }

	TReal MapInternalToNative(TReal internal) const
		{ return internal; }

	TReal MapNativeToInternal(TReal native) const
		{ return native; }

	virtual TReal GetScale() const { return 1; }
	virtual TReal GetShift() const { return 0; }

	bool IsIdentity() const
		{ return true; }

	bool operator != (const IdentityIntensityMapping &) const { return false; }
};


} // end of namespace propagation

#include "PropagationTools.txx"

#endif // PROPAGATIONTOOLS_H
