#include "PropagationTools.h"
#include "GreedyException.h"
#include "ImageRegionConstIteratorWithIndexOverride.h"
#include <itkImageRegionIterator.h>
#include <itkAffineTransform.h>
#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkComposeImageFilter.h>
#include <itkImageDuplicator.h>
#include <vtkAppendPolyData.h>
#include <vtkDiscreteMarchingCubes.h>
#include <vtkShortArray.h>
#include <vtkPointData.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTransform.h>




namespace propagation
{

template<typename TReal>
PropagationTools<TReal>
::PropagationTools()
{

}

template<typename TReal>
PropagationTools<TReal>
::~PropagationTools()
{

}

template<typename TReal>
typename PropagationTools<TReal>::TImage3D::Pointer
PropagationTools<TReal>
::CastLabelToRealImage(TLabelImage3D *input)
{
	auto output = TImage3D::New();
	output->SetRegions(input->GetLargestPossibleRegion());
	output->SetDirection(input->GetDirection());
	output->SetOrigin(input->GetOrigin());
	output->SetSpacing(input->GetSpacing());
	output->Allocate();

	itk::ImageRegionIterator<TLabelImage3D> it_input(
				input, input->GetLargestPossibleRegion());
	itk::ImageRegionIterator<TImage3D> it_output(
				output, output->GetLargestPossibleRegion());

	// Deep copy pixels
	while (!it_input.IsAtEnd())
		{
		it_output.Set(it_input.Get());
		++it_output;
		++it_input;
		}

	return output;
}

template<typename TReal>
typename PropagationTools<TReal>::TLabelImage3D::Pointer
PropagationTools<TReal>
::ResliceLabelImageWithIdentityMatrix(TImage3D *ref, TLabelImage3D *src)
{
	// Code adapted from c3d command -reslice-identity
	typedef itk::AffineTransform<double, 3> TranType;
	typename TranType::Pointer atran = TranType::New();
	atran->SetIdentity();

	// Build the resampling filter
	typedef itk::ResampleImageFilter<TLabelImage3D, TLabelImage3D> ResampleFilterType;
	typename ResampleFilterType::Pointer fltSample = ResampleFilterType::New();

	fltSample->SetTransform(atran);

	// Initialize the resampling filter with an identity transform
	fltSample->SetInput(src);

	// Set the unknown intensity to positive value
	fltSample->SetDefaultPixelValue(0);

	// Set the interpolator
	typedef itk::NearestNeighborInterpolateImageFunction<TLabelImage3D, double> NNInterpolator;
	fltSample->SetInterpolator(NNInterpolator::New());

	// Calculate where the transform is taking things
	itk::ContinuousIndex<double, 3> idx[3];
	for(size_t i = 0; i < 3; i++)
		{
		idx[0][i] = 0.0;
		idx[1][i] = ref->GetBufferedRegion().GetSize(i) / 2.0;
		idx[2][i] = ref->GetBufferedRegion().GetSize(i) - 1.0;
		}
	for(size_t j = 0; j < 3; j++)
		{
		itk::ContinuousIndex<double, 3> idxmov;
		itk::Point<double, 3> pref, pmov;
		ref->TransformContinuousIndexToPhysicalPoint(idx[j], pref);
		pmov = atran->TransformPoint(pref);
		src->TransformPhysicalPointToContinuousIndex(pmov, idxmov);
		}

	// Set the spacing, origin, direction of the output
	fltSample->UseReferenceImageOn();
	fltSample->SetReferenceImage(ref);
	fltSample->Update();

	return fltSample->GetOutput();
}

template<typename TReal>
void
PropagationTools<TReal>
::ConnectITKToVTK(itk::VTKImageExport<TLabelImage3D> *fltExport,vtkImageImport *fltImport)
{
	fltImport->SetUpdateInformationCallback( fltExport->GetUpdateInformationCallback());
	fltImport->SetPipelineModifiedCallback( fltExport->GetPipelineModifiedCallback());
	fltImport->SetWholeExtentCallback( fltExport->GetWholeExtentCallback());
	fltImport->SetSpacingCallback( fltExport->GetSpacingCallback());
	fltImport->SetOriginCallback( fltExport->GetOriginCallback());
	fltImport->SetScalarTypeCallback( fltExport->GetScalarTypeCallback());
	fltImport->SetNumberOfComponentsCallback( fltExport->GetNumberOfComponentsCallback());
	fltImport->SetPropagateUpdateExtentCallback( fltExport->GetPropagateUpdateExtentCallback());
	fltImport->SetUpdateDataCallback( fltExport->GetUpdateDataCallback());
	fltImport->SetDataExtentCallback( fltExport->GetDataExtentCallback());
	fltImport->SetBufferPointerCallback( fltExport->GetBufferPointerCallback());
	fltImport->SetCallbackUserData( fltExport->GetCallbackUserData());
}

template<typename TReal>
typename PropagationTools<TReal>::TMeshPointer
PropagationTools<TReal>
::GetMeshFromLabelImage(TLabelImage3D *img)
{
	short imax = img->GetBufferPointer()[0];
	short imin = imax;
	for(size_t i = 0; i < img->GetBufferedRegion().GetNumberOfPixels(); i++)
		{
		short x = img->GetBufferPointer()[i];
		imax = std::max(imax, x);
		imin = std::min(imin, x);
		}

	typedef itk::VTKImageExport<TLabelImage3D> ExporterType;
	typename ExporterType::Pointer fltExport = ExporterType::New();
	fltExport->SetInput(img);
	vtkImageImport *fltImport = vtkImageImport::New();
	ConnectITKToVTK(fltExport.GetPointer(), fltImport);

	// Append filter for assembling labels
	vtkAppendPolyData *fltAppend = vtkAppendPolyData::New();

	// Extracting one label at a time and assigning label value
	for (short i = 1; i <= imax; i += 1.0)
		{
		// Extract one label
		vtkDiscreteMarchingCubes *fltDMC = vtkDiscreteMarchingCubes::New();
		fltDMC->SetInputConnection(fltImport->GetOutputPort());
		fltDMC->ComputeGradientsOff();
		fltDMC->ComputeScalarsOff();
		fltDMC->SetNumberOfContours(1);
		fltDMC->ComputeNormalsOn();
		fltDMC->SetValue(0, i);
		fltDMC->Update();

		vtkPolyData *labelMesh = fltDMC->GetOutput();

		// Set scalar values for the label
		vtkShortArray *scalar = vtkShortArray::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(labelMesh->GetNumberOfPoints());
		scalar->Fill(i);
		scalar->SetName("Label");
		labelMesh->GetPointData()->SetScalars(scalar);
		fltAppend->AddInputData(labelMesh);
		}

	fltAppend->Update();

	// Compute the transform from VTK coordinates to NIFTI/RAS coordinates
	// Create the transform filter
	vtkTransformPolyDataFilter *fltTransform = vtkTransformPolyDataFilter::New();
	fltTransform->SetInputData(fltAppend->GetOutput());

	typedef vnl_matrix_fixed<double, 4, 4> Mat44;
	Mat44 vtk2out;
	Mat44 vtk2nii = ConstructVTKtoNiftiTransform(
		img->GetDirection().GetVnlMatrix().as_ref(),
		img->GetOrigin().GetVnlVector(),
		img->GetSpacing().GetVnlVector());

	vtk2out = vtk2nii;

	// Update the VTK transform to match
	vtkTransform *transform = vtkTransform::New();
	transform->SetMatrix(vtk2out.data_block());
	fltTransform->SetTransform(transform);
	fltTransform->Update();

	// Get final output
	return fltTransform->GetOutput();
}

template<typename TReal>
typename PropagationTools<TReal>::TLabelImage3D::Pointer
PropagationTools<TReal>
::TrimLabelImage(TLabelImage3D *input, double vox, typename TLabelImage3D::RegionType &roi)
{
	typedef typename TLabelImage3D::RegionType RegionType;
	typedef itk::ImageRegionIteratorWithIndex<TLabelImage3D> Iterator;

	// Initialize the bounding box
  RegionType bbox;

	// Find the extent of the non-background region of the image
	Iterator it(input, input->GetBufferedRegion());
	for( ; !it.IsAtEnd(); ++it)
		if(it.Value() != 0)
      ExpandRegion(bbox, it.GetIndex());

	typename TLabelImage3D::SizeType radius;
	for(size_t i = 0; i < 3; i++)
		radius[i] = (int) ceil(vox);
  bbox.PadByRadius(radius);

	// Make sure the bounding box is within the contents of the image
  bbox.Crop(input->GetBufferedRegion());

	// Chop off the region
	typedef itk::RegionOfInterestImageFilter<TLabelImage3D, TLabelImage3D> TrimFilter;
	typename TrimFilter::Pointer fltTrim = TrimFilter::New();
	fltTrim->SetInput(input);
	fltTrim->SetRegionOfInterest(bbox);
	fltTrim->Update();

  // Copy bounding box to output roi
  roi = bbox;

	return fltTrim->GetOutput();
}

template<typename TReal>
void
PropagationTools<TReal>
::ExpandRegion(itk::ImageRegion<3> &region, const itk::Index<3> &idx)
{
	if(region.GetNumberOfPixels() == 0)
		{
		region.SetIndex(idx);
		for(size_t i = 0; i < 3; i++)
			region.SetSize(i, 1);
		}
	else {
		for(size_t i = 0; i < 3; i++)
			{
			if(region.GetIndex(i) > idx[i])
				{
				region.SetSize(i, region.GetSize(i) + (region.GetIndex(i) - idx[i]));
				region.SetIndex(i, idx[i]);
				}
			else if(region.GetIndex(i) + (long) region.GetSize(i) <= idx[i]) {
				region.SetSize(i, 1 + idx[i] - region.GetIndex(i));
				}
			}
	}
}


template<typename TReal>
vnl_matrix_fixed<double,4,4>
PropagationTools<TReal>
::ConstructNiftiSform(vnl_matrix<double> m_dir, vnl_vector<double> v_origin,
											vnl_vector<double> v_spacing)
{
	// Set the NIFTI/RAS transform
	vnl_matrix<double> m_ras_matrix;
	vnl_diag_matrix<double> m_scale, m_lps_to_ras;
	vnl_vector<double> v_ras_offset;

	// Compute the matrix
	m_scale.set(v_spacing);
	m_lps_to_ras.set(vnl_vector<double>(3, 1.0));
	m_lps_to_ras[0] = -1;
	m_lps_to_ras[1] = -1;
	m_ras_matrix = m_lps_to_ras * m_dir * m_scale;

	// Compute the vector
	v_ras_offset = m_lps_to_ras * v_origin;

	// Create the larger matrix
	vnl_vector<double> vcol(4, 1.0);
	vcol.update(v_ras_offset);

	vnl_matrix_fixed<double,4,4> m_sform;
	m_sform.set_identity();
	m_sform.update(m_ras_matrix);
	m_sform.set_column(3, vcol);
	return m_sform;
}

template<typename TReal>
vnl_matrix_fixed<double,4,4>
PropagationTools<TReal>
::ConstructVTKtoNiftiTransform(vnl_matrix<double> m_dir, vnl_vector<double> v_origin,
															 vnl_vector<double> v_spacing)
{
	vnl_matrix_fixed<double,4,4> vox2nii = ConstructNiftiSform(m_dir, v_origin, v_spacing);
	vnl_matrix_fixed<double,4,4> vtk2vox;
	vtk2vox.set_identity();
	for(size_t i = 0; i < 3; i++)
		{
		vtk2vox(i,i) = 1.0 / v_spacing[i];
		vtk2vox(i,3) = - v_origin[i] / v_spacing[i];
		}
	return vox2nii * vtk2vox;
}

template<typename TReal>
template<class TImage>
itk::SmartPointer<TImage>
PropagationTools<TReal>
::ReadImage(const std::string &filename)
{
	using TReader = itk::ImageFileReader<TImage>;
	typename TReader::Pointer reader = TReader::New();
	reader->SetFileName(filename.c_str());
	reader->Update();
	return reader->GetOutput();
}

template<typename TReal>
template<class TImage>
void
PropagationTools<TReal>
::WriteImage(TImage *img, const std::string &filename, itk::IOComponentEnum comp)
{
	if(dynamic_cast<TVectorImage3D *>(img))
		TLDDMM3D::vimg_write(dynamic_cast<TVectorImage3D *>(img), filename.c_str(), comp);
	else if(dynamic_cast<typename TLDDMM3D::ImageType *>(img))
		TLDDMM3D::img_write(dynamic_cast<typename TLDDMM3D::ImageType *>(img), filename.c_str(), comp);
	else if(dynamic_cast<typename TLDDMM3D::CompositeImageType *>(img))
		TLDDMM3D::cimg_write(dynamic_cast<typename TLDDMM3D::CompositeImageType *>(img), filename.c_str(), comp);
	else
		{
		// Some other type (e.g., LabelImage). We use the image writer and ignore the comp
		typedef itk::ImageFileWriter<TImage> WriterType;
		typename WriterType::Pointer writer = WriterType::New();
		writer->SetFileName(filename.c_str());
		writer->SetUseCompression(true);
		writer->SetInput(img);
		writer->Update();
		}
}

template<typename TReal>
template<class TTimePointImage, class TFullImage>
typename TTimePointImage::Pointer
PropagationTools<TReal>
::ExtractTimePointImage(TFullImage *full_img, unsigned int tp)
{
	// Logic adapated from SNAP ImageWrapper method:
	// ConfigureTimePointImageFromImage4D()
	// Always use 1-based index for time point
	assert(tp > 0);

	unsigned int nt = full_img->GetBufferedRegion().GetSize()[3u];
	unsigned int bytes_per_volume = full_img->GetPixelContainer()->Size() / nt;

	typename TImage3D::Pointer tp_img = TImage3D::New();

	typename TImage3D::RegionType region;
	typename TImage3D::SpacingType spacing;
	typename TImage3D::PointType origin;
	typename TImage3D::DirectionType dir;
	for(unsigned int j = 0; j < 3; j++)
		{
		region.SetSize(j, full_img->GetBufferedRegion().GetSize()[j]);
		region.SetIndex(j, full_img->GetBufferedRegion().GetIndex()[j]);
		spacing[j] = full_img->GetSpacing()[j];
		origin[j] = full_img->GetOrigin()[j];
		for(unsigned int k = 0; k < 3; k++)
			dir(j,k) = full_img->GetDirection()(j,k);
		}

	// All of the information from the 4D image is propagaged to the 3D timepoints
	tp_img->SetRegions(region);
	tp_img->SetSpacing(spacing);
	tp_img->SetOrigin(origin);
	tp_img->SetDirection(dir);
	tp_img->SetNumberOfComponentsPerPixel(full_img->GetNumberOfComponentsPerPixel());
	tp_img->Allocate();

	// Set the buffer pointer
	tp_img->GetPixelContainer()->SetImportPointer(
				full_img->GetBufferPointer() + bytes_per_volume * (tp - 1),
				bytes_per_volume);

	return tp_img;
}

template<typename TReal>
template<class TImage>
itk::SmartPointer<TImage>
PropagationTools<TReal>
::Resample3DImage(TImage* input, double factor,
									ResampleInterpolationMode intpMode, double smooth_sigma)
{
	typedef itk::DiscreteGaussianImageFilter<TImage,TImage> SmoothFilter;
	typename TImage::Pointer imageToResample = input;

	// Smooth image if needed
	if (smooth_sigma > 0)
		{
		typename SmoothFilter::Pointer fltDSSmooth = SmoothFilter::New();
		typename SmoothFilter::ArrayType variance;
		for (int i = 0; i < 3; ++i)
			variance[i] = smooth_sigma * smooth_sigma;

		fltDSSmooth->SetInput(input);
		fltDSSmooth->SetVariance(variance);
		fltDSSmooth->UseImageSpacingOn();
		fltDSSmooth->Update();
		imageToResample = fltDSSmooth->GetOutput();
		}

	// Create resampled images
	typedef itk::ResampleImageFilter<TImage, TImage> ResampleFilter;
	typedef itk::LinearInterpolateImageFunction<TImage, double> LinearInterpolator;
	typedef itk::NearestNeighborInterpolateImageFunction<TImage, double> NNInterpolator;

	typename ResampleFilter::Pointer fltResample = ResampleFilter::New();
	fltResample->SetInput(imageToResample);
	fltResample->SetTransform(itk::IdentityTransform<double, 3u>::New());

	switch (intpMode)
		{
		case ResampleInterpolationMode::Linear:
			fltResample->SetInterpolator(LinearInterpolator::New());
			break;
		case ResampleInterpolationMode::NearestNeighbor:
			fltResample->SetInterpolator(NNInterpolator::New());
			break;
		default:
			throw GreedyException("Unkown Interpolation Mode");
		}

	typename TImage::SizeType sz;
	for(size_t i = 0; i < 3; i++)
		sz[i] = (unsigned long)(imageToResample->GetBufferedRegion().GetSize(i) * factor + 0.5);

	// Compute the spacing of the new image
	typename TImage::SpacingType spc_pre = imageToResample->GetSpacing();
	typename TImage::SpacingType spc_post = spc_pre;
	for(size_t i = 0; i < 3; i++)
		spc_post[i] *= imageToResample->GetBufferedRegion().GetSize()[i] * 1.0 / sz[i];

	// Get the bounding box of the input image
	typename TImage::PointType origin_pre = imageToResample->GetOrigin();

	// Recalculate the origin. The origin describes the center of voxel 0,0,0
	// so that as the voxel size changes, the origin will change as well.
	typename TImage::SpacingType off_pre = (imageToResample->GetDirection() * spc_pre) * 0.5;
	typename TImage::SpacingType off_post = (imageToResample->GetDirection() * spc_post) * 0.5;
	typename TImage::PointType origin_post = origin_pre - off_pre + off_post;

	// Set the image sizes and spacing.
	fltResample->SetSize(sz);
	fltResample->SetOutputSpacing(spc_post);
	fltResample->SetOutputOrigin(origin_post);
	fltResample->SetOutputDirection(imageToResample->GetDirection());

	// Set the unknown intensity to positive value
	fltResample->SetDefaultPixelValue(0);

	// Perform resampling
	fltResample->UpdateLargestPossibleRegion();

	return fltResample->GetOutput();
}

template<typename TReal>
template<class TInputImage, class TOutputImage>
typename TOutputImage::Pointer
PropagationTools<TReal>
::ThresholdImage(TInputImage *img, typename TInputImage::PixelType lower, typename TInputImage::PixelType upper,
									typename TOutputImage::PixelType value_in, typename TOutputImage::PixelType value_out)
{
	using ThresholdFilter = itk::BinaryThresholdImageFilter<TLabelImage3D, TLabelImage3D>;
	typename ThresholdFilter::Pointer fltThreshold = ThresholdFilter::New();
	fltThreshold->SetInput(img);
	fltThreshold->SetLowerThreshold(lower);
	fltThreshold->SetUpperThreshold(upper);
	fltThreshold->SetInsideValue(value_in);
	fltThreshold->SetOutsideValue(value_out);
	fltThreshold->Update();

	return fltThreshold->GetOutput();
}

template<typename TReal>
template<class TInputImage, class TOutputImage>
typename TOutputImage::Pointer
PropagationTools<TReal>
::DilateImage(TInputImage *img, size_t radius, typename TInputImage::PixelType value)
{
	// Label dilation
	using Element = itk::BinaryBallStructuringElement<TReal, 3u> ;
	typename Element::SizeType sz = { radius, radius, radius };
	Element elt;
	elt.SetRadius(sz);
	elt.CreateStructuringElement();

	typedef itk::BinaryDilateImageFilter<TInputImage, TOutputImage, Element> DilateFilter;
	typename DilateFilter::Pointer fltDilation = DilateFilter::New();
	fltDilation->SetInput(img);
	fltDilation->SetDilateValue(value);
	fltDilation->SetKernel(elt);
	fltDilation->Update();

	return fltDilation->GetOutput();
}

template<typename TReal>
template <class TInputImage, class TIntensityMapping>
typename PropagationTools<TReal>::TCompositeImage3D::Pointer
PropagationTools<TReal>
::CastToCompositeImage(TInputImage *img)
{
	typedef UnaryFunctorImageToSingleComponentVectorImageFilter<
			TInputImage, TCompositeImage3D, TIntensityMapping> FilterType;
	typedef itk::ImageSource<TCompositeImage3D> VectorImageSource;

	TIntensityMapping intensityMapping;
	itk::SmartPointer<FilterType> filter = FilterType::New();
	filter->SetInput(img);
	filter->SetFunctor(intensityMapping);
	itk::SmartPointer<VectorImageSource> imgSource = filter.GetPointer();
	imgSource->UpdateOutputInformation();
	imgSource->Update();
	return imgSource->GetOutput();
}

template<typename TReal>
typename PropagationTools<TReal>::TCompositeImage3D::Pointer
PropagationTools<TReal>
::CastImageToCompositeImage(TImage3D *img)
{
	auto flt = itk::ComposeImageFilter<TImage3D>::New();
	flt->SetInput(0, img);
	flt->Update();
	return flt->GetOutput();
}

template<typename TReal>
template<class TImage>
typename TImage::Pointer
PropagationTools<TReal>
::CreateEmptyImage(TImage *sample)
{
  auto duplicator = itk::ImageDuplicator<TImage>::New();
  duplicator->SetInputImage(sample);
  duplicator->Update();
  auto imgout = duplicator->GetOutput();
  imgout->FillBuffer(itk::NumericTraits<typename TImage::PixelType>::Zero);
  return imgout;
}


template <class TInputImage, class TOutputImage, class TFunctor>
void
UnaryFunctorImageToSingleComponentVectorImageFilter<TInputImage, TOutputImage, TFunctor>
::DynamicThreadedGenerateData(const OutputImageRegionType &outputRegionForThread)
{
	// Use our fast iterators for vector images
	typedef itk::ImageLinearIteratorWithIndex<OutputImageType> IterBase;
	typedef IteratorExtender<IterBase> IterType;

	typedef typename OutputImageType::InternalPixelType OutputComponentType;
	typedef typename InputImageType::InternalPixelType InputComponentType;

	// Define the iterators
	IterType outputIt(this->GetOutput(), outputRegionForThread);
	int line_len = outputRegionForThread.GetSize(0);

	// Using a generic ITK iterator for the input because it supports RLE images and adaptors
	itk::ImageScanlineConstIterator< InputImageType > inputIt(this->GetInput(), outputRegionForThread);

	while ( !inputIt.IsAtEnd() )
		{
		// Get the pointer to the input and output pixel lines
		OutputComponentType *out = outputIt.GetPixelPointer(this->GetOutput());

		for(int i = 0; i < line_len; i++, ++inputIt)
			{
			out[i] = m_Functor(inputIt.Get());
			}

		outputIt.NextLine();
		inputIt.NextLine();
		}
}



} // end of namespace propagation

