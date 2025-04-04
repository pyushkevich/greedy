/*=========================================================================

  Program:   picsl_greedy: Greedy Python Bindings
  Module:    GreedyPythonBindings.cxx
  Language:  C++
  Website:   https://greedy.readthedocs.io/
  Copyright (c) 2024 Paul A. Yushkevich

  This file is part of Greedy, a command-line companion registration tool

  Greedy is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#include <GreedyAPI.h>
#include <CommandLineHelper.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/iostream.h"
#include <iostream>
#include <itkImage.h>
#include <itkMatrixOffsetTransformBase.h>
#include <itkMetaDataObject.h>
#include <MultiChunkGreedy.h>
#include <PointSetGeodesicShooting.h>
#include <PointSetGeodesicToWarp.h>

namespace py=pybind11;

using namespace std;

/**
 * This class handles importing data and metadata from SimpleITK images
 */
template <typename TImage>
class ImageImport
{
public:
  using ImageType = TImage;
  using ComponentType = typename ImageType::InternalPixelType;
  using RegionType = typename ImageType::RegionType;
  using SpacingType = typename ImageType::SpacingType;
  using PointType = typename ImageType::PointType;

  using ImportArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
  static constexpr unsigned int VDim = ImageType::ImageDimension;

  ImageImport(py::object sitk_image)
  {
    // Confirm that the image is of correct type
    py::object sitk = py::module_::import("SimpleITK");
    if(!py::isinstance(sitk_image, sitk.attr("Image")))
      throw std::runtime_error("Input not a SimpleITK image!");

    // Check that the number of components is correct
    unsigned int ncomp = sitk_image.attr("GetNumberOfComponentsPerPixel")().cast<int>();
    if(ncomp != 1)
      throw std::runtime_error("Vector images are not supported!");

    // Extract the image array from the image
    py::object arr = sitk.attr("GetArrayFromImage")(sitk_image);
    ImportArray arr_c = arr.cast<ImportArray>();

    // Check that the image dimensions are correct
    py::buffer_info arr_info = arr_c.request();
    if(arr_info.ndim != VDim)
      throw std::runtime_error("Incompatible array dimensions!");

    // Get image spacing, origin, direction
    std::array<double, VDim> spacing = sitk_image.attr("GetSpacing")().cast< std::array<double, VDim> >();
    std::array<double, VDim> origin = sitk_image.attr("GetOrigin")().cast< std::array<double, VDim> >();
    std::array<double, VDim*VDim> dir = sitk_image.attr("GetDirection")().cast< std::array<double, VDim*VDim> >();

    typename ImageType::RegionType itk_region;
    typename ImageType::SpacingType itk_spacing;
    typename ImageType::PointType itk_origin;
    typename ImageType::DirectionType itk_dir;
    int q = 0;
    for(unsigned int i = 0; i < arr_info.ndim; i++)
    {
      itk_region.SetSize(i, arr_info.shape[(VDim-1) - i]); // Shape is reversed between ITK and Numpy
      itk_spacing[i] = spacing[i];
      itk_origin[i] = origin[i];
      for(unsigned int j = 0; j < VDim; j++)
        itk_dir[i][j] = dir[q++];
    }

    this->image = ImageType::New();
    this->image->SetRegions(itk_region);
    this->image->SetOrigin(itk_origin);
    this->image->SetSpacing(itk_spacing);
    this->image->SetDirection(itk_dir);
    this->image->SetNumberOfComponentsPerPixel(ncomp);

    // We have to make a copy of the buffer, otherwise the memory may get deallocated
    // TODO: in the future, maybe avoid data duplication?
    ComponentType *ptr_copy = new ComponentType[arr_info.size];
    memcpy(ptr_copy, arr_c.data(), sizeof(ComponentType) * arr_info.size);
    this->image->GetPixelContainer()->SetImportPointer(ptr_copy, arr_info.size, true);

    for (const auto &key : sitk_image.attr("GetMetaDataKeys")()) 
    {
      const auto &value = sitk_image.attr("GetMetaData")(key);
      itk::EncapsulateMetaData<std::string>(
        this->image->GetMetaDataDictionary(), 
        std::string(py::str(key)).c_str(), 
        std::string(py::str(value)).c_str());
    }
  }

  std::string GetInfoString() {
    std::ostringstream oss;
    image->Print(oss);
    return oss.str();
  }

  ImageType* GetImage() const { return image; }

private:
  typename ImageType::Pointer image;
};


template <typename TTransform, typename TPixel>
class AffineTransformImport
{
public:
  using TransformType = TTransform;
  using MatrixType = typename TransformType::MatrixType;
  using OffsetType = typename TransformType::OffsetType;
  using ImportArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
  static constexpr unsigned int VDim = TransformType::InputSpaceDimension;
  using Greedy = GreedyApproach<VDim, TPixel>;

  AffineTransformImport(ImportArray arr)
  {
    if(arr.ndim() != 2 || arr.shape(0) != VDim+1 || arr.shape(0) != VDim+1)
      throw std::runtime_error("Incorrect array dimensions for affine transform");

    vnl_matrix<double> Q(VDim+1, VDim+1);
    for(size_t r = 0; r < VDim; r++)
      for(size_t c = 0; c < VDim; c++)
        Q(r,c) = arr.at(r,c);

    typedef itk::MatrixOffsetTransformBase<double, VDim, VDim> TransformType;
    typename TransformType::Pointer tran = TransformType::New();
    Greedy::MapRASMatrixToITKTransform(Q, tran.GetPointer());
  }

  std::string GetInfoString() {
    std::ostringstream oss;
    tran->Print(oss);
    return oss.str();
  }

  TransformType* GetTransform() const { return tran; }

private:
  typename TransformType::Pointer tran;
};


template <typename TImage>
class ImageExport
{
public:
  using ImageType = TImage;
  using ComponentType = typename ImageType::InternalPixelType;
  using RegionType = typename ImageType::RegionType;
  using SpacingType = typename ImageType::SpacingType;
  using PointType = typename ImageType::PointType;
  static constexpr unsigned int VDim = ImageType::ImageDimension;

  ImageExport(ImageType* image)
  {
    // SimpleITK 
    py::object sitk = py::module_::import("SimpleITK");

    // Create a numpy array from the image buffer
    unsigned int ncomp = image->GetNumberOfComponentsPerPixel();
    if(ncomp > 1)
    {
      std::vector<py::ssize_t> shape(VDim + 1);
      for(unsigned int i = 0; i < VDim; i++)
        shape[i] = image->GetBufferedRegion().GetSize((VDim-1) - i);
      shape[VDim] = ncomp;
      py::buffer_info bi(
        image->GetBufferPointer(), sizeof(ComponentType),
        py::format_descriptor<ComponentType>::format(),
        shape.size(), shape,
        py::detail::c_strides(shape, sizeof(ComponentType)));
      py::array arr(bi);

      // Generate a simple ITK image from this
      this->sitk_image = sitk.attr("GetImageFromArray")(arr, true);
    }
    else
    {
      std::vector<py::ssize_t> shape(VDim);
      for(unsigned int i = 0; i < VDim; i++)
        shape[i] = image->GetBufferedRegion().GetSize((VDim-1) - i);
      py::buffer_info bi(
        image->GetBufferPointer(), sizeof(ComponentType),
        py::format_descriptor<ComponentType>::format(),
        shape.size(), shape,
        py::detail::c_strides(shape, sizeof(ComponentType)));
      py::array arr(bi);

      // Generate a simple ITK image from this
      this->sitk_image = sitk.attr("GetImageFromArray")(arr, false);
    }

    // Update the spacing, etc
    std::array<double, VDim> spacing, origin;
    std::array<double, VDim*VDim> dir;
    for(unsigned int i = 0, q = 0; i < VDim; i++)
    {
      spacing[i] = image->GetSpacing()[i];
      origin[i] = image->GetOrigin()[i];
      for(unsigned int j = 0; j < VDim; j++, q++)
        dir[q] = image->GetDirection()[i][j];
    }
    this->sitk_image.attr("SetSpacing")(spacing);
    this->sitk_image.attr("SetOrigin")(origin);
    this->sitk_image.attr("SetDirection")(dir);
  }

  py::object sitk_image;
};


template <typename TTransform, typename TPixel>
class AffineTransformExport
{
public:
  using TransformType = TTransform;
  using MatrixType = typename TransformType::MatrixType;
  using OffsetType = typename TransformType::OffsetType;

  static constexpr unsigned int VDim = TransformType::InputSpaceDimension;
  using Greedy = GreedyApproach<VDim, TPixel>;

  AffineTransformExport(TransformType* tran)
  {
    vnl_matrix<double> Q = Greedy::MapITKTransformToRASMatrix(tran);
    arr = py::array_t<double>({ VDim+1, VDim+1 });
    for(unsigned int r = 0; r < VDim + 1; r++)
      for(unsigned int c = 0; c < VDim + 1; c++)
        arr.mutable_at(r,c) = Q(r,c);
  }

  py::array_t<double> arr;
};


template <typename TPixel, unsigned int VDim>
class GreedyAPIWrapper
{
public:
  using Greedy = GreedyApproach<VDim, TPixel>;
  using LDDMMType = typename Greedy::LDDMMType;
  using ImageType = typename Greedy::ImageType;
  using CompositeImageType = typename Greedy::CompositeImageType;
  using CompositeImagePointer = typename Greedy::CompositeImagePointer;
  using TransformType = typename Greedy::LinearTransformIOType;

  void Execute(
    const string &cmd, py::object sout, py::object serr, const py::kwargs& kwargs)
  {
    // Redirect the outputs if needed
    py::scoped_ostream_redirect r_out(std::cout, sout);
    py::scoped_ostream_redirect r_err(std::cerr, serr);

    // From kwargs, assign images to pass as inputs and outputs
    for(auto it : kwargs)
      SetCachedObject(it.first.cast<std::string>(), it.second.cast<py::object>());

    // Parse the command line to generate parameters
    CommandLineHelper cl(cmd.c_str());
    cl.set_file_check_bypass_labels(api.GetCachedObjectNames());
    GreedyParameters param = greedy_parse_commandline(cl, false);

    // Run the algorithm
    api.Run(param);
  }

  py::object GetCachedObject(std::string label)
  {
    using ImageBase = itk::ImageBase<VDim>;
    itk::Object *object = api.GetCachedObject(label);

    if(auto *ibase = dynamic_cast<ImageBase *>(object))
    {
      CompositeImagePointer cimg = LDDMMType::as_cimg(ibase);
      if(cimg)
        return ImageExport<CompositeImageType>(cimg).sitk_image;
    }
    else if(auto *tform = dynamic_cast<TransformType *>(object))
    {
      return AffineTransformExport<TransformType, TPixel>(tform).arr;
    }

    return py::none();
  }

  void SetCachedObject(std::string label, py::object object)
  {
    py::object sitk = py::module_::import("SimpleITK");
    using nparray = py::array_t<double, py::array::c_style | py::array::forcecast>;

    if(object.is_none())
    {
      // Pass as an empty input
      api.AddCachedOutputObject(label, (itk::Object *) nullptr);
    }
    else if(py::isinstance(object, sitk.attr("Image")))
    {
      // Pass as a "real" input
      ImageImport<CompositeImageType> import(object);
      api.AddCachedInputObject(label, import.GetImage());
    }
    else if(auto arr = object.cast<nparray>())
    {
      AffineTransformImport<TransformType, TPixel> import(arr);
      api.AddCachedInputObject(label, import.GetTransform());
    }
  }

  py::list GetMetricReport()
  {
    auto metric_log = api.GetMetricLog();
    auto last_log = api.GetLastMetricReport();
    size_t n_comp = last_log.ComponentPerPixelMetrics.size();
    py::list out_report;
    for(auto &it_level: metric_log)
    {
      py::dict out_level;
      py::array_t<double> total_metric(it_level.size());
      py::array_t<double> mask_volume(it_level.size());
      py::array_t<double> percomp_metric({it_level.size(), n_comp});
      for(int i = 0; i < it_level.size(); i++)
      {
        total_metric.mutable_at(i) = it_level[i].TotalPerPixelMetric;
        mask_volume.mutable_at(i) = it_level[i].MaskVolume;
        for(int j = 0; j < n_comp; j++)
          percomp_metric.mutable_at(i, j) = it_level[i].ComponentPerPixelMetrics[j];

      }
      out_level["TotalPerPixelMetric"] = total_metric;
      out_level["ComponentPerPixelMetrics"] = percomp_metric;
      out_level["MaskVolume"] = mask_volume;
      out_report.append(out_level);
    }
    return out_report;
  }

protected:
  Greedy api;
};



template <typename TPixel, unsigned int VDim>
void instantiate_greedy(py::handle m, const char *name)
{
  using Greedy = GreedyAPIWrapper<TPixel, VDim>;
  py::class_<Greedy>(m, name, "Python API for the PICSL greedy tool")
    .def(py::init<>([]() {
      auto *c = new Greedy();
      return c;
    }))
    .def("execute", &Greedy::Execute,
         "Execute one or more commands using the greedy command line interface",
         py::arg("command"),
         py::arg("out") = py::module_::import("sys").attr("stdout"),
         py::arg("err") = py::module_::import("sys").attr("stdout"))
    .def("__getitem__", &Greedy::GetCachedObject)
    .def("__setitem__", &Greedy::SetCachedObject)
    .def("metric_log", &Greedy::GetMetricReport)
    ;
}



template <unsigned int VDim>
class GreedyMultiChunkWrapper
{
public:
  void Execute(
    const string &cmd, py::object sout, py::object serr, const py::kwargs& kwargs)
  {
    // Redirect the outputs if needed
    py::scoped_ostream_redirect r_out(std::cout, sout);
    py::scoped_ostream_redirect r_err(std::cerr, serr);

           // Parse the command line to generate parameters
    CommandLineHelper cl(cmd.c_str());
    auto [chunk_greedy_param, greedy_param] = greedy_multi_chunk_parse_parameters(cl, false);
    run_multichunk_greedy<VDim>(chunk_greedy_param, greedy_param);
  }
};

template <unsigned int VDim>
void instantiate_multichunk_greedy(py::handle m, const char *name)
{
  using API = GreedyMultiChunkWrapper<VDim>;
  py::class_<API>(m, name, "Python API for the PICSL multi-chunk greedy tool")
    .def(py::init<>([]() {
      auto *c = new API();
      return c;
    }))
    .def("run", &API::Execute,
         "Execute the multi-chunk greedy command",
         py::arg("command"),
         py::arg("out") = py::module_::import("sys").attr("stdout"),
         py::arg("err") = py::module_::import("sys").attr("stdout"))
    ;
}


template <typename TPixel, unsigned int VDim>
class LMShootAPIWrapper
{
public:

  void ExecuteFit(
    const string &cmd, py::object sout, py::object serr, const py::kwargs& kwargs)
  {
    // Redirect the outputs if needed
    py::scoped_ostream_redirect r_out(std::cout, sout);
    py::scoped_ostream_redirect r_err(std::cerr, serr);

    // Parse the command line to generate parameters
    CommandLineHelper cl(cmd.c_str());
    auto param = lmshoot_parse_commandline(cl, false);
    PointSetShootingProblem<TPixel, VDim>::minimize(param);
  }

  void ExecuteApply(
    const string &cmd, py::object sout, py::object serr, const py::kwargs& kwargs)
  {
    // Redirect the outputs if needed
    py::scoped_ostream_redirect r_out(std::cout, sout);
    py::scoped_ostream_redirect r_err(std::cerr, serr);

           // Parse the command line to generate parameters
    CommandLineHelper cl(cmd.c_str());
    auto param = lmtowarp_parse_commandline(cl, false);
    PointSetGeodesicToWarp<TPixel, VDim>::run(param);
  }
};


template <typename TPixel, unsigned int VDim>
void instantiate_lmshoot(py::handle m, const char *name)
{
  using API = LMShootAPIWrapper<TPixel, VDim>;
  py::class_<API>(m, name, "Python API for the PICSL lmshoot tool")
    .def(py::init<>([]() {
      auto *c = new API();
      return c;
    }))
    .def("fit", &API::ExecuteFit,
         "Execute one or more commands using the lmshoot command line interface",
         py::arg("command"),
         py::arg("out") = py::module_::import("sys").attr("stdout"),
         py::arg("err") = py::module_::import("sys").attr("stdout"))
    .def("apply", &API::ExecuteApply,
         "Execute one or more commands using the lmshoot command line interface",
         py::arg("command"),
         py::arg("out") = py::module_::import("sys").attr("stdout"),
         py::arg("err") = py::module_::import("sys").attr("stdout"))
    ;
}

PYBIND11_MODULE(picsl_greedy, m) {
  instantiate_greedy<double, 2>(m, "Greedy2D");
  instantiate_greedy<double, 3>(m, "Greedy3D");
  instantiate_greedy<float, 2>(m, "GreedyFloat2D");
  instantiate_greedy<float, 3>(m, "GreedyFloat3D");

  instantiate_lmshoot<double, 2>(m, "LMShoot2D");
  instantiate_lmshoot<double, 3>(m, "LMShoot3D");
  instantiate_lmshoot<float, 2>(m, "LMShootFloat2D");
  instantiate_lmshoot<float, 3>(m, "LMShootFloat3D");

  instantiate_multichunk_greedy<2>(m, "MultiChunkGreedy2D");
  instantiate_multichunk_greedy<3>(m, "MultiChunkGreedy3D");
};
