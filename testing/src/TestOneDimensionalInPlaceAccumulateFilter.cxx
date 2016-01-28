/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "OneDimensionalInPlaceAccumulateFilter.h"
#include "itkTimeProbe.h"

int main(int argc, char *argv[])
{
  typedef itk::VectorImage<float, 3> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  ImageType::SizeType radius; radius.Fill(2);


  typedef OneDimensionalInPlaceAccumulateFilter<ImageType> AccumFilterType;

  itk::ImageSource<ImageType>::Pointer pipeTail;
  for(int dir = 0; dir < ImageType::ImageDimension; dir++)
    {
    AccumFilterType::Pointer accum = AccumFilterType::New();
    accum->SetInput(pipeTail.IsNull() ? reader->GetOutput() : pipeTail->GetOutput());
    accum->SetDimension(dir);
    accum->SetRadius(radius[dir]);
    pipeTail = accum;

    itk::TimeProbe tp;
    tp.Start();
    accum->Update();
    tp.Stop();

    printf("Direction %d elapsed ms: %6.2f\n", dir, 1000 * tp.GetTotal());
    }

  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[2]);
  writer->SetInput(pipeTail->GetOutput());
  writer->Update();
}
