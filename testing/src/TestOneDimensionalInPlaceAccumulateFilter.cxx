/*=========================================================================

  Program:   GreedyReg fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  GreedyReg initial development was funded by the NIH grant R01 EB017255.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

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
