#include "PointSetGeodesicToWarp.h"
#include "GreedyException.h"
#include <iostream>

int main(int argc, char *argv[])
{
  WarpGenerationParameters param;
  if(argc < 2)
    return lmtowarp_usage(true);

  try
  {
    // Parse the command line
    param = lmtowarp_parse_commandline(argc, argv, true);

    // Run the main code
    if(param.use_float)
    {
      switch(param.dim)
      {
        case 2: return PointSetGeodesicToWarp<float, 2>::run(param);
        case 3: return PointSetGeodesicToWarp<float, 3>::run(param);
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
      }
    }
    else
    {
      switch(param.dim)
      {
        case 2: return PointSetGeodesicToWarp<double, 2>::run(param);
        case 3: return PointSetGeodesicToWarp<double, 3>::run(param);
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
      }
    }
  }
  catch(std::exception &exc)
  {
    std::cerr << "ABORTING PROGRAM DUE TO RUNTIME EXCEPTION -- "
              << exc.what() << std::endl;
    return -1;
  }

}
