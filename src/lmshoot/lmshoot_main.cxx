#include "PointSetGeodesicShooting.h"
#include "GreedyException.h"
#include <iostream>

int main(int argc, char *argv[])
{
  ShootingParameters param;
  if(argc < 2)
    return lmshoot_usage(true);

  try
  {
    // Parse the command line
    CommandLineHelper cl(argc, argv);
    param = lmshoot_parse_commandline(cl, true);

    // Run the main code
    if(param.use_float)
    {
      switch(param.dim)
      {
        case 2: return PointSetShootingProblem<float, 2>::minimize(param);
        case 3: return PointSetShootingProblem<float, 3>::minimize(param);
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
      }
    }
    else
    {
      switch(param.dim)
      {
        case 2: return PointSetShootingProblem<double, 2>::minimize(param);
        case 3: return PointSetShootingProblem<double, 3>::minimize(param);
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
