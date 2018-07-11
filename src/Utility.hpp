#ifndef UTILITY_HPP
#define UTILITY_HPP
#include <iostream>
#include "IOstreamColor.h"
#include <chrono>

namespace Utility {
/** \brief scoped time calculation using std::chrono*/
struct ScopedChronoTime
{
public:

    ScopedChronoTime(std::string moduleName):strModuleName(moduleName)
    {
        t1 = std::chrono::steady_clock::now();
    }
    ~ScopedChronoTime()
    {
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double time= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        std::cout<<YELLOW<<"<<" << strModuleName << ">> module time cost "
                <<time * 1000<<" ms"<<NOCOLOR<<std::endl;
    }
private:
    std::string strModuleName;
    std::chrono::steady_clock::time_point t1;
};

}
#endif // UTILITY_HPP
