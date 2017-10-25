#ifndef LICON_COMMON_HPP
#define LICON_COMMON_HPP

#include <ctime>

#include "log/log.hpp"

#include "function/help_function.hpp"

// the alias of float type
typedef float F;

/**
 * the macro of log
 */
#ifndef LOG_INFO
#define LOG_INFO(STR, ...) wzp::log::info(STR, ##__VA_ARGS__)
#endif

#ifndef LOG_FATAL
#define LOG_FATAL(STR, ...) wzp::log::fatal(STR, ##__VA_ARGS__)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(STR, ...) wzp::log::error(STR, ##__VA_ARGS__)
#endif

/**
 * the tuple get macro
 */
#ifndef GET_0
#define GET_0(x) std::get<0>(x)
#endif

#ifndef GET_1
#define GET_1(x) std::get<1>(x)
#endif

#ifndef GET_2
#define GET_2(x) std::get<2>(x)
#endif

#ifndef GET_3
#define GET_3(x) std::get<3>(x)
#endif


// the environment set up functions
namespace licon
{

// the phase of train or test
enum class Phase { TRAIN, TEST };

//set the environment, set the rand seed
inline static void EnvSetUp() {
    srand(time(NULL));
    wzp::log::log_init();
}

} //licon

#endif /*LICON_COMMON_HPP*/