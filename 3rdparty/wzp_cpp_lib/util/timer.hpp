#ifndef MY_TIMER_HPP_
#define MY_TIMER_HPP_
#include <chrono>

using namespace std;
using namespace std::chrono;

namespace wzp{


class Timer{
public:
    Timer() : m_begin(high_resolution_clock::now()) {}
    //reset
    void reset(){
        m_begin = high_resolution_clock::now();
    }

    //ms
    int64_t elapsed() const{
        return duration_cast<chrono::milliseconds>(high_resolution_clock::now() - m_begin).count();
    }

    //us
    int64_t elapsed_micro() const{
        return duration_cast<chrono::microseconds>(high_resolution_clock::now() - m_begin).count();
    }

    //ns
    int64_t elapsed_nano() const{
        return duration_cast<chrono::nanoseconds>(high_resolution_clock::now() - m_begin).count();
    }

    //s
    int64_t elapsed_seconds() const{
        return duration_cast<chrono::seconds>(high_resolution_clock::now() - m_begin).count();
    }

    //minutes
    int64_t elapsed_minutes() const{
        return duration_cast<chrono::minutes>(high_resolution_clock::now() - m_begin).count();
    }

    //hour
    int64_t elapsed_hours() const{
        return duration_cast<chrono::hours>(high_resolution_clock::now() - m_begin).count();
    }
private:
    time_point<high_resolution_clock> m_begin;//the clock to get the time begin
};

}//wzp

#endif /*MY_TIMER_HPP_*/