#ifndef TYPE_INFO_HPP_
#define TYPE_INFO_HPP_

#include <typeinfo>
#include <string>

namespace wzp{
    template<typename... Args> struct TypeConnect;
    
    //impl the TypeConnect
    template<typename First, typename... Rest>
    struct TypeConnect<First, Rest...>{
        static std::string getName(){
            return typeid(First).name() + std::string(" ") + TypeConnect<Rest...>::getName();
        }
    };
    
    template<typename Last>
    struct TypeConnect<Last>{
        static std::string getName(){
            return typeid(Last).name();
        }
    };
    
}//wzp

#endif /*TYPE_INFO_HPP_*/