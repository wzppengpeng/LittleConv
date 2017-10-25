#ifndef REFLECTION_HPP_
#define REFLECTION_HPP_

#include <memory>
#include <string>
#include <functional>
#include <map>

#include "function/apply_tuple.hpp"

/**
 *
 *
 *this class is a singleton class to registry class and users can get class just vy their names
 *the method to solve ths problem is that using the static virables will build first in gcc
 *the problem now is that this model cannot solve the template, but I think that not enssencial
 *another problem is that now it can  receive the class which has any paramter
 *
 *I think the best method to use it is define the variable in cpp file, if define in hpp cannot reach
 *the target to split factory
 *
 * How to registry: declare a static Factory<Base>::Registry<Sub> member virable by the macro
 * and define it in the cpp file
 *
 * FactoryRegister(Base, Sub)
 */

namespace wzp {

template<typename Base, typename... Args>
class Factory {
public:
    template<typename Sub>
    struct Registry
    {
        /**
         * Use the static virable
         */
        Registry(const std::string& class_name) {
            Factory<Base, Args...>::get().m_map.emplace(class_name,
                [] (const Args&... args) {return new Sub(args...);});
        }

        Registry(std::string&& class_name) {
            Factory<Base, Args...>::get().m_map.emplace(class_name,
                [] (const Args&... args) {return new Sub(args...);});
        }

    };

    /**
     * create a dynamic function of Base type point
     * @param  class_name the string of the registry name
     * @return            the Base type point
     */
    static Base* create(const std::string& class_name, const std::tuple<Args...>& t) {
        auto it = Factory<Base, Args...>::get().m_map.find(class_name);
        if(it != Factory<Base, Args...>::get().m_map.end()) {
            return wzp::apply(it->second, t);
        }
        else {
            return nullptr;
        }
    }

    static std::unique_ptr<Base> create_unique(const std::string& class_name,
        const std::tuple<Args...>& t) {
        return std::unique_ptr<Base>(create(class_name, t));
    }

    static std::shared_ptr<Base> create_shared(const std::string& class_name,
        const std::tuple<Args...>& t) {
        return std::shared_ptr<Base>(create(class_name, t));
    }

    // the no args create type
    static Base* create(const std::string& class_name) {
        auto it = Factory<Base, Args...>::get().m_map.find(class_name);
        if(it != Factory<Base, Args...>::get().m_map.end()) {
            return (it->second)();
        }
        else {
            return nullptr;
        }
    }

    static std::unique_ptr<Base> create_unique(const std::string& class_name) {
        return std::unique_ptr<Base>(create(class_name));
    }

    static std::shared_ptr<Base> create_shared(const std::string& class_name) {
        return std::shared_ptr<Base>(create(class_name));
    }

private:
    Factory<Base, Args...>() {};

    Factory<Base, Args...>(const Factory<Base, Args...>&) = delete;

    Factory<Base, Args...>(Factory<Base, Args...>&&) = delete;

    static Factory<Base, Args...>& get() {
        static Factory<Base, Args...> m_instance;
        return m_instance;
    }

public:
    std::map<std::string, std::function<Base*(const Args&...)> > m_map;
};

}
//the define macro
#ifndef FactoryRegister
#define FactoryRegister(Base, Sub, ...) wzp::Factory<Base, ##__VA_ARGS__>::Registry<Sub>
#endif


#endif