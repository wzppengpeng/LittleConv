#ifndef REFLECTION_HPP_
#define REFLECTION_HPP_

#include <memory>
#include <string>
#include <functional>
#include <map>

/**
 *
 *
 *this class is a singleton class to registry class and users can get class just vy their names
 *the method to solve ths problem is that using the static virables will build first in gcc
 *the problem now is that this model cannot solve the template, but I think that not enssencial
 *another problem is that now it can only receive the class which has void paramter to init= =
 *
 *I think the best method to use it is define the variable in cpp file, if define in hpp cannot reach
 *the target to split factory
 *
 * How to registry: declare a static Reflection<Base>::Registry<Sub> member virable by the macro
 * and define it in the cpp file
 *
 * ReflectionRegister(Base, Sub)
 */

namespace wzp {

template<typename Base>
class Reflection {
public:
    template<typename Sub>
    struct Registry
    {
        /**
         * Use the static virable
         */
        Registry(const std::string& class_name) {
            Reflection<Base>::get().m_map.emplace(class_name, []{return new Sub();});
        }

    };

    /**
     * create a dynamic function of Base type point
     * @param  class_name the string of the registry name
     * @return            the Base type point
     */
    static Base* create(const std::string& class_name) {
        auto it = Reflection<Base>::get().m_map.find(class_name);
        if(it != Reflection<Base>::get().m_map.end()) {
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
    Reflection<Base>() {};

    Reflection<Base>(const Reflection<Base>&) = delete;

    Reflection<Base>(Reflection<Base>&&) = delete;

    static Reflection<Base>& get() {
        static Reflection<Base> m_instance;
        return m_instance;
    }

public:
    std::map<std::string, std::function<Base*()> > m_map;
};

}
//the define macro
#define ReflectionRegister(Base, Sub) wzp::Reflection<Base>::Registry<Sub>


#endif