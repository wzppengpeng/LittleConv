#ifndef SINGLETON_HPP
#define SINGLETON_HPP

#include <stdexcept>

namespace wzp {

//每一种类型会据现为一种类型，因此通过模版参数T来区分不同的的单例!!!!

template<typename T>
class Singleton
{
public:
    //init the instace
    template<typename... Args>
    static T* Instance(Args&&... args) {
        if(m_instance == nullptr)
            m_instance = new T(std::forward<Args>(args)...);
        return m_instance;
    }

    //when the m_insatce hase been initianized, use this interface to get the instance,
    //else it will throw a logic error
    static T* GetInstance() {
        if(m_instance == nullptr)
            throw std::logic_error("the instace has not been initianized first");
        return m_instance;
    }

    //the destructor of T
    static void DestroyInstance() {
        if(m_instance != nullptr) {
            delete m_instance;
            m_instance = nullptr;
        }
    }

private:
    //constructor
    Singleton(void);
    virtual ~Singleton(void);//let the object can only gen in heap
    Singleton(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator = (const Singleton&) = delete;

private:
    static T* m_instance;//the instace ptr of the singleton mode
};

template<typename T> T* Singleton<T>::m_instance = nullptr;//init the m_instance ptr

} // wzp

#endif // SINGLETON_HPP
