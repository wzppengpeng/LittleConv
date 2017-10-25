#ifndef SMART_SINGLETON_HPP
#define SMART_SINGLETON_HPP


#include <stdexcept>
#include <memory>

namespace wzp {

template<typename T>
class SmartSingleton {
public:
    template<typename... Args>
    static std::shared_ptr<T> Instance(Args&&... args) {
        if(m_instance.get() == nullptr) {
            m_instance = std::make_shared<T>(std::forward<Args>(args)...);
        }
        return m_instance;
    }

    static std::shared_ptr<T> GetInstance() {
        if(m_instance.get() == nullptr)
            throw std::logic_error("the instace has not been initianized first");
        return m_instance;
    }

private:
    static std::shared_ptr<T> m_instance;

private:
    SmartSingleton(void);
    virtual ~SmartSingleton() {}
    SmartSingleton(SmartSingleton&&) = delete;
    SmartSingleton(const SmartSingleton&) = delete;
    SmartSingleton& operator= (const SmartSingleton&) = delete;
    SmartSingleton& operator= (SmartSingleton&&) = delete;
};

template<typename T>
std::shared_ptr<T> SmartSingleton<T>::m_instance(nullptr);

} // wzp


#endif // SMART_SINGLETON_HPP
