#ifndef EVENT_HPP
#define EVENT_HPP

#include <map>

using ll = long long;

namespace wzp {
//the Observer mode, delegate by Events
//use += as connect, give a Func into the map
//use -= key, erase the Func out of the map
//use notify, to run all Func which save in the map
//Observer: other objects push a function here, when some values changed, call all objects functions
//but here is more powerful, just like the C# delegate, with the same input data, call all delegate functions
template<typename Func>
class Events {
public:
    Events() = default;
    ~Events() {}

    ll operator += (Func&& fun) {
        return connect(std::forward<Func>(fun));
    }

    ll operator += (Func& fun) {
        return connect(fun);
    }

    Events& operator -= (ll key) {
        dis_connect(key);
        return *this;
    }

    //the notify function
    template<typename... Args>
    void operator() (Args&&... args) {
        notify(std::forward<Args>(args)...);
    }

    void clear() {
        m_connections.clear();
        m_next_key = 0;
    }

private:
    //the push function, push a F into the Func map
    template<typename F>
    void push(ll index, F&& fun) {
        m_connections.emplace(index, std::move(fun));
    }

    template<typename F>
    void push(ll index, F& fun) {
        m_connections.emplace(index, fun);
    }

    template<typename F>
    ll assign(F&& f) {
        auto index = m_next_key++;
        push(index, std::forward<F>(f));
        return index;
    }

    //the connect function, the connect can only receive the default Func type
    ll connect(Func&& fun) {
        return assign(std::forward<Func>(fun));
    }

    ll connect(Func& fun) {
        return assign(fun);
    }

    void dis_connect(ll key) {
        m_connections.erase(key);
    }

    template<typename... Args>
    void notify(Args&&... args) {
        for(auto& p : m_connections) {
            p.second(std::forward<Args>(args)...);
        }
    }

private:
    //let this module noncopy
    Events(const Events&) = delete;
    Events(Events&&) = delete;

private:

    ll m_next_key = 0;
    std::map<ll, Func> m_connections;
};

} // wzp

#endif // EVENT_HPP
