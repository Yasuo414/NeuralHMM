#include <iostream>
#include "garbage_collector/GarbageCollector.h"

template<typename T>
void GarbageCollector::erase(T& object) {
    if constexpr (std::is_same_v<T, std::string>) {
        object.clear();
        std::cout << "Object of type String was deleted." << std::endl;
    } else if constexpr (std::is_same_v<T, std::vector<typename T::value_type>>) {
        object.clear();
        std::cout << "The Vector object has been deleted." << std::endl;
    } else {
        std::cerr << "Unknown type, cannot be deleted." << std::endl;
    }
}