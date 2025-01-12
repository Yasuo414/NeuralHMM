#ifndef GARBAGE_COLLECTOR_H
#define GARBAGE_COLLECTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

class GarbageCollector {
public:
    template<typename T>
    static void erase(T& object);
};

#endif