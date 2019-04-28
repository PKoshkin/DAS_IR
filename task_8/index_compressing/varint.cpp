#include <iostream>
#include <vector>
#include <climits>
#include <ctime>
#include <random>


const std::size_t HANDLING_BITS = 3;
const std::size_t BITS_IN_FIRST_BYTE = CHAR_BIT - HANDLING_BITS;
typedef unsigned char Byte;


template<typename T>
std::size_t get_bytes_number(T data) {
    std::size_t significant_bits_number = 0;
    while (data > 0) {
        data >>= 1;
        ++significant_bits_number;
    }
    std::size_t bytes_to_encode_number = 1;
    while ((8 * bytes_to_encode_number - 3) < significant_bits_number) {
        ++bytes_to_encode_number;
    }
    return bytes_to_encode_number;
}


template<typename T>
void add_object(T data, std::vector<Byte>& bytes) {
    std::size_t bytes_number = get_bytes_number(data);
    Byte* data_pointer = (Byte*)&data;

    Byte bytes_number_bits = ((bytes_number - 1) << BITS_IN_FIRST_BYTE);
    bytes.push_back(bytes_number_bits | data_pointer[bytes_number - 1]);
    for (int i = bytes_number - 2; i >= 0; --i) {
        bytes.push_back(data_pointer[i]);
    }
}


template<typename T>
std::vector<Byte> encode(const T* data, std::size_t count) {
    std::vector<Byte> bytes;
    for (std::size_t i = 0; i < count; ++i) {
        add_object(data[i], bytes);
    }
    /*
    for (int i = 0; i < bytes.size(); ++i) {
        std::cout << "byte[" << i << "]=" << (int)bytes[i] << std::endl;
    }
    */
    return bytes;
}


template<typename T>
std::size_t read_object(const Byte* data, std::vector<T>& objects) {
    std::size_t bytes_number = (data[0] >> BITS_IN_FIRST_BYTE) + 1;

    T object = data[0] & (UCHAR_MAX >> HANDLING_BITS);
    for (size_t i = 1; i < bytes_number; ++i) {
        object <<= CHAR_BIT;
        object += data[i];
    }
    objects.push_back(object);
    return bytes_number;
}

template<typename T>
std::vector<T> decode(const Byte* data, std::size_t length) {
    std::vector<T> objects;
    std::size_t bytes_index = 0;
    while (bytes_index < length) {
        bytes_index += read_object(data + bytes_index, objects);
    }
    return objects;
}
