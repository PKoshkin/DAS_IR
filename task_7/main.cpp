#include <iostream>
#include <stack>
#include <vector>
#include <vector>
#include <climits>
#include <bitset>


const std::size_t HANDLING_BITS = 3;


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
void add_object(T data, std::vector<char>& bytes) {
    std::size_t bytes_number = get_bytes_number(data);
    char* data_pointer = (char*)&data;
    for (std::size_t i = 0; i < bytes_number; ++i) {
        bytes.push_back(data_pointer[i]);
    }
}


template<typename T>
std::vector<char> encode(const T* data, std::size_t count) {
    std::vector<char> bytes;
    for (std::size_t i = 0; i < count; ++i) {
        add_object(data[i], bytes);
    }
    return bytes;
}


template<typename T>
std::size_t read_object(const char* data, std::vector<T>& objects) {
    std::size_t bytes_number = (data[0] >> (CHAR_BIT - HANDLING_BITS)) + 1;
    return 0;
}

template<typename T>
std::vector<T> decode(const char* data, std::size_t length) {
    std::vector<T> objects;
    std::size_t bytes_index = 0;
    while (bytes_index < length) {
        bytes_index += read_object(data + bytes_index, objects);
    }
    return objects;
}

int main () {

    std::uint64_t number;
    std::cin >> number;

    char* char_p = (char*)&number;
    for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
        std::cout << std::bitset<8>(char_p[i]) << std::endl;
    }
    std::cout << "----------------------" << std::endl;





    std::cout << "number: " << number << std::endl;
    std::vector<char> bytes = encode(&number, 1);
    for (int i = 0; i < bytes.size(); ++i) {
        std::cout << "char " << i << ": " << (int)bytes[i] << std::endl;
    }
    std::cout << "bytes.size(): " << bytes.size() << std::endl;
    std::vector<std::uint64_t> number_decode = decode<std::uint64_t>(bytes.data(), bytes.size());
    std::cout << "number_decode.size(): " << number_decode.size() << std::endl;
    for (int i = 0; i < number_decode.size(); ++i) {
        std::cout << "number_decode[" << i << "]: " << number_decode[i] << std::endl;
    }

    return 0;
}
