#include <iostream>
#include <stack>
#include <vector>
#include <vector>
#include <climits>
#include <bitset>


const std::size_t HANDLING_BITS = 3;
const std::size_t BITS_IN_FIRST_BYTE = CHAR_BIT - HANDLING_BITS;


template<typename T>
struct BitsHandler {
public:
    T data;
    std::size_t significant_digits_number;
    std::size_t first_byte_bits_number;
    std::size_t bytes_number;

    BitsHandler(T in_data) : data(in_data), significant_digits_number(0), first_byte_bits_number(0), bytes_number(0) {
        while (in_data > 0) {
            in_data >>= 1;
            ++significant_digits_number;
        }
        if (significant_digits_number % CHAR_BIT > 0) {
            first_byte_bits_number = significant_digits_number % CHAR_BIT;
            bytes_number = significant_digits_number / CHAR_BIT + 1;
        } else {
            first_byte_bits_number = CHAR_BIT;
            bytes_number = significant_digits_number / CHAR_BIT;
        }

        if (first_byte_bits_number > BITS_IN_FIRST_BYTE) {
            first_byte_bits_number %= BITS_IN_FIRST_BYTE;
            ++bytes_number;
        }
        std::cout << "bytes number: " << bytes_number << std::endl;
        std::cout << "first_byte_bits_number: " << first_byte_bits_number << std::endl;
        std::cout << "significant_digits_number: " << significant_digits_number << std::endl;
    }

    char get_byte(size_t index) {
        if (index == 0) {
            unsigned char bytes_number_bits = ((bytes_number - 1) << BITS_IN_FIRST_BYTE);
            std::cout << "bytes_number_bits: " << (int)bytes_number_bits << std::endl;
            return  bytes_number_bits | (data >> (significant_digits_number - first_byte_bits_number));
        } else if (index == 1) {
            return UCHAR_MAX & (data >> (significant_digits_number - first_byte_bits_number - (index - 1) * CHAR_BIT));
        } else {
            return UCHAR_MAX & (data >> (significant_digits_number - first_byte_bits_number - (index - 1) * CHAR_BIT));
        }
    }

    void add_bytes(std::vector<char>& bytes) {
        for (size_t index = 0; index < bytes_number; ++index) {
            bytes.push_back(get_byte(index));
        }
    }
};


template<typename T>
void add_object(T data, std::vector<char>& bytes) {
    BitsHandler bites_handler(data);
    bites_handler.add_bytes(bytes);
}


template<typename T>
std::vector<char> encode(const T* data, std::size_t count) {
    std::vector<char> bytes;
    for (std::size_t i = 0; i < count; ++i) {
        add_object(data[i], bytes);
    }
    return bytes;
}


std::size_t get_significant_digits_num(char byte) {
    std::size_t counter = 0;

    std::cout << "get_significant_digits_num byte: " << (int)byte << std::endl;

    while (byte > 0) {
        byte >>= 1;
        ++counter;
    }
    return counter;
}


template<typename T>
std::size_t read_object(const char* data, std::vector<T>& objects) {
    std::size_t bytes_number = (data[0] >> BITS_IN_FIRST_BYTE) + 1;
    T object = 0;

    //std::cout << "bytes number in read_object: " << bytes_number << std::endl;

    object += data[0] & (UCHAR_MAX >> HANDLING_BITS);
    for (size_t i = 1; i < bytes_number; ++i) {
        std::cout << "object before: " << object << std::endl;
        object <<= CHAR_BIT;
        object += data[i];
        std::cout << "object after: " << object << std::endl;
    }

    objects.push_back(object);
    return bytes_number;
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
    std::uint64_t num = 511;
    char* char_p = (char*)&num;
    for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
        std::cout << std::bitset<8>(char_p[i]) << std::endl;
    }
    std::cout << "----------------------" << std::endl;


    std::uint64_t number;
    std::cin >> number;
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
