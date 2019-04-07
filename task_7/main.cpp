#include <iostream>
#include <stack>
#include <vector>
#include <vector>
#include <climits>


const std::size_t HANDLING_BITS = 3;
const std::size_t BITS_IN_FIRST_BYTE = CHAR_BIT - HANDLING_BITS;


template<typename T>
struct BitsHandler {
public:
    T data;
    std::size_t ending_zeros_number;
    std::size_t full_bytes_number;
    std::size_t first_byte_bits_number;
    std::size_t last_byte_bits_number;
    std::size_t bytes_number;

    BitsHandler(T in_data) :
            data(in_data),
            ending_zeros_number(0),
            full_bytes_number(0),
            first_byte_bits_number(0),
            last_byte_bits_number(0),
            bytes_number(0)
    {
        while (in_data % 2 == 0) {
            in_data >>= 1;
            ++ending_zeros_number;
        }

        std::size_t significant_digits_number = 0;

        while (in_data > 0) {
            in_data >>= 1;
            ++significant_digits_number;
        }

        bytes_number = 1;
        if (significant_digits_number <= BITS_IN_FIRST_BYTE) {
            first_byte_bits_number = significant_digits_number;
        } else {
            first_byte_bits_number = BITS_IN_FIRST_BYTE;
            significant_digits_number -= BITS_IN_FIRST_BYTE;
            last_byte_bits_number = significant_digits_number % CHAR_BIT;
            full_bytes_number = significant_digits_number / CHAR_BIT;
            bytes_number += full_bytes_number;
            if (last_byte_bits_number > 0) {
                bytes_number += 1;
            }
        }
    }

    char get_first_byte() {
        unsigned char bytes_number_bits = (bytes_number - 1) << BITS_IN_FIRST_BYTE;
        unsigned char first_byte = data >> (ending_zeros_number + last_byte_bits_number + CHAR_BIT * full_bytes_number);
        std::cout << "bytes number: " << bytes_number << std::endl;
        std::cout << "bytes number bits : " << (int)bytes_number_bits << std::endl;
        std::cout << "first byte: " << (int)first_byte << std::endl;
        std::cout << "sum: " << (int)(bytes_number_bits | first_byte) << std::endl;
        return bytes_number_bits | first_byte;
    }

    char get_full_byte(size_t index) {
        assert(full_bytes_number > index);
        return (data >> (ending_zeros_number + last_byte_bits_number + CHAR_BIT * (full_bytes_number - index))) & UCHAR_MAX;
    }

    char get_last_byte() {
        assert(last_byte_bits_number > 0);
        return (data >> ending_zeros_number) & (UCHAR_MAX >> (CHAR_BIT - last_byte_bits_number));
    }

    void add_bytes(std::vector<char>& bytes) {
        bytes.push_back(get_first_byte());
        for (size_t index = 0; index < full_bytes_number; ++index) {
            bytes.push_back(get_full_byte(index));
        }
        if (last_byte_bits_number > 0) {
            bytes.push_back(get_last_byte());
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
    for (size_t i = 1; i < bytes_number - 1; ++i) {
        std::cout << "object before: " << object << std::endl;
        object <<= CHAR_BIT;
        object += data[i];
        std::cout << "object after: " << object << std::endl;
    }

    std::size_t significant_digits_num = get_significant_digits_num(data[bytes_number - 1]);
    std::cout << "significant_digits_num: " << significant_digits_num << std::endl;
    std::cout << "object before: " << object << std::endl;
    object <<= significant_digits_num;
    object += data[bytes_number - 1];
    std::cout << "object after: " << object << std::endl;

    std::cout << "zero bytes: " << sizeof(T) - bytes_number - 1 << std::endl;

    for (int i = 0; i < sizeof(T) - bytes_number - 1; ++i) {
        object <<= CHAR_BIT;
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
    std::uint64_t number = 251658240;
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
