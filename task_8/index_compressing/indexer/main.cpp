#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <inttypes.h>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <exception>
#include <stdexcept>

#include "../varint.h"

int main(void) {
    std::unordered_map<std::string, std::vector<uint32_t>> index;
    std::unordered_map<uint32_t, std::string> docid_to_url;

    uint32_t docid = 0;
    std::string url;
    while (std::cin >> url) {
        size_t word_count = 0;
        std::cin >> word_count;

        ++docid;
        docid_to_url[docid] = url;

        for (size_t i = 0; i < word_count; ++i) {
            std::string word;
            std::cin >> word;

            index[word].push_back(docid);
        }
    }

    for (auto& p : index) {
        std::vector<uint32_t>& posting = p.second;

        std::sort(posting.begin(), posting.end());
        posting.erase(std::unique(posting.begin(), posting.end()), posting.end());
    }

    std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> word_to_offset_and_length;

    {
        std::ofstream output("docid_to_url");
        for (const auto& p : docid_to_url) {
            const uint32_t docid = p.first;
            const std::string& url = p.second;

            output << docid << '\t' << url << '\n';
        }
    }
    {
        std::ofstream output("index", std::ios::binary);
        for (const auto& p : index) {
            const std::string& word = p.first;
            const std::vector<uint32_t>& posting = p.second;
            std::vector<uint32_t> delta_encoded_posting;
            delta_encoded_posting.push_back(posting[0]);
            for (int i = 1; i < posting.size(); ++i) {
                delta_encoded_posting.push_back(posting[i] - posting[i - 1]);
            }
            std::vector<Byte> varint_encoded_posting = encode(delta_encoded_posting.data(), delta_encoded_posting.size());

            const uint32_t current_position = output.tellp();

            output.write(
                reinterpret_cast<char*>(varint_encoded_posting.data()),
                varint_encoded_posting.size() * sizeof(Byte)
            );

            const uint32_t next_position = output.tellp();
            const uint32_t length = next_position - current_position;

            word_to_offset_and_length[word] = std::make_pair(current_position, length);
        }
    }
    {
        std::ofstream output("word_to_offset_and_length");
        for (const auto& p : word_to_offset_and_length) {
            const std::string& word = p.first;
            const std::pair<uint32_t, uint32_t>& offset_and_length = p.second;

            output << word << '\t' <<
                      offset_and_length.first << '\t' <<
                      offset_and_length.second << '\n';
        }
    }

    return 0;
}
