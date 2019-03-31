#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>

const std::size_t PARTS_NUM = 4;

const std::uint64_t maskes[] = {
    18446462598732840960ULL,
    281470681743360ULL,
    4294901760ULL,
    65535ULL
};

const std::uint8_t shifts[] = {
    48,
    32,
    16,
    0
};

std::uint64_t number_of_bits(std::uint64_t i) {
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

std::uint8_t calculate_distance(std::uint64_t hash_1, std::uint64_t hash_2) {
    return number_of_bits(hash_1 ^ hash_2);
}

std::uint64_t get_part(std::uint64_t hash, std::uint8_t part_index) {
    return (hash & maskes[part_index]) >> shifts[part_index];
}

void save_hash(std::uint64_t hash, std::map<std::uint16_t, std::set<std::uint64_t>>* hashes_by_parts) {
    for (int i = 0; i < PARTS_NUM; ++i) {
        hashes_by_parts[i][get_part(hash, i)].insert(hash);
    }
}

int main() {
    std::ifstream input_file_stream("simhash_sorted.txt");
    std::ofstream log_file_stream("log.txt");
    std::ofstream output("output_final.txt");

    std::string simhash_string;
    std::map<std::uint16_t, std::set<std::uint64_t>> hashes_by_parts[4];
    std::set<std::uint64_t> hashes;
    std::size_t counter = 0;
    while (std::getline(input_file_stream, simhash_string)) {
        ++counter;
        if (counter % 10000 == 0) {
            log_file_stream << "Reading string " << counter << std::endl;
        }
        std::uint64_t hash = std::stoul(simhash_string);
        save_hash(hash, hashes_by_parts);
        hashes.insert(hash);
    }

    log_file_stream << "Reading data finished" << std::endl;

    counter = 0;
    while (hashes.size() > 0) {
        ++counter;
        if (counter % 100 == 0) {
            log_file_stream << "Hashes to process: " << hashes.size() << std::endl;
        }
        std::uint64_t hash = *(hashes.begin());
        int group_size = 1;
        std::set<std::uint64_t> group;
		group.insert(hash);

        for (int i = 0; i < PARTS_NUM; ++i) {
            std::uint16_t part = get_part(hash, i);
            for (std::uint64_t dublicate_candidate : hashes_by_parts[i][part]) {
                int distance = calculate_distance(dublicate_candidate, hash);
                if (distance < 4) {
					group.insert(dublicate_candidate);
                }
            }
        }
        for (std::uint64_t hash_to_delete : group) {
            for (int i = 0; i < PARTS_NUM; ++i) {
                std::uint16_t part = get_part(hash_to_delete, i);
                hashes_by_parts[i][part].erase(hash_to_delete);
            }
            hashes.erase(hash_to_delete);
        }
        output << group.size() << std::endl;
    }

    return 0;
}
