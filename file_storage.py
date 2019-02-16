import os
import zlib

class FileStorage(object):

    def __init__(self, file_name_prefix, readonly=False):
        self.file_name = file_name_prefix
        self.readonly = readonly
        self.dict_file = None
        self.data_file = None
        self.dict = {}
        self.open()


    def open(self, file_name_prefix=''):
        if file_name_prefix != '':
            self.file_name = file_name_prefix
        if self.file_name == '':
            raise ValueError('file name is empty')
        self.dict = {}
        if self.readonly:
            self.data_file = open(self.file_name + '.data', 'rb')
            self.dict_file = open(self.file_name + '.dict', 'r')
        else:
            self.data_file = open(self.file_name + '.data', 'a+b')
            self.dict_file = open(self.file_name + '.dict', 'a+')
        with open(self.file_name + '.dict', 'r') as temp_dict:
            for line in temp_dict:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                self.dict[parts[0]] = (int(parts[1]), int(parts[2]))
    
    
    def check_open(self):
        if self.data_file is None or self.dict_file is None:
            self.open()
    
    
    def count(self):
        self.check_open()
        
        return len(self.dict)


    def read_by_offset(self, offset, length):
        self.check_open()
        
        self.data_file.seek(offset)
        compressed_data = self.data_file.read(length)
        decompressed_data = zlib.decompress(compressed_data)
        return decompressed_data.decode('utf8')


    def read(self, key):
        self.check_open()
        
        if key not in self.dict:
            return None
        offset, length = self.dict[key]
        return self.read_by_offset(offset, length)


    def write(self, key, value):
        self.check_open()
        
        if key not in self.dict:
            self.data_file.seek(0, os.SEEK_END)
            offset_begin = self.data_file.tell()
            compressed_data = zlib.compress(value.encode('utf8'))
            self.data_file.write(compressed_data)
            offset_end = self.data_file.tell()
            self.dict_file.write(key + '\t' + str(offset_begin) + '\t' + str(offset_end - offset_begin) + '\n')
            self.dict[key] = (offset_begin, offset_end - offset_begin)
            self.data_file.flush()
            self.dict_file.flush()
        return self.dict[key]

    
    def keys(self):
        self.check_open()
        
        for k, _ in self.dict.items():
            yield k


    def items(self):
        self.check_open()
        
        for k, (o, l) in self.dict.items():
            yield (k, self.read_by_offset(o, l))


    def close(self):
        if self.data_file is not None:
            self.data_file.close()
        if self.dict_file is not None:
            self.dict_file.close()
        self.data_file = None
        self.dict_file = None
        self.dict = {}
