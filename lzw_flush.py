"""
LZW Compression with Flush Strategy

Dictionary Structure: Trie (Prefix Tree)
-Phrases represented as (parent_index, next_char) tuples
    -parent_index: pointer to parent phrase
    -next_char: byte value
-Hash Map Modification
    -nodes can have many children, so child pointers would take a lot of space
    -instead of child pointers, we use a hash map
    -map (parent_index, next_char) -> child_index
    -python dict used as hash map

Encoder Output Stream: 
-32-bit length header, followed by a sequence of pointers.
-The parent phrase is already in the dictionary and is being extended by next_char. 


Flush Strategy
-Once dictionary reaches max size, clear it and start from scratch
"""

from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataEncoder, DataDecoder
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter

class LZWEncoderDictionary:
    """
    Wrapper around Python dictionary
    Hash (parent_index, character) -> child_index

    Initialize Alphabet: Each char will have index = value
    We could initialize with self.dictionary = {(None, i): i for i in range(256)} but 
    this would be redundant since each empty parent pointer takes 12 bytes.   
    Instead, we treat the 256 chars implicitly. 
    We omit them from the dictionary, but start the dictionary indices at 256. 
    """

    def __init__(self, max_size=4096):
        self.dictionary = {}
        self.next_index = 256 # next available index
        self.max_size = max_size
    
    def search(self, parent_index, next_char):
        """
        Returns child_index if key is found, None otherwise
        Only works when parent_index > 0 (not a char)
        """
        return self.dictionary.get((parent_index, next_char))

    def insert(self, parent_index, next_char):
        """Adds {(parent_index, next_char): next_index} if dictionary not full"""
        if self.next_index == self.max_size:
            self.dictionary = {}
            self.next_index = 256

        self.dictionary[(parent_index, next_char)] = self.next_index
        self.next_index += 1


class LZWDecoderDictionary:
    """
    To reconstruct a phrase from a tuple we now work up the tree instead of down. 
    Technically this is achieved via the reversed hash map child_index -> (parent_index, character)
    However, since child_index will progress like 1,2,3,... we can just use an array of tuples
    
    Since we only need to access parent pointers, no hashing is needed. 
    
    The 256 chars are implicit in dictionary. We use an offset of 256.
    """

    def __init__(self, max_size=4096):
        """Initialize empty dictionary"""
        self.dictionary = []
        self.max_size = max_size
        self.next_index = 256

    def insert(self, parent_index, next_char):
        """Adds (parent_index, next_char) if dictionary not full"""
        if self.next_index == self.max_size:
            self.dictionary = []
            self.next_index = 256
            
        self.dictionary.append((parent_index, next_char))
        self.next_index += 1
    
    
    def unravel_phrase(self, index):
        """
        Work backwards to construct the phrase at given index, repeatedly adding the last character and then 
        checking the dictionary entry at parent index

        Return: bytes(phrase)
        """
        phrase = []
        current_index = index
        
        while current_index > 255:
            parent_index, char = self.dictionary[current_index-256] # 256 offset
            if char is not None:
                phrase.append(char)
            current_index = parent_index
            
        # Single char remaining, and index is value  
        phrase.append(current_index)
    
        phrase.reverse()
        return bytes(phrase)
        


class LZWEncoder:
    """
    Processes data using an LZWEncoderDictionary 
    """

    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.pointer_size = (max_size - 1).bit_length()
        self.dictionary = LZWEncoderDictionary(max_size)

    def encode(self, data: bytearray):
        """
        Produces stream of pointers while constructing dictionary

        Leftover: we only output a pointer when a new phrase is encountered. But it is possible (in fact likely) that the 
        last few bytes will match with an existing phrase, so that no outout is produced. We can include this partial phrase 
        by forming a tuple from the parent_index and placeholder char 0. The decoder, which knows the length, will be able to 
        identify the placeholder.
        
        """
        stream = []
        parent_index = data[0] # first char
        for char in data[1:]:
            child_index = self.dictionary.search(parent_index, char)
            # If end of phrase, add index to stream and add tuple to dictionary, and start new phrase at char
            if child_index is None: 
                stream.append(parent_index)
                self.dictionary.insert(parent_index, char)
                parent_index = char
            # Else, keep building phrase
            else: 
                parent_index = child_index
        
        # Handle leftover 
        if parent_index != 0:
            stream.append(parent_index)

        return stream
    
    def encode_binary(self, data):
        """
        Encodes data as a bitarray

        32 bit length header
        Pointers: 12 bit index
        """
        bit_stream = BitArray()
        bit_stream += uint_to_bitarray(len(data), 32)
        pointers = self.encode(data)
        for pointer in pointers:
            bit_stream += uint_to_bitarray(pointer, self.pointer_size)

        return bit_stream


class LZWDecoder:

    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.pointer_size = (max_size - 1).bit_length()
        self.dictionary = LZWDecoderDictionary(max_size)
    
    def decode(self, pointers, length):
        """
        Takes in stream of pointers, produces bytearray of characters 
        """
        output = bytearray()

        prev_pointer = pointers[0]
        prev_phrase = self.dictionary.unravel_phrase(prev_pointer)
        output.extend(prev_phrase)

        for pointer in pointers[1:]:
            # Handle KwKwK special case 
            if pointer == self.dictionary.next_index:
                phrase = prev_phrase + prev_phrase[:1] # concatenate prev phrase with its first symbol 
            else:
                phrase = self.dictionary.unravel_phrase(pointer)
            
            output.extend(phrase)
            self.dictionary.insert(prev_pointer, phrase[0])
            prev_pointer = pointer
            prev_phrase = phrase
    
            #output.extend(phrase[1:])

        output = output[:length] # clips the (potential) placeholder byte
        return output 


    def decode_binary(self, bit_stream):
        length = bitarray_to_uint(bit_stream[:32])
        pointers = []
        for i in range(32, len(bit_stream), self.pointer_size):
            pointer = bitarray_to_uint(bit_stream[i:i+self.pointer_size])
            pointers.append(pointer)

        return self.decode(pointers, length)
    

"""
Block Compatible Encoder/Decoder

Accepts a single data block as input. We have no need for multiple data blocks. 

Multiple blocks are necessary for very large data, but partitioning the data means it isn't 
pure LZW, since it cannot match phrases that contain block boundaries. 
"""

class LZWFlushEncoder(DataEncoder):
    def __init__(self, max_size=4096):
        self.max_size = max_size

    def reset(self):
        pass  # Dictionary is created fresh in encode_block

    def encode_block(self, data_block: DataBlock) -> BitArray:
        data = bytearray(data_block.data_list)
        encoder = LZWEncoder(self.max_size)
        return encoder.encode_binary(data)

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int = None):
        # Ignore block_size, read entire file at once
        with Uint8FileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                # Read entire file as one block
                data_list = []
                while True:
                    symbol = fds.get_symbol()
                    if symbol is None:
                        break
                    data_list.append(symbol)
                
                if data_list:
                    output = self.encode_block(DataBlock(data_list))
                    writer.write_block(output)


class LZWFlushDecoder(DataDecoder):
    def __init__(self, max_size=4096):
        self.max_size = max_size

    def reset(self):
        pass  # Dictionary is created fresh in decode_block

    def decode_block(self, bitarray: BitArray):
        decoder = LZWDecoder(self.max_size)
        decoded = decoder.decode_binary(bitarray)
        return DataBlock(list(decoded)), len(bitarray)

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)

