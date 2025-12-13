"""
LZ78 Compression with Flush Strategy

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
-32-bit length header, followed by tuples (parent_index, next_char). 
-The parent phrase is already in the dictionary and is being extended by next_char. 


Flush Strategy
-Once dictionary reaches max size, clear it and start from scratch
"""

from dataclasses import dataclass
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataEncoder, DataDecoder
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter

@dataclass
class LZ78Tuple:
    """
    parent_index: pointer to parent phrase (root is 0)
    next_char: byte value
    """
    parent_index: int 
    next_char: int

class LZ78EncoderDictionary:
    """
    Wrapper around Python dictionary
    Hash (parent_index, character) -> child_index
    Empty string at index 0 (root) is implicit, not actually stored
    """

    def __init__(self, max_size=4096):
        """
        dictionary: map (parent_index, character) -> child_index
        next_index: next available index, incremented on each insert
        """
        self.dictionary = {}
        self.next_index = 1 
        self.max_size = max_size
    
    def search(self, parent_index, next_char):
        """Returns child_index if key is found, None otherwise"""
        return self.dictionary.get((parent_index, next_char))

    def insert(self, parent_index, next_char):
        """Reset if full, then add {(parent_index, next_char): next_index}"""
        if self.next_index == self.max_size:
            self.dictionary = {}
            self.next_index = 1
            
        self.dictionary[(parent_index, next_char)] = self.next_index
        self.next_index += 1


class LZ78DecoderDictionary:
    """
    To reconstruct a phrase from a tuple we now work up the tree instead of down. 
    Since we only ever need to access parent pointers, no hashing is needed. 

    Technically we use the reversed map child_index -> (parent_index, character).
    However, since child_index will progress like 1,2,3,... we can just use an array of tuples.
    """

    def __init__(self, max_size=4096):
        """
        dictionary: array of (parent_index, character) tuples
        Initialize dictionary with the empty tuple at index 0
        """
        self.dictionary = [(None,None)]
        self.max_size = max_size

    def insert(self, parent_index, next_char):
        """Reset if full, then add (parent_index, next_char)"""
        if len(self.dictionary) == self.max_size:
            self.dictionary = [(None,None)]

        self.dictionary.append((parent_index, next_char))
    
    def unravel_phrase(self, index):
        """
        Work backwards to construct the phrase at given index, repeatedly adding the last character and then 
        checking the dictionary entry at parent index
        """
        phrase = []
        current_index = index
        
        while current_index != 0:
            parent_index, char = self.dictionary[current_index]
            if char is not None:
                phrase.append(char)
            current_index = parent_index
        
        phrase.reverse()
        return bytes(phrase)


class LZ78Encoder:
    """
    Processes data using an LZ78EncoderDictionary 
    """

    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.pointer_size = (max_size - 1).bit_length()
        self.dictionary = LZ78EncoderDictionary(max_size)

    def encode(self, data: bytearray):
        """
        Produces stream of LZ78Tuples while constructing dictionary

        Leftover Handling 
        We only ouput a tuple when a new phrase is encountered. But it is possible (in fact likely) that the 
        last few bytes will match with an existing phrase, so that no outout is produced. We can include this 
        partial phrase by forming a tuple from the parent_index and placeholder char 0. The decoder, which 
        knows the messaage length, will truncate accordingly.
        """
        stream = []
        parent_index = 0
        for char in data:
            child_index = self.dictionary.search(parent_index, char)
            # If end of phrase, add tuple to stream and dictionary
            if child_index is None: 
                stream.append(LZ78Tuple(parent_index, char))
                self.dictionary.insert(parent_index, char)
                parent_index = 0
            # Else, keep building phrase
            else: 
                parent_index = child_index
        
        # Handle leftover 
        if parent_index != 0:
            stream.append(LZ78Tuple(parent_index, 0))

        return stream
    
    def encode_binary(self, data):
        """
        Encodes data as a bitarray

        32 bit length header
        Tuples: (pointer_size bit index, 8 bit char) 
        """
        bit_stream = BitArray()
        bit_stream += uint_to_bitarray(len(data), 32)
        tuples = self.encode(data)
        for tuple in tuples:
            bit_stream += uint_to_bitarray(tuple.parent_index, self.pointer_size)
            bit_stream += uint_to_bitarray(tuple.next_char, 8)

        return bit_stream


class LZ78Decoder:
    """
    Processes stream of LZ78Tuples using LZ78DecoderDictionary
    """

    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.pointer_size = (max_size - 1).bit_length()
        self.dictionary = LZ78DecoderDictionary(max_size)
    
    def decode(self, tuples, length):
        """
        Takes in stream of tuples, produces bytearray of characters 
        """
        output = bytearray()

        for tuple in tuples:
            # unravel parent phrase, add that and next char to the output stream
            phrase = self.dictionary.unravel_phrase(tuple.parent_index)
            output.extend(phrase)
            output.append(tuple.next_char)

            self.dictionary.insert(tuple.parent_index, tuple.next_char) # add phrase to dictionary

        output = output[:length] # clips the (potential) placeholder byte
        return output 


    def decode_binary(self, bit_stream):
        """
        Decodes bitarray into length header and stream of tuples
        """
        length = bitarray_to_uint(bit_stream[:32])
        tuples = []
        tuple_width = self.pointer_size + 8
        for i in range(32, len(bit_stream), tuple_width):
            index = bitarray_to_uint(bit_stream[i:i+self.pointer_size])
            char = bitarray_to_uint(bit_stream[i+self.pointer_size:i+tuple_width])
            tuples.append(LZ78Tuple(index, char))

        return self.decode(tuples, length)
    

"""
DataBlock Compatible Encoder/Decoder

Accepts a single data block as input. We have no need for multiple data blocks. 

Multiple blocks are necessary for very large data, but partitioning the data means it isn't 
pure LZ78, since it cannot match phrases that contain block boundaries. 
"""

class LZ78FlushEncoder(DataEncoder):
    def __init__(self, max_size=4096):
        self.max_size = max_size

    def reset(self):
        pass  

    def encode_block(self, data_block: DataBlock) -> BitArray:
        data = bytearray(data_block.data_list)
        encoder = LZ78Encoder(self.max_size)
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


class LZ78FlushDecoder(DataDecoder):
    def __init__(self, max_size=4096):
        self.max_size = max_size

    def reset(self):
        pass  

    def decode_block(self, bitarray: BitArray):
        decoder = LZ78Decoder(self.max_size)
        decoded = decoder.decode_binary(bitarray)
        return DataBlock(list(decoded)), len(bitarray)

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)


    
