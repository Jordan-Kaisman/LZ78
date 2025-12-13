"""
LZW Compression with LRU Prune Strategy

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


LRU Prune Strategy
-Once dictionary reaches max size, remove lru leaf to make room for each insert
"""

from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataEncoder, DataDecoder
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter


class LZWDictionaryBase:
    """
    Base class for LZW Encoder and Decoder Dictionaries with LRU Pruning
    Implements DLL and Auxiliary Arrays for pruning
    
    Codes 0-255 are implicit single-byte entries (never in DLL, never pruned)
    """

    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.next_index = 256  # start after implicit alphabet
        self.free_index = None
        self.full = False

        # DLL pointer arrays
        self.head = -1
        self.tail = -1
        self.prev = [-1] * max_size
        self.next = [-1] * max_size

        # Auxiliary arrays
        self.parent = [-1] * max_size
        self.char = [0] * max_size
        self.child_count = [0] * max_size

    def dll_insert_head(self, index):
        """Insert leaf at head"""
        self.prev[index] = -1
        self.next[index] = self.head
        if self.head != -1:
            self.prev[self.head] = index
        self.head = index
        if self.tail == -1:
            self.tail = index

    def dll_insert_tail(self, index):
        """Insert leaf at tail"""
        self.next[index] = -1
        self.prev[index] = self.tail
        if self.tail != -1:
            self.next[self.tail] = index
        self.tail = index
        if self.head == -1:
            self.head = index

    def dll_remove(self, index):
        """Remove leaf at arbitrary position"""
        prev_index = self.prev[index]
        next_index = self.next[index]

        if self.head == index:
            self.head = next_index
        else:
            self.next[prev_index] = next_index

        if self.tail == index:
            self.tail = prev_index
        else:
            self.prev[next_index] = prev_index

        self.prev[index] = -1
        self.next[index] = -1


class LZWEncoderDictionary(LZWDictionaryBase):
    """
    LZW Encoder Dictionary with LRU Pruning
    
    Hash (parent_index, character) -> child_index
    Codes 0-255 are implicit, dictionary only stores codes >= 256
    """

    def __init__(self, max_size=4096):
        super().__init__(max_size)
        self.dictionary = {}

    def search(self, parent_index, next_char):
        """Returns child_index if key is found, None otherwise"""
        return self.dictionary.get((parent_index, next_char))

    def prune(self):
        target_index = self.tail
        parent_index = self.parent[target_index]
        char = self.char[target_index]

        # Remove from DLL
        self.dll_remove(target_index)

        # Remove from dictionary
        del self.dictionary[(parent_index, char)]
        self.free_index = target_index
        self.parent[target_index] = -1
        self.char[target_index] = 0
        self.child_count[target_index] = 0

        # Update Parent: decrement child count, if childless and not implicit root, add as leaf
        self.child_count[parent_index] -= 1
        if self.child_count[parent_index] == 0 and parent_index >= 256:
            self.dll_insert_tail(parent_index)

    def insert(self, parent_index, next_char):
        """Prune if full, then add (parent_index, next_char) to dictionary"""
        if self.full:
            self.prune()
            idx = self.free_index
        else:
            idx = self.next_index
            self.next_index += 1
            if self.next_index == self.max_size:
                self.full = True

        self.dictionary[(parent_index, next_char)] = idx
        self.parent[idx] = parent_index
        self.char[idx] = next_char
        self.child_count[parent_index] += 1

        # If parent was a leaf in DLL, remove it (no longer a leaf)
        if self.child_count[parent_index] == 1 and parent_index >= 256:
            self.dll_remove(parent_index)

        self.dll_insert_head(idx)


class LZWDecoderDictionary(LZWDictionaryBase):
    """
    LZW Decoder Dictionary with LRU Pruning
    
    Uses parent/char arrays to reconstruct phrases by walking up the tree
    Codes 0-255 are implicit single-byte entries
    """

    def get_following_index(self):
        if self.full:
            return self.tail
        return self.next_index

    def prune(self):
        target_index = self.tail
        parent_index = self.parent[target_index]

        # Remove from DLL
        self.dll_remove(target_index)

        # Clear slot
        self.free_index = target_index
        self.parent[target_index] = -1
        self.char[target_index] = 0
        self.child_count[target_index] = 0

        # Update Parent: decrement child count, if childless and not implicit root, add as leaf
        self.child_count[parent_index] -= 1
        if self.child_count[parent_index] == 0 and parent_index >= 256:
            self.dll_insert_tail(parent_index)

    def insert(self, parent_index, next_char):
        """Prune if full, then add entry"""
        if self.full:
            self.prune()
            idx = self.free_index
        else:
            idx = self.next_index
            self.next_index += 1
            if self.next_index == self.max_size:
                self.full = True

        self.parent[idx] = parent_index
        self.char[idx] = next_char
        self.child_count[parent_index] += 1

        # If parent was a leaf in DLL, remove it (no longer a leaf)
        if self.child_count[parent_index] == 1 and parent_index >= 256:
            self.dll_remove(parent_index)

        self.dll_insert_head(idx)

    def unravel_phrase(self, index):
        """
        Work backwards to construct the phrase at given index
        Handles implicit single-byte codes 0-255
        """
        if index < 256:
            return bytes([index])

        phrase = []
        current_index = index

        while current_index > 255:
            phrase.append(self.char[current_index])
            current_index = self.parent[current_index]

        # Single char remaining, and index is value
        phrase.append(current_index)  # final single byte (0-255)

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
        Produces stream of code indices while constructing dictionary
        
        LZW differences from LZ78:
        - Start with first byte as initial phrase (implicit in dictionary)
        - Emit only the code index (no char)
        - Reset to breaking char, not root
        """
        stream = []
        parent_index = data[0]  # first byte is implicit code

        for char in data[1:]:
            child_index = self.dictionary.search(parent_index, char)
            if child_index is None:
                # End of phrase: emit code, add new entry, restart from char
                stream.append(parent_index)
                self.dictionary.insert(parent_index, char)
                parent_index = char
            else:
                # Keep building phrase
                parent_index = child_index

        # Emit final phrase
        stream.append(parent_index)

        return stream

    def encode_binary(self, data):
        """
        Encodes data as a bitarray
        
        32 bit length header
        Codes: pointer_size bit indices
        """
        bit_stream = BitArray()
        bit_stream += uint_to_bitarray(len(data), 32)
        pointers = self.encode(data)
        for pointer in pointers:
            bit_stream += uint_to_bitarray(pointer, self.pointer_size)

        return bit_stream


class LZWDecoder:
    """
    Processes stream of code indices using LZWDecoderDictionary
    """

    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.pointer_size = (max_size - 1).bit_length()
        self.dictionary = LZWDecoderDictionary(max_size)


    def decode(self, pointers, length):
        """
        Takes in stream of code indices, produces bytearray of characters 
        """
        output = bytearray()

        # Handle first pointer
        prev_pointer = pointers[0]
        prev_phrase = self.dictionary.unravel_phrase(prev_pointer)
        output.extend(prev_phrase)

        for pointer in pointers[1:]:
            # KwKwK special case: need to anticipate next index to be added
            if pointer == self.dictionary.get_following_index():
                phrase = prev_phrase + prev_phrase[:1]
            else:
                phrase = self.dictionary.unravel_phrase(pointer)

            output.extend(phrase)

            # insert previous phrase extended by first char of current phrase
            self.dictionary.insert(prev_pointer, phrase[0])

            prev_pointer = pointer
            prev_phrase = phrase

        return output

    def decode_binary(self, bit_stream):
        """
        Decodes bitarray into length header and stream of code indices
        """
        length = bitarray_to_uint(bit_stream[:32])
        pointers = []
        for i in range(32, len(bit_stream), self.pointer_size):
            pointer = bitarray_to_uint(bit_stream[i:i + self.pointer_size])
            pointers.append(pointer)

        return self.decode(pointers, length)


class LZWPruneEncoder(DataEncoder):
    def __init__(self, max_size=4096):
        self.max_size = max_size

    def reset(self):
        pass

    def encode_block(self, data_block: DataBlock) -> BitArray:
        data = bytearray(data_block.data_list)
        encoder = LZWEncoder(self.max_size)
        return encoder.encode_binary(data)

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int = None):
        with Uint8FileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                data_list = []
                while True:
                    symbol = fds.get_symbol()
                    if symbol is None:
                        break
                    data_list.append(symbol)

                if data_list:
                    output = self.encode_block(DataBlock(data_list))
                    writer.write_block(output)


class LZWPruneDecoder(DataDecoder):
    def __init__(self, max_size=4096):
        self.max_size = max_size

    def reset(self):
        pass

    def decode_block(self, bitarray: BitArray):
        decoder = LZWDecoder(self.max_size)
        decoded = decoder.decode_binary(bitarray)
        return DataBlock(list(decoded)), len(bitarray)

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)



