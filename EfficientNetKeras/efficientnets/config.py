import re

class BlockArgs:
    def __init__(self, input_filters=None,
                output_filters=None,
                kernel_size=None,
                strides=None,
                num_repeat=None,
                se_ratio=None,
                expand_ratio=None,
                identity_skip=True):
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip

    # !review after having implementation
    def decode_block_string(self, block_string):
        '''Get a block arguments through a string notation of arguments'''
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            extract = re.split(r'(\d.*)', op)
            if len(extract) >= 2:
                key, value = extract[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        self.input_filters = int(options['i'])
        self.output_filters = int(options['o'])
        self.kernel_size = int(options['k'])
        self.num_repeat = int(options['r'])
        self.identity_skip = 'noskip' not in block_string
        self.se_ratio = float(options['se']) if 'se' in options else None
        self.expand_ratio = int(options['e'])
        self.strides = (int(options['s'][0]), int(options['s'][1]))

        return self

    def encode_block_string(self, block):
        '''
        Encode a block to a string
        Encoding schema: "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
            with X: number from 0 to 9, {} encapsulates optional args
        To deserialize an encoded block string, use the class method :
        ```python
            BlockArgs.from_block_string(block_string)
        ```
        '''
        args = [
            f'r{block.num_repeat}',
            f'k{block.kernel_size}',
            f's{block.strides[0], block.strides[1]}',
            f'e{block.expand_ratio}',
            f'i{block.input_filters}',
            f'o{block.output_filters}'
        ]

        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append(f'se{block.se_ratio}')

        if not block.identity_skip:
            args.append('noskip')

        return '_'.join(args)

    @classmethod
    def from_block_string(cls, block_string):
        block = cls()
        return block.decode_block_string(block_string)

# Default list of blocks for EfficientNets
def get_default_block_list():
    DEFAULT_BLOCK_LIST = [
        BlockArgs(32, 16, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1),
        BlockArgs(16, 24, kernel_size=3, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
        BlockArgs(24, 40, kernel_size=5, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6)
    ]
    return DEFAULT_BLOCK_LIST