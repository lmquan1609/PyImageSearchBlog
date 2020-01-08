import torch
from torch import nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Arguments:
        block_args: nametupled
        global_params: namedtupled
    """
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self._batch_norm_epsilon = 1 - global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) \
                    and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        input_filters = self._block_args.input_filters # number of input channels
        output_filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(input_filters, output_filters, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(output_filters,
                                    eps=self._batch_norm_epsilon,
                                    momentum=self._batch_norm_momentum)

        # Depthwise separable convolution
        kernel_size = self._block_args.kernel_size
        stride = self._block_args.stride
        self._depthwise_conv = Conv2d(output_filters, output_filters, kernel_size,
                                    stride=stride,
                                    groups=1,
                                    bias=False)
        self._bn1 = nn.BatchNorm2d(output_filters,
                                    eps=self._batch_norm_epsilon,
                                    momentum=self._batch_norm_momentum)
        
        # Squeeze and Excitation layer if desired
        if self.has_se:
            num_squeezed_channels = max(1, 
                int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(output_filters, num_squeezed_channels, 1)
            self._se_expand = Conv2d(num_squeezed_channels, output_filters, 1)

        # Output phase
        final_output_filters = self._block_args.output_filters
        self._project_conv = Conv2d(output_filters, final_output_filters, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(final_output_filters,
                                eps=self._batch_norm_epsilon,
                                momentum=self._batch_norm_momentum)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        else:
            x = inputs
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = x * torch.sigmoid(x)
        
        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            
            x = x + inputs
        
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with `.from_name` or `.from_pretrained` methods

    Arguments:
        block_args: List of Block Arguments to construct blocks
        global_params: namedtuple, A set of GlobalParams shared between blocks
    """

    def __init__(self, block_args=None, global_params=None):
        super().__init__()
        assert isinstance(block_args, list), 'block_args should be a list'
        assert len(block_args) > 0, 'length of block_args must be greater than 0'
        self._global_params = global_params
        self._block_args = block_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        batch_norm_momentum = 1 - self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._block_args.batch_norm_epsilon

        # Stem
        in_channels = 3 # RGB
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, 3, stride=2, bias=False)
        # !: self._conv_stem = Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels,
                                eps=batch_norm_epsilon,
                                momentum=batch_norm_momentum)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeat(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                                        input_filters=block_args.output_filters,
                                        stride=1)
            
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, 1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels,
                                eps=self._batch_norm_epsilon,
                                momentum=self._batch_norm_momentum)
        
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """Returns output of the final convolutional layer"""
        # Stem
        x = self._swish(self.bn0(self._conv_stem(inputs)))
        
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """Calls extract_features to extract features, applies final linear layer and return logits"""
        batch_size = inputs.shape[0]
        # Convolutional layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.flatten(x, start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x
    
    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        block_args, global_params = get_model_params(model_name, override_params)
        return cls(block_args, global_params)
    
    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, 3, stride=2, bias=False)
        return model
    
    @staticmethod
    def _check_model_name_is_valid(model_name, need_pretrained_weights=False):
        """Validate model name. Currently, Only EfficientNet-B0, 1, 2, 3 available"""
        num_models = 4 if need_pretrained_weights else 8
        valid_models = [f'efficientnet-b{i}' for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError(f"Model_name should be one of: {', '.join(valid_models)}")