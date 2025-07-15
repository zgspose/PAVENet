# Copyright (c) Hikvision Research Institute. All rights reserved.
from .builder import (build_attention, build_positional_encoding,
                      build_transformer_layer_sequence, build_transformer,
                      build_text_encoder, ATTENTION, POSITIONAL_ENCODING,
                      TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER, TEXT_ENCODER)
from .positional_encoding import RelSinePositionalEncoding
from .text_encoder import PseudoTextEncoder, CLIPTextEncoder
from .transformer import (SOITTransformer, PETRTransformer, VideoPoseTransformerV1, # 新添加  --- 添加时间 2024-8-02
                          PETRTransformerV1,  # 新添加  --- 添加时间 2024-8-14
                          PetrTransformerDecoder, VideoPoseTransformerDecoderV1,
                          MultiScaleDeformablePoseAttention)

__all__ = [
    'build_attention', 'build_positional_encoding',
    'build_transformer_layer_sequence', 'build_transformer',
    'build_text_encoder', 'ATTENTION', 'POSITIONAL_ENCODING',
    'TRANSFORMER_LAYER_SEQUENCE', 'TRANSFORMER', 'TEXT_ENCODER',
    'RelSinePositionalEncoding', 'PseudoTextEncoder', 'CLIPTextEncoder',
    'SOITTransformer', 'PETRTransformer', 'PetrTransformerDecoder',
    'VideoPoseTransformerV1', 'VideoPoseTransformerDecoderV1',
    'MultiScaleDeformablePoseAttention', 'PETRTransformerV1'
]
