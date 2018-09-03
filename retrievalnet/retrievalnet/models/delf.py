import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v1 as resnet
from .layers import delf_attention, image_normalization, dimensionality_reduction


class Delf(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, None], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'normalize_input': False,
            'use_attention': False,
            'attention_kernel': 1,
            'normalize_feature_map': True,
            'normalize_average': True,
            'dimensionality_reduction': None,
            'proj_regularizer': 0.,
    }

    @staticmethod
    def tower(image, mode, config):
        image = image_normalization(image)
        if image.shape[-1] == 1:
            image = tf.tile(image, [1, 1, 1, 3])

        with slim.arg_scope(resnet.resnet_arg_scope()):
            is_training = config['train_backbone'] and (mode == Mode.TRAIN)
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=is_training):
                _, encoder = resnet.resnet_v1_50(image,
                                                 is_training=is_training,
                                                 global_pool=False,
                                                 scope='resnet_v1_50')
        feature_map = encoder['resnet_v1_50/block3']

        if config['use_attention']:
            descriptor = delf_attention(feature_map, config, mode == Mode.TRAIN,
                                        resnet.resnet_arg_scope())
        else:
            descriptor = tf.reduce_max(feature_map, [1, 2])

        if config['dimensionality_reduction']:
            descriptor = dimensionality_reduction(descriptor, config)
        return descriptor

    def _model(self, inputs, mode, **config):
        # This model does not support training
        config['train_backbone'] = False
        config['train_attention'] = False
        return {'descriptor': self.tower(inputs['image'], mode, config)}

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
