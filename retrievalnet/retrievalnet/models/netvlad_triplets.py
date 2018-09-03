import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import resnet_v1 as resnet
from .layers import vlad, dimensionality_reduction, image_normalization, triplet_loss


class NetvladTriplets(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
            'p': {'shape': [None, None, None, 1], 'type': tf.float32},
            'n': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'triplet_margin': 0.5,
            'intermediate_proj': None,
            'n_clusters': 64,
            'dimensionality_reduction': None,
            'proj_regularizer': 0.,
            'train_backbone': False,
            'train_vlad': True,
            'loss_in': False,
            'loss_squared': True,
    }

    @staticmethod
    def tower(image, mode, config):
        image = image_normalization(image)
        if image.shape[-1] == 1:
            image = tf.tile(image, [1, 1, 1, 3])

        with slim.arg_scope(resnet.resnet_arg_scope()):
            training = config['train_backbone'] and (mode == Mode.TRAIN)
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=training):
                _, encoder = resnet.resnet_v1_50(image,
                                                 is_training=training,
                                                 global_pool=False,
                                                 scope='resnet_v1_50')
        feature_map = encoder['resnet_v1_50/block3']
        descriptor = vlad(feature_map, config, mode == Mode.TRAIN)
        if config['dimensionality_reduction']:
            descriptor = dimensionality_reduction(descriptor, config)
        return descriptor

    def _model(self, inputs, mode, **config):
        if mode == Mode.PRED:
            descriptor = self.tower(inputs['image'], mode, config)
            return {'descriptor': descriptor}
        else:
            descriptors = {}
            for e in ['image', 'p', 'n']:
                with tf.name_scope('triplet_{}'.format(e)):
                    descriptors['descriptor_'+e] = self.tower(inputs[e], mode, config)
            return descriptors

    def _loss(self, outputs, inputs, **config):
        loss, _, _ = triplet_loss(outputs, inputs, **config)
        return loss

    def _metrics(self, outputs, inputs, **config):
        loss, distance_p, distance_n = triplet_loss(outputs, inputs, **config)
        return {'loss': loss, 'distance_p': distance_p, 'distance_n': distance_n}
