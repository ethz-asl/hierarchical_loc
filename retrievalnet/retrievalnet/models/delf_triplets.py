import tensorflow as tf

from .base_model import BaseModel, Mode
from .delf import Delf
from .layers import triplet_loss


class DelfTriplets(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
            'p': {'shape': [None, None, None, 1], 'type': tf.float32},
            'n': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'use_attention': True,
            'attention_kernel': 1,
            'normalize_average': True,
            'normalize_feature_map': True,
            'triplet_margin': 0.5,
            'dimensionality_reduction': None,
            'proj_regularizer': 0.,
            'train_backbone': False,
            'train_attention': True,
            'loss_in': False,
            'loss_squared': True,
    }

    def _model(self, inputs, mode, **config):
        if mode == Mode.PRED:
            descriptor = Delf.tower(inputs['image'], mode, config)
            return {'descriptor': descriptor}
        else:
            descriptors = {}
            for e in ['image', 'p', 'n']:
                with tf.name_scope('triplet_{}'.format(e)):
                    descriptors['descriptor_'+e] = Delf.tower(inputs[e], mode, config)
            return descriptors

    def _loss(self, outputs, inputs, **config):
        loss, _, _ = triplet_loss(outputs, inputs, **config)
        return loss

    def _metrics(self, outputs, inputs, **config):
        loss, distance_p, distance_n = triplet_loss(outputs, inputs, **config)
        return {'loss': loss, 'distance_p': distance_p, 'distance_n': distance_n}
