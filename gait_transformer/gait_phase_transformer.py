import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


# core transformer components

def mlp(x, hidden_units, dropout_rate, activation=tf.nn.gelu):
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class LayerScale(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
    def build(self, input_shape):
        # lazy build with the projection dimension
        self.scale = self.add_weight(shape=(1, 1, input_shape[-1]), initializer=tf.keras.initializers.Constant(0.1), dtype=self.dtype)
    def call(self, input, **kwargs):
        return input * self.scale


class FeedForward(keras.Sequential):

    def __init__(self, hidden_units, dropout_rate=0.0, activation=tf.nn.gelu, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build(self, input_shape):
        projection_dim = input_shape[-1]

        self.add(layers.Dense(self.hidden_units, activation=self.activation))
        if self.dropout_rate > 0.0:
            self.add(layers.Dropout(self.dropout_rate))

        self.add(layers.Dense(projection_dim))
        if self.dropout_rate > 0.0:
            self.add(layers.Dropout(self.dropout_rate))

        super(FeedForward, self).build(input_shape)


class EncoderTransformerLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim_scale=1.0, dropout_rate=0.0, survival_prob=1.0, layer_scale=False, att=None, activation=tf.nn.gelu):
        """ Encoder transformer block similar to DETR implementation

            Params:
                embed_dim: dimension to tokens
                num_heads: number of heads for MHSA
                ff_dim: list of dimensions for hidden layers in FFN
                att (optional): MultiHeadAttention layer, can be passed to share weights
        """
        super(EncoderTransformerLayer, self).__init__()
        self.att = att or layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = FeedForward(int(embed_dim * ff_dim_scale), dropout_rate=dropout_rate, activation=activation)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        if layer_scale:
            self.layerscale1 = LayerScale()
            self.layerscale2 = LayerScale()
        else:
            self.layerscale1 = self.layerscale2 = layers.Lambda(lambda x : x)

        if survival_prob < 1.0:
            import tensorflow_addons as tfa

            self.stochastic_depth1 = tfa.layers.StochasticDepth(survival_prob)
            self.stochastic_depth2 = tfa.layers.StochasticDepth(survival_prob)
        else:
            self.stochastic_depth1 = self.stochastic_depth2 = layers.Add()

    def call(self, value, positional_encoding=None, training=None):

        if positional_encoding is None:
            query = key = value
        else:
            query = key = value + positional_encoding

        attn_output = self.att(query, key=key, value=value, training=training)
        attn_output = self.layerscale1(attn_output, training=training)
        out1 = self.layernorm1(self.stochastic_depth1([value,  attn_output], training=training))

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.layerscale2(ffn_output, training=training)
        return self.layernorm2(self.stochastic_depth2([out1, ffn_output], training=training))


def get_pos_encoding(positions, depth, min_rate=1.0/10000.0, dtype=None):
    positions = tf.cast(positions, dtype=dtype)
    angle_rate_exponents = tf.cast(tf.linspace(0,1,depth//2), dtype=dtype)
    angle_rates = min_rate**(angle_rate_exponents)
    angle_rads = tf.expand_dims(positions, 1)*tf.expand_dims(angle_rates, axis=0)
    sines = tf.math.sin(angle_rads)
    cosines = tf.math.cos(angle_rads)
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding


def get_pos_encoding_matrix(num_positions, depth, min_rate=1.0/10000.0, dtype=tf.float32):
    positions = tf.range(num_positions)
    return get_pos_encoding(positions, depth, min_rate, dtype)

# loosely based on https://keras.io/examples/vision/image_classification_with_vision_transformer/
# with some guidance from https://github.com/Visual-Behavior/detr-tensorflow/blob/main/detr_tf/networks/transformer.py


class PositionalEncoding(layers.Layer):
    def __init__(self, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
    def call(self, input, training=None):
        T = tf.shape(input)[1]


def get_gait_phase_transformer(
    dataset,
    projection_dim = 256,
    num_heads = 4,
    ffn_units_scale = 2,
    transformer_layers = 5,
    dropout_rate = 0.0,
    output_dropout_rate = 0.0,
    survival_prob = 1.0,
    layer_scale = False,
    shared = True,
    repeat_positional = True,
    mlp_head_units = [256, 256]  # Size of the dense layers of the final classifier
):

    input_shape = dataset.element_spec[0].shape
    T = input_shape[1]  # sequence length
    num_joints = input_shape[2]
    joint_dim = input_shape[3]
    num_outputs = dataset.element_spec[1].shape[2]

    # replace sequence length with none to allow inference on difference lengths
    inputs = layers.Input(shape=(None, *input_shape[2:]))

    # concatenate joints into one dimension
    flat_inputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], num_joints * joint_dim])
    encoded_poses = layers.Dense(units=projection_dim, name='embedding')(flat_inputs)

    positional_encoding = tf.expand_dims(get_pos_encoding_matrix(tf.shape(inputs)[1], projection_dim, dtype=encoded_poses.dtype), axis=0)

    # to reuse the parameters for attention itself
    shared_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate) if shared else None

    # stack some of these transformers
    for i in range(transformer_layers):
        enc = EncoderTransformerLayer(projection_dim, num_heads, ffn_units_scale, survival_prob=survival_prob, dropout_rate=dropout_rate, layer_scale=layer_scale, att=shared_att)
        encoded_poses = enc(encoded_poses, positional_encoding if (i == 0 or repeat_positional) else None)

    features = mlp(encoded_poses, hidden_units=mlp_head_units, dropout_rate=output_dropout_rate)
    output = layers.Dense(num_outputs)(features)
    output = tf.expand_dims(output, axis=-1)

    return keras.Model(inputs=inputs, outputs=output, name='gait_phase_transformer')


def get_gait_phase_stride_transformer(
    dataset = None,
    M = 17,
    kp_dim = 17,
    kp_idx_keep=None,
    projection_dim = 256,
    num_heads = 6,
    ffn_units_scale = 2,
    transformer_layers = 5,
    dropout_rate = 0.1,
    output_dropout_rate = 0.0,
    survival_prob = 1.0,
    layer_scale = True,
    shared = False,
    repeat_positional = False,
    keypoint_mlp=None,
    activation=tf.nn.gelu,
    derotate=False,
    center=False,
    physics_consistency_loss=0.0,
    foot_vel_loss=False,
    mlp_head_units = [256, 256],  # Size of the dense layers of the final classifier
):

    if dataset is None:
        dataset_sig = ((tf.TensorSpec(shape=[None, None, kp_dim, 3], dtype=tf.float32), tf.TensorSpec(shape=(None), dtype=tf.float32)),
                        tf.TensorSpec(shape=[None, None, M, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, None, M, 1], dtype=tf.float32))
    else:
        dataset_sig = dataset.element_spec

    input_shape = dataset_sig[0][0].shape
    T = input_shape[1]  # sequence length
    num_joints = input_shape[2]
    joint_dim = input_shape[3]
    num_outputs = dataset_sig[1].shape[2]

    kp_inputs = layers.Input(shape=(None, *input_shape[2:]))
    height_inputs = layers.Input(shape=())

    x = kp_inputs

    if derotate:
        print('Derotate')
        #right_hip_idx = joint_names.index('Right hip')
        #left_hip_idx = joint_names.index('Left hip')
        #thorax_idx = joint_names.index('Thorax')
        right_hip_idx, left_hip_idx, thorax_idx = 1, 4, 8

        mid_hip = (kp_inputs[:, :, right_hip_idx, :] + kp_inputs[:, :, left_hip_idx, :]) / 2

        v1 = kp_inputs[:, :, right_hip_idx, :] - kp_inputs[:, :, left_hip_idx, :]
        v2 = kp_inputs[:, :, thorax_idx, :] - mid_hip

        v1 = v1 + tf.reshape(tf.constant([1e-9, 0, 0]), [1, 1, 3])
        v2 = v2 + tf.reshape(tf.constant([0, 1e-9, 0]), [1, 1, 3])

        v1, _ = tf.linalg.normalize(v1, axis=-1)
        v2, _ = tf.linalg.normalize(v2, axis=-1)

        v3 = -tf.linalg.cross(v1, v2)   # compute the forward axis
        v3, _ = tf.linalg.normalize(v3, axis=-1)
        v2o = tf.linalg.cross(v1, v3)  # recompute the properly orthogonal torso direction

        R = tf.stack([v1, v3, v2o], axis=-1)

        x = tf.linalg.matmul(x, R)
    elif center:
        print('Center')
        # expects the data to be 2D keypoints so joint names order is different
        kp = x

        left_hip_idx, right_hip_idx = 11, 12
        mid_hip = (kp[:, :, right_hip_idx, :] + kp[:, :, left_hip_idx, :]) / 2

        x = x - mid_hip[:, :, None, :]
    else:
        print('No derotate')

    if kp_idx_keep is not None:
        kp = x
        num_joints = len(kp_idx_keep)
        kp_idx_keep = tf.convert_to_tensor(kp_idx_keep)
        x = layers.Lambda(lambda x: tf.experimental.numpy.take(x, kp_idx_keep, 2))(kp)

    if keypoint_mlp is not None:
        x = mlp(x, hidden_units=mlp_head_units + [3], dropout_rate=0.0, activation=activation)

    # concatenate joints into one dimension
    flat_inputs = tf.reshape(x, [tf.shape(kp_inputs)[0], tf.shape(x)[1], num_joints * joint_dim])
    encoded_poses = layers.Dense(units=projection_dim, name='embedding')(flat_inputs)

    positional_encoding = tf.expand_dims(get_pos_encoding_matrix(tf.shape(kp_inputs)[1], projection_dim, dtype=encoded_poses.dtype), axis=0)

    flat_demographics = tf.reshape(height_inputs, [tf.shape(height_inputs)[0], 1, 1])
    encoded_demographics = layers.Dense(units=projection_dim, name='demographics_embedding')(flat_demographics)
    demographics_encoding = tf.reshape(layers.Embedding(1, projection_dim)(tf.range(10)), [1, 10, projection_dim])

    positional_encoding = tf.concat([demographics_encoding, positional_encoding], axis=1)
    encoded_features = tf.concat([tf.tile(encoded_demographics, [1, 10, 1]), encoded_poses], axis=1)

    # to reuse the parameters for attention itself
    shared_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate) if shared else None

    # stack some of these transformers
    for i in range(transformer_layers):
        enc = EncoderTransformerLayer(projection_dim, num_heads, ffn_units_scale,
                                      survival_prob=survival_prob, dropout_rate=dropout_rate,
                                      layer_scale=layer_scale, att=shared_att, activation=activation)
        encoded_features = enc(encoded_features, positional_encoding if (i == 0 or repeat_positional) else None)

    features = mlp(encoded_features, hidden_units=mlp_head_units, dropout_rate=output_dropout_rate, activation=activation)
    output_layer = layers.Dense(num_outputs)
    output = output_layer(features)
    output = tf.expand_dims(output, axis=-1)

    output = output[:, 10:, :, :]

    if physics_consistency_loss > 0:

        class LossLayer(layers.Layer):

            def call(self, x):

                dt = 1/30.0
                discrete_velocity = (x[:, 1:, 8:10, :] - x[:, :-1, 8:10, :]) / dt
                adjusted_velocity = x[:, :-1, 11:13, :] - x[:, :-1, 10:11, :]
                velocity_error = tf.reduce_sum((discrete_velocity - adjusted_velocity)**2) * physics_consistency_loss

                self.add_loss(velocity_error)

                gait_cos = x[:, :, :4, 0]
                gait_sin = x[:, :, 4:8, 0]
                phase = tf.math.atan2(gait_sin, gait_cos)

                left_down = phase[:, :, 0] < phase[:, :, 2]
                right_down = phase[:, :, 1] < phase[:, :, 3]

                left_down_vel = tf.boolean_mask(x[:, :, 11, 0], left_down)
                right_down_vel = tf.boolean_mask(x[:, :, 12, 0], right_down)

                if foot_vel_loss:
                    self.add_loss(tf.reduce_sum(left_down_vel ** 2) * physics_consistency_loss)
                    self.add_loss(tf.reduce_sum(right_down_vel ** 2) * physics_consistency_loss)

                return x

        output = LossLayer()(output)

    model = keras.Model(inputs=[kp_inputs, height_inputs], outputs=output, name='gait_phase_transformer')

    return model


def shift_generator(keypoints3d, stride=1, L=90):
    N = keypoints3d.shape[0]
    length_idx = np.arange(L)
    samples = np.arange(0, N-L+1, stride)
    for s in samples:
        yield keypoints3d[length_idx+s]


def chunk_generator(keypoints3d, stride=1, L=90, batch_size=32):
    shift_sample_iter = shift_generator(keypoints3d, stride=stride, L=L)

    while True:
        try:
            chunk = []
            for i in range(batch_size):
                sample = next(shift_sample_iter)
                chunk.append(sample)
            yield np.array(chunk)
        except StopIteration as s:
            if len(chunk) > 0:
                yield np.array(chunk)
            break


def gait_phase_stride_inference(keypoints3d, height, regressor, L, batch_size=128):

    from tqdm import tqdm

    # compute rolling phases
    if keypoints3d.shape[0] >= L:

        O = (L-1) // 2

        chunk_iter = chunk_generator(keypoints3d, L=L, batch_size=batch_size)
        results = []
        for chunk in tqdm(chunk_iter):
            chunk_height = height / 1000.0 * np.ones((chunk.shape[0],))
            pred = regressor((chunk, chunk_height), training=False)[..., 0].numpy()
            results.append(pred)

        results = np.concatenate(results, axis=0)
        phases = np.concatenate([results[0, :O], results[:, O], results[-1, O+1:]], axis=0)

    else:
        phases = regressor((keypoints3d[None, ...], height[None, ...] / 1000.0),
                           training=False)[0, ..., 0].numpy()

    stride = phases[:, 8:]
    phases = phases[:, :8]

    return phases, stride


def load_default_model():

    # default model is in the assets directory below this file
    import os

    model_file = os.path.join(os.path.dirname(__file__), "assets", "model_v0.2.h5")

    model_params = {
        "transformer_layers": 6,
        "ffn_units_scale": 2,
        "projection_dim": 128,
        "layer_scale": False,
        "survival_prob": 1.0,
        "dropout_rate": 0.1,
        "num_heads": 3,
        "repeat_positional": True,
        "shared": True,
        "output_dropout_rate": 0,
        "derotate": False,
        "physics_consistency_loss": 0.001,
        "foot_vel_loss": False,
        "kp_idx_keep": [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],
        "mlp_head_units": [256, 256],
        "M": 19,
        "kp_dim": 17,
    }

    transformer_model = get_gait_phase_stride_transformer(**model_params)

    transformer_model.load_weights(model_file)

    return transformer_model
