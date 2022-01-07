import tensorflow_hub as hub
# import tensorflow as tf
# layer = hub.load("/Users/dttai11/albert/trainingTest/albert_base_2")
# layer = "https://tfhub.dev/google/albert_base/2"
# layer = hub.load("https://tfhub.dev/google/albert_base/2")
# hub_layer = hub.KerasLayer("/Users/dttai11/albert/trainingTest/albert_base_2")
# model = tf.keras.Sequential([hub_layer])
# model.summary()

from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf

tags = set()
# if is_training:
#   tags.add("train")
# albert_module = hub.load("https://tfhub.dev/google/albert_base/1", tags=tags,
#                            )
#
# albert_module.summary()
input_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
input_mask = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
sequence_mask = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
# albert = hub.KerasLayer(
#     "https://tfhub.dev/google/albert_base/3",
#     trainable=True,
#     signature="tokens",
#     output_key="pooled_output",
# )
albert = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/albert_base/2",signature='tokens' , signature_outputs_as_dict=True, trainable=True)])
features = {
    "input_ids": input_ids,
    "input_mask": input_mask,
    "segment_ids": sequence_mask,
}

out = albert(features)
model = tf.keras.Model(inputs=[input_ids, input_mask, sequence_mask], outputs=out)
# model.compile("adam", loss="sparse_categorical_crossentropy")
# model.summary()
albert.summary()
# print(f"The Hub encoder has {len(albert.trainable_variables)} trainable variables")