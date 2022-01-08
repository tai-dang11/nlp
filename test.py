import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import create_optimizer
from transformers import TFAutoModel
#
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
# # tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
# # model = TFAutoModelForSequenceClassification.from_pretrained('albert-base-v2')
#
#
# imdb = load_dataset("imdb")
#
# def preprocess_function(examples):
#     return tokenizer(examples["text"], truncation=True)
#
# tokenized_imdb = imdb.map(preprocess_function, batched=True)
#
# # from transformers.data.data_collator import tf_default_data_collator
# # data_collator = tf_default_data_collator
# data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
# batch_size = 16
#
# tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
#     columns=['attention_mask', 'input_ids', 'label'],
#     shuffle=True,
#     batch_size=batch_size,
#     collate_fn=data_collator,
# )
#
# tf_validation_dataset = tokenized_imdb["train"].to_tf_dataset(
#     columns=['attention_mask', 'input_ids', 'label'],
#     shuffle=False,
#     batch_size=batch_size,
#     collate_fn=data_collator,
# )
#
# num_epochs = 5
# batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
# total_train_steps = int(batches_per_epoch * num_epochs)
# optimizer, schedule = create_optimizer(
#     init_lr=2e-5,
#     num_warmup_steps=0,
#     num_train_steps=total_train_steps
# )
#
# model.compile(optimizer=optimizer)
# # tf.keras.utils.plot_model(model, show_shapes=True, dpi=48)
# # model.summary()
# # model.fit(
# #     tf_train_dataset,
# #     # validation_data=tf_validation_dataset,
# #     epochs=1,
# # )
#
# # model.get_config()
# # model.save_pretrained("my_model")
# # model.push_to_hub('huggingface.co/SS8/test2')
# tokenizer.push_to_hub('huggingface.co/SS8/test2')
# # model.save
# # WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b030dfd0>, because it is not built.
# # WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20b3310>, because it is not built.
# # WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20c36a0>, because it is not built.
# # WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20cba30>, because it is not built.
# # WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20d9dc0>, because it is not built.
# # WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20ed190>, because it is not built.
# # Traceback (most recent call last):
# #   File "/Users/dttai11/nlp/test.py", line 54, in <module>
# #     model.save_pretrained("my_model", saved_model=True)
# #   File "/Users/dttai11/miniforge3/envs/nlp/lib/python3.8/site-packages/transformers/modeling_tf_utils.py", line 1345, in save_pretrained
# #     self.save(saved_model_dir, include_optimizer=False, signatures=self.serving)
# #   File "/Users/dttai11/miniforge3/envs/nlp/lib/python3.8/site-packages/keras/utils/traceback_utils.py", line 67, in error_handler
# #     raise e.with_traceback(filtered_tb) from None
# #   File "/Users/dttai11/miniforge3/envs/nlp/lib/python3.8/site-packages/tensorflow/python/training/tracking/data_structures.py", line 869, in _checkpoint_dependencies
# #     raise ValueError(
# # ValueError: Unable to save the object {'loss': <function dummy_loss at 0x28d86c430>, 'logits': None} (a dictionary wrapper constructed automatically on attribute assignment). The wrapped dictionary was modified outside the wrapper (its final value was {'loss': <function dummy_loss at 0x28d86c430>, 'logits': None}, its value when a checkpoint dependency was added was None), which breaks restoration on object creation.
# #
# # If you don't need this dictionary checkpointed, wrap it in a non-trackable object; it will be subsequently ignored.
#
# #zArUvzIKpwoqyympFPzkclWgFcEUNsbSpGNxYAtnaMhrNXQwvtrIHWiRHQIfGSmKiRGRRLMCqDIfaARPOPoxXLsVTvyJWSoChphrwytPLDrGUnJDdNImfsXJcYvqdBWB


model = TFAutoModel.from_pretrained("SS8/test2")
model.summary()
