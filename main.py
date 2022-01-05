from transformers import Trainer, AlbertTokenizer, TFAlbertForQuestionAnswering, TrainingArguments
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForQuestionAnswering.from_pretrained('albert-base-v2')

squad_v2 = False
# model_checkpoint = "distilbert-base-uncased"
batch_size = 16
from datasets import load_dataset, load_metric
datasets = load_dataset("squad_v2" if squad_v2 else "squad")

pad_on_right = tokenizer.padding_side == "right"
max_length = 384  # The maximum length of a feature (question and context)
doc_stride = 128
def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


tokenized_datasets = datasets.map(
    prepare_train_features, batched=True, remove_columns=datasets["train"].column_names
)

learning_rate = 2e-5
num_train_epochs = 2
weight_decay = 0.01
from  transformers.data.data_collator import tf_default_data_collator

data_collator = tf_default_data_collator
train_set = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
validation_set = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)
# def callback(checkpoint_path, LOGS):
#
#     checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=False, mode='min')
#     reduce_lr = ReduceLROnPlateau(factor=0.50, monitor='loss', patience=3, min_lr=0.000001, verbose=1)
#     tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0, write_graph=True, write_images=True)
#     callbacks_list = [checkpoint, reduce_lr, tensorboard]
#
#     return callbacks_list
#
# checkpoint_path ="/Users/dttai11/nlp/logs"
# LOGS = './logs/tensorboard'
# callbacks_list = callback(checkpoint_path,LOGS)

from transformers import create_optimizer

total_train_steps = (len(tokenized_datasets["train"]) // batch_size) * num_train_epochs

optimizer, schedule = create_optimizer(
    init_lr=learning_rate, num_warmup_steps=0, num_train_steps=total_train_steps
)

model.compile(optimizer=optimizer)

# model.fit(train_set, validation_data=validation_set, epochs=1)

# model.save("/Users/dttai11/nlp/logs")
model.save_pretrained("my_model", saved_model=True)
# model.save("model_name",save_format='tf')


# WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b030dfd0>, because it is not built.
# WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20b3310>, because it is not built.
# WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20c36a0>, because it is not built.
# WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20cba30>, because it is not built.
# WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20d9dc0>, because it is not built.
# WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.dropout.Dropout object at 0x2b20ed190>, because it is not built.
# Traceback (most recent call last):
#   File "/Users/dttai11/nlp/test.py", line 54, in <module>
#     model.save_pretrained("my_model", saved_model=True)
#   File "/Users/dttai11/miniforge3/envs/nlp/lib/python3.8/site-packages/transformers/modeling_tf_utils.py", line 1345, in save_pretrained
#     self.save(saved_model_dir, include_optimizer=False, signatures=self.serving)
#   File "/Users/dttai11/miniforge3/envs/nlp/lib/python3.8/site-packages/keras/utils/traceback_utils.py", line 67, in error_handler
#     raise e.with_traceback(filtered_tb) from None
#   File "/Users/dttai11/miniforge3/envs/nlp/lib/python3.8/site-packages/tensorflow/python/training/tracking/data_structures.py", line 869, in _checkpoint_dependencies
#     raise ValueError(
# ValueError: Unable to save the object {'loss': <function dummy_loss at 0x28d86c430>, 'logits': None} (a dictionary wrapper constructed automatically on attribute assignment). The wrapped dictionary was modified outside the wrapper (its final value was {'loss': <function dummy_loss at 0x28d86c430>, 'logits': None}, its value when a checkpoint dependency was added was None), which breaks restoration on object creation.
#
# If you don't need this dictionary checkpointed, wrap it in a non-trackable object; it will be subsequently ignored.
