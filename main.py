import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import create_optimizer
from transformers import TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

imdb = load_dataset("imdb")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
#/Users/dttai11/.cache/huggingface/datasets/imdb
batch_size = 16

tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'label'],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

tf_validation_dataset = tokenized_imdb["train"].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'label'],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)

num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
# batches_per_epoch = 256
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps
)

model.compile(optimizer=optimizer)

model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=1,
)