from transformers import TFAlbertForQuestionAnswering
from data import model_checkpoint, tokenizer, squad_datasets, datasets,metric
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import TensorBoard

def train(model,batch_size, epochs, lr, train_squad, valid_squad):

    model.compile(optimizer=tfa.optimizers.LAMB(lr,))

    tensorboard_callback = TensorBoard(log_dir="./qa_model_save/logs")

    callbacks = [tensorboard_callback]

    model.fit(
        train_squad,
        validation_data=valid_squad,
        batch_size=batch_size,
        epochs=epochs,
        callbacks = callbacks
    )

    model.push_to_hub('huggingface.co/SS8/albert_squad_2.0')
    tokenizer.push_to_hub('huggingface.co/SS8/albert_squad_2.0')
    model.save_weights('model.h5', save_format="h5")

tf_train_set,tf_validation_set,tokenized_datasets = squad_datasets(datasets)

# albert = TFAlbertForQuestionAnswering.from_pretrained(model_checkpoint)

# train(model = albert, batch_size = 16, epochs=3,lr = 3e-5, train_squad = tf_train_set,valid_squad=tf_validation_set)
albert = TFAlbertForQuestionAnswering.from_pretrained('/Users/dttai11/nlp/huggingface.co/SS8/albert_squad_2.0')
albert.summary()

