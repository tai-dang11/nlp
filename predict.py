from transformers import AutoTokenizer
from transformers import TFAutoModelForQuestionAnswering
import tensorflow as tf

model_checkpoint = "/Users/dttai11/nlp/huggingface.co/SS8/Albert-basev2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model.summary()
# text = r"""
# Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
# architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
# Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
# TensorFlow 2.0 and PyTorch.
# """
#
# questions = [
#     "How many pretrained models are available in Transformers?",
#     "What does Transformers provide?",
#     "Transformers provides interoperability between which frameworks?",
# ]

# for question in questions:
#     inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="tf")
#     input_ids = inputs["input_ids"].numpy()[0]
#     text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
#     output = model(inputs)
#     answer_start = tf.argmax(
#         output.start_logits, axis=1
#     ).numpy()[0]  # Get the most likely beginning of answer with the argmax of the score
#     answer_end = (
#         tf.argmax(output.end_logits, axis=1) + 1
#     ).numpy()[0]  # Get the most likely end of answer with the argmax of the score
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
#     print(f"Question: {question}")
#     print(f"Answer: {answer}")



from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "/Users/dttai11/nlp/huggingface.co/SS8/Albert-basev2"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally 
established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 
75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula.The Vatican Library is a research library for history, law, 
philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from 
books published between 1801 and 1990 can be requested in person or by mail. In March 2014, the Vatican Library began an initial four-year project of digitising its collection 
of manuscripts, to be made available online. The Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items.   
Scholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican. The Pre-Lateran period, comprising 
the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant."""

questions = [
    "When was the Vat formally opened?",
    "what is the library for?",
    "for what subjects?",
    "where is the Vat?",
    'how many books does it have?',
    "what is Vatican Apostolic Library?",
    "what happened in the 17th century?",
    "to whom the library open for?",
]


for question in questions:
    print(question_answerer(question=question, context=context))
