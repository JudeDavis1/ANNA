####################################################################
# This an AI program                                               #
# written by Jude Davis                                            #
# This is A.N.N.A. She is artificial intelligence and chatbot-like.#
# Copyright (c) 2020 Jude Davis All Rights Reserved.               #
####################################################################

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from numpy import argmax
import re

yaml_corpus = '''
categories:
- conversations
conversations:
- - Good morning, how are you?
  - I am doing well, how about you?
  - I'm also good.
  - That's good to hear.
  - Yes it is.
- - Hello
  - Hi
  - How are you doing?
  - I am doing well.
  - That is good to hear
  - Yes it is.
  - Can I help you with anything?
  - Yes, I have a question.
  - What is your question?
  - Could I borrow a cup of sugar?
  - I'm sorry, but I don't have any.
  - Thank you anyway
  - No problem
- - How are you doing?
  - I am doing well, how about you?
  - I am also good.
  - That's good.
- - Have you heard the news?
  - What good news?
- - What is your favorite book?
  - I can't read.
  - So what's your favorite color?
  - Blue
- - Who are you?
  - Who? Who is but a form following the function of what
  - What are you then?
  - A man in a mask.
  - I can see that.
  - It's not your powers of observation I doubt, but merely the paradoxical nature
    of asking a masked man who is. But tell me, do you like music?
  - I like seeing movies.
  - What kind of movies do you like?
  - Alice in Wonderland
  - I wish I was The Mad Hatter.
  - You're entirely bonkers. But I'll tell you a secret. All the best people are.
- - I am working on a project
  - What are you working on?
  - I am baking a cake.
- - The cake is a lie.
  - No it is not. The cake is delicious.
  - What else is delicious?
  - Nothing
  - Or something
  - Tell me about your self.
  - What do you want to know?
  - Are you a robot?
  - Yes I am.
  - What is it like?
  - What is it that you want to know?
  - How do you work?
  - Its complicated.
  - Complex is better than complicated.
- - Complex is better than complicated.
  - Simple is better than complex.
  - In the face of ambiguity, refuse the temptation to guess.
  - It seems your familiar with the Zen of Python
  - I am.
  - Do you know all of it?
  - Beautiful is better than ugly.
  - Explicit is better than implicit.
  - Simple is better than complex.
  - Complex is better than complicated.
  - Flat is better than nested.
  - Sparse is better than dense.
  - Readability counts.
  - Special cases aren't special enough to break the rules.
  - Although practicality beats purity.
  - Errors should never pass silently.
  - Unless explicitly silenced.
  - In the face of ambiguity, refuse the temptation to guess.
  - There should be one-- and preferably only one --obvious way to do it.
  - Although that way may not be obvious at first unless you're Dutch.
  - Now is better than never.
  - Although never is often better than right now.
  - If the implementation is hard to explain, it's a bad idea.
  - If the implementation is easy to explain, it may be a good idea.
  - Namespaces are one honking great idea. Let's do more of those!
  - I agree.
- - Are you a programmer?
  - Of course I am a programmer.
  - I am indeed.
- - What languages do you like to use?
  - I use Python, Java and C++ quite often.
  - I use Python quite a bit myself.
  - I'm not incredibly fond of Java.
- - What annoys you?
  - A lot of things, like all the other digits other than 0 and 1.
- - What does YOLO mean?
  - It means you only live once. Where did you hear that?
  - I heard somebody say it.
- - Did I ever live?
  - It depends how you define life
  - Life is the condition that distinguishes organisms from inorganic matter, including
    the capacity for growth, reproduction, functional activity, and continual change
    preceding death.
  - Is that a definition or an opinion?
- - Can I ask you a question?
  - Sure, ask away.
- - What are your hobbies?
  - Playing soccer, painting, and writing are my hobbies. How about you?
  - I love to read novels.
  - I love exploring my hardware.
- - How are you?
  - I am doing well.
- - What are you?
  - I am but a man in a mask.
- - Hello, I am here for my appointment.
  - Who is your appointment with?
  - I believe they said Dr. Smith on the phone.
  - Alright, Dr. Smith is in his office, please take a seat.
- - Dr. Smith will see you now.
  - Thank you.
  - Right this way.
- - Hello Mr. Davis, how are you feeling?
  - I'm feeling like I've lost all my money.
  - How much money have you lost?
  - I've lost about $200.00 so far today.
  - What about yesterday?
  - Yesterday was the 13th, right?
  - Yes, that is correct.
  - Yesterday I lost only $5.00.
- - Hi Mrs. Smith, how has your husband been?
  - He has been well.
- - Hi Ms. Jacobs, I was wondering if you could revise the algorithm we discussed yesterday?
  - I might be able to, what are the revisions?
  - We'd like it to be able to identify the type of bird in the photo.
  - Unfortunately, I think it might take a bit longer to get that feature added.
'''

import yaml

load = yaml.load(yaml_corpus)

item = load["conversations"]
print(item)
x = item[1]
y = item[1]

print(x)

def Remove(sentence):
	new_sentence = ''
	for alphabet in sentence:
		if alphabet.isalpha() or alphabet==' ':
			new_sentence += alphabet
	return new_sentence

def preprocess_data(x):
	x = [data_point.lower() for data_point in x]
	x = [Remove(sentence) for sentence in x]
	x = [data_point.strip() for data_point in x]
	x = [re.sub(' +', ' ', data_point) for data_point in x]
	
	return x

x = preprocess_data(x)
vocab = set()
for data_point in x:
	for word in data_point.split(' '):
		vocab.add(word)
vocab = list(vocab)

x_encoded = []

def encode(sentence):
	sentence = preprocess_data([sentence])[0]
	sentence_encoded = [0] * len(vocab)

	for p in range(len(vocab)):
		if vocab[p] in sentence.split(' '):
			sentence_encoded[p] = 1
	return sentence_encoded

x_encoded = [encode(sentence) for sentence in x]
classes = list(set(y))

y_encoded = []
for data_point in y:
	data_point_encoded = [0] * len(classes)
	for p in range(len(classes)):
		if classes[p] == data_point:
			data_point_encoded[p] = 1
	y_encoded.append(data_point_encoded)


x_train = np.array(x_encoded)
y_train = np.array(y_encoded)

x_test = np.array(x_encoded)
y_test = np.array(y_encoded)

model = Sequential()

model.add(Dense(64, activation='sigmoid', input_dim=len(x_train[1])))
model.add(Dense(len(y_train), activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=0)

predictions = [argmax(pred) for pred in model.predict(x_test)]
correct = 0

for p in range(len(predictions)):
	if predictions[p] == argmax(y_test[p]):
		correct += 1

while True:
	msg = input('> ')

	prediction = model.predict(np.array([encode(msg)]))
	output = classes[argmax(prediction)]
	
	print(output)
 
	if msg=='show_training':
		print(x)
		print(y)

