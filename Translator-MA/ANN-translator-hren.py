# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:16:35 2023

@author: User
"""

from __future__ import absolute_import, division, print_function
# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
#tf.enable_eager_execution() NO ATTRIBUTE, figure out
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import time
print(tf.__version__)#to check the tensorflow version

"""
Imports:
    future se koristi za izraćun predviđanja,
    tesnorflow za rad s umjetnom inteligencijom,
    matplotlib za ispis predviđanja kao grafikon,
    sklearn za rad s strojnim učenjem,
    unicodedata koristi kako bi algoritam imao čisti naćin pretvorbe riječi u
        odgovarajuće ascii brojeve,
    re za rad s regularnim izrazima,
    numpy za matematičke funkcije,
    os za povezivanje algoritma na resurse računala,
    time za mjerenje vremena unutar programa (no such thing as too much data to
                                              mesure)
"""

#open file location with language data
#files look as such:
#       TargetLanguage    OriginalLanguage
path_to_file = "C:/Users/User/Desktop/Translator-MA/lang-dir/hrv-eng/hrv3.txt"

#function to translate strign into ASCII numbers
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
  if unicodedata.category(c) != 'Mn')
def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())
 #making sure that the function-symbols stay but not as part of the word
 #ex. "Hello !" instead of "Hello!" 
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
 #remove non-space whitespaces (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.rstrip().strip()
 #adding a start and an end token to the sentence
  w = '<start> ' + w + ' <end>'
  return w
#1. Remove the accents and diacritics
#2. Clean the sentences
#3. Return word pairs in the format: [Start Language, Target Language]
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-16 LE').read().strip().split('\n')
    #Not sure why, but .txt file encodings were all over the place,
    #annoying for development, leading to being faster just copying the entire
    #code, rather than actually automizing the process
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return word_pairs

#Creating indexes withc class
#i.e. if the word "dad" is 5th in the corpus, the translation "tata" is also
#       both words get the same index, and the ANN finds them that way
class LanguageIndex():
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
   
    self.create_index()
   
  def create_index(self):
    for phrase in self.lang:
      self.vocab.update(phrase.split(' '))
   
    self.vocab = sorted(self.vocab)
   
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(self.vocab):
      self.word2idx[word] = index + 1
   
    for word, index in self.word2idx.items():
      self.idx2word[index] = word
def max_length(tensor):
    return max(len(t) for t in tensor)
def load_dataset(path, num_examples):
    #creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)
    #index language using the class defined above    
    inp_lang = LanguageIndex(hr for en, hr in pairs)
    targ_lang = LanguageIndex(en for en, hr in pairs)
    #input = hr -> en; find croatian pairs
    #outpus = en; find croatian pairs
    
    #make the input and output vectors
   
    #input language sentances
    #rudimentary tokenization, by splitting the sentance at every space
    input_tensor = [[inp_lang.word2idx[s] for s in hr.split(' ')] for en, hr in pairs]
   
    #output language sentances
    #find the english translation of the croatian token
    #IMPORTANT: this looks for one word at a time, and will rip appart target
    #               sentances inside the corpus to find it, unlike corpus trans.
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, hr in pairs]
   
    #Calculate max_length of input and output tensor
    #Set inputs to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
   
    #Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp,padding='post')
                                                              
                                                       
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,maxlen=max_length_tar,padding='post')
                                                                  
                                                                  
   
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar

#training dataset size
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)
#split into sections to limit statistical impobabilities
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

#defining variables and fine tuning inputs
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

def gru(units):
 if tf.test.is_gpu_available():
   return tf.keras.layers.CuDNNGRU(units,
   return_sequences=True,
   return_state=True,
   recurrent_initializer='glorot_uniform')
 else:
   return tf.keras.layers.GRU(units,
   return_sequences=True,
   return_state=True,
   recurrent_activation='sigmoid',
   recurrent_initializer='glorot_uniform')
#encoder turns the regular words into vectors and weights
class Encoder(tf.keras.Model):
 def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
   super(Encoder, self).__init__()
   self.batch_sz = batch_sz
   self.enc_units = enc_units
   self.embedding = tf.keras.layers.Embedding(vocab_size,   embedding_dim)
   self.gru = gru(self.enc_units)

 def call(self, x, hidden):
   x = self.embedding(x)
   output, state = self.gru(x, initial_state = hidden) 
   return output, state
 
 def initialize_hidden_state(self):
   return tf.zeros((self.batch_sz, self.enc_units))
#read the word that the input-language vectors represent, and find the
#output-language vectors
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
       
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
       
    def call(self, x, hidden, enc_output):
     #Adding to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
       
        #1 at the last axis because applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
       
        attention_weights = tf.nn.softmax(score, axis=1)
       
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
       
        x = self.embedding(x)
       
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
       
        #Pass conjoined vector throu the GRU
        output, state = self.gru(x)
       

        output = tf.reshape(output, (-1, output.shape[2]))
       

        x = self.fc(output)
       
        return x, state, attention_weights
       
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))
#run encoder and decoder
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

#training algorithm:
optimizer = tf.keras.optimizers.Adam()
def loss_function(real, pred):
 mask = 1 - np.equal(real, 0)
 loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
 return tf.reduce_mean(loss_)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
 encoder=encoder,
 decoder=decoder)

#split into 10 supersets (in case the corpus is bigger; i.e. german-english)
EPOCHS = 10
for epoch in range(EPOCHS):
 start = time.time()
 
 hidden = encoder.initialize_hidden_state()
 total_loss = 0
 
 for (batch, (inp, targ)) in enumerate(dataset):
   loss = 0
 
   with tf.GradientTape() as tape:
     enc_output, enc_hidden = encoder(inp, hidden)
 
     dec_hidden = enc_hidden
 
     dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] *  BATCH_SIZE, 1) 
 
     #Teacher forcing - feeding the target as the next input
     for t in range(1, targ.shape[1]):
       #passing enc_output to the decoder
       predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
 
       loss += loss_function(targ[:, t], predictions)
 
       #using teacher forcing
       dec_input = tf.expand_dims(targ[:, t], 1)
 
   batch_loss = (loss / int(targ.shape[1]))
 
   total_loss += batch_loss
 
   variables = encoder.variables + decoder.variables
 
   gradients = tape.gradient(loss, variables)
 
   optimizer.apply_gradients(zip(gradients, variables))
 
   if batch % 100 == 0:
     print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
     batch,
     batch_loss.numpy()))
 #saving (checkpoint) the model every 2 epochs
 if (epoch + 1) % 2 == 0:
   checkpoint.save(file_prefix = checkpoint_prefix)
 
 print('Epoch {} Loss {:.4f}'.format(epoch + 1,
   total_loss / N_BATCH))
 print('Time taken for 1 epoch {} sec\n'.format(time.time() -   start))
 
#evaluation function meant to check the similarity between two tensors withing
#within any given translation - will be used to visualise output
def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
 attention_plot = np.zeros((max_length_targ, max_length_inp))
 
 sentence = preprocess_sentence(sentence)
 inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
 inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
 inputs = tf.convert_to_tensor(inputs)
 
 result = ''
 hidden = [tf.zeros((1, units))]
 enc_out, enc_hidden = encoder(inputs, hidden)
 dec_hidden = enc_hidden
 dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
 for t in range(max_length_targ):
   predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
 
 #storing the attention weights to plot later on
   attention_weights = tf.reshape(attention_weights, (-1, ))
   attention_plot[t] = attention_weights.numpy()
   predicted_id = tf.argmax(predictions[0]).numpy()
   result += targ_lang.idx2word[predicted_id] + ' '
   if targ_lang.idx2word[predicted_id] == '<end>':
     return result, sentence, attention_plot
 
 #the predicted ID is fed back into the model
   dec_input = tf.expand_dims([predicted_id], 0)
   return result, sentence, attention_plot

#function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
 fig = plt.figure(figsize=(10,10))
 ax = fig.add_subplot(1, 1, 1)
 ax.matshow(attention, cmap='viridis')
 
 fontdict = {'fontsize': 14}
 
 ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
 ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
plt.show()

#function of translation: put everything together: txt->encode->decode->learn->
#add weight->retest->add weight->try translating
def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
 result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
 
 print('Input: {}'.format(sentence))
 print('Predicted translation: {}'.format(result))
 
 attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
 plot_attention(attention_plot, sentence.split(' '), result.split(' '))
 
 #restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# On trči.
# On trči.
# On trči brzo.
# On trči vrlo brzo.
# Trči vrlo brzo i skače vrlo visoko.
# Trči velikom brzinom dok ponekad skače vrlo visoko.
# Muški subjekt izvodi radnju trčanja koja se sastoji od brzog ubrzanja unutar ograničenja horizontalne osi kretanja na čvrstim površinama; kao i, u isto vrijeme, izvođenje visinskih pomaka, koji se nazivaju "skakanje".
# Djevojčica i dječak se druže.
# Par se druži, ne radi puno, samo uživa u društvu.
# Brate, zabava je konačno upaljena, zaboga, fam.
# U jednoj rupi u zemlji živio je hobit. Ne gadnu, prljavu, mokru rupu, punu vrhova crva i mirisa koji se cijedi, niti suhu, golu, pješčanu rupu u kojoj nema ičega za sjesti ili jesti: bila je to hobitska rupa, i to znači udobnost.


#TESTING:
translate(u'Probaj malo.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Zabavi se!', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Lijepo se ponašaj.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Pazi!', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Pokušavam.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Napad.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Savršen.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Ovo bi moglo boljeti.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Ja sam turist.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'On trci.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'On trci.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'On trci brzo.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'On trci vrlo brzo.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Trci vrlo brzo i skace vrlo visoko.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Trci velikom brzinom dok ponekad skace vrlo visoko.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Muski subjekt izvodi radnju trcanja koja se sastoji od brzog ubrzanja unutar ogranicenja horizontalne osi kretanja na cvrstim površinama; kao i, u isto vrijeme, izvođenje visinskih pomaka, koji se nazivaju "skakanje".', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Djevojcica i djecak se druze.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Par se druzi, ne radi puno, samo uživa u drustvu.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'Brate, zabava je konacno upaljena, zaboga, fam.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'U jednoj rupi u zemlji zivio je hobit. Ne gadnu, prljavu, mokru rupu, punu vrhova crva i mirisa koji se cijedi, niti suhu, golu, pjescanu rupu u kojoj nema icega za sjesti ili jesti: bila je to hobitska rupa, i to znaci udobnost.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)