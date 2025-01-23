# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:41:23 2023

@author: User
"""
import nltk
from nltk.translate import IBMModel1, Alignment, AlignedSent


file = open("C:/Users/User/Desktop/Translator-MA/lang-dir/hrv-eng/HrEn50/hrv3-50.txt", encoding="utf-16 LE")

entire = []
eng = []
hrv = []

for i in file.readlines():
    entire.append(i.split("\t"))
    

for x in entire:
    eng.append(x[0].lower().strip("\n"))#.split(" "))
    hrv.append(x[1].lower().strip("\n"))#.split(" "))

def train_translation_model(source_sentences, target_sentences):
    bitext = []
    for src_sent, tgt_sent in zip(source_sentences, target_sentences):
        src_words = nltk.word_tokenize(src_sent)
        tgt_words = nltk.word_tokenize(tgt_sent)
        bitext.append(AlignedSent(src_words, tgt_words))
    
    ibm_model = IBMModel1(bitext, 5)
    return ibm_model

def translate_sentence(translation_model, sentence):
    translated_sentence = []
    for word in sentence.split():
        best_translation = ''
        max_prob = 0.0
        if word in translation_model.translation_table:
            for tgt_word in translation_model.translation_table[word]:
                alignment_prob = translation_model.translation_table[word][tgt_word]
                if alignment_prob > max_prob:
                    max_prob = alignment_prob
                    best_translation = tgt_word
        translated_sentence.append(best_translation if best_translation else word)
    return " ".join(translated_sentence)

# =============================================================================
# Testing
# =============================================================================
source_sentences = eng
target_sentences = hrv

translation_model = train_translation_model(source_sentences, target_sentences)


# He runs.
# He is running.
# He is running fast.
# He is running very fast.
# He is running very fast and he is jumping very high.
# He is running at great speeds while sometimes jumping very high.
# The male subject is performing the action of running which consists of rapid acceleration within the constraints of the horizontal axis of movement on solid surfaces; as well as, at the same time, performing altitudinal shifts, referred to as “jumping”.
# The girl and the boy are hanging out.
# The couple is hanging out, not doing much of anything, just enjoying the company provided.
# Bro, the party finna be lit, on god, fam.

# In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.

source_sentence = "Try some."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Have fun!"
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Behave yourself."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Look out!"
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "I try."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Attack."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Perfect."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "This might hurt."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "I'm a tourist."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "He runs."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "He is running."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "He is running fast."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "He is running very fast."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "He is running very fast and he is jumping very high."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "The male subject is performing the action of running which consists of rapid acceleration within the constraints of the horizontal axis of movement on solid surfaces; as well as, at the same time, performing altitudinal shifts, referred to as 'jumping'."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "The girl and the boy are hanging out."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "The couple is hanging out, not doing much of anything, just enjoying the company provided."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Bro, the party finna be lit, on god, fam."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
print("\n")

source_sentences = hrv
target_sentences = eng

translation_model = train_translation_model(source_sentences, target_sentences)


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


source_sentence = "Probaj malo."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Zabavi se!"
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Lijepo se ponašaj."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Pazi!"
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Pokušavam."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Napad."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Savršen."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Ovo bi moglo boljeti."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Ja sam turist."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "On trči."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "On trči."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "On trči brzo."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "On trči vrlo brzo."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Trči vrlo brzo i skače vrlo visoko."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Trči velikom brzinom dok ponekad skače vrlo visoko."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Muški subjekt izvodi radnju trčanja koja se sastoji od brzog ubrzanja unutar ograničenja horizontalne osi kretanja na čvrstim površinama; kao i, u isto vrijeme, izvođenje visinskih pomaka, koji se nazivaju 'skakanje'."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Djevojčica i dječak se druže."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Par se druži, ne radi puno, samo uživa u društvu."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Brate, zabava je konačno upaljena, zaboga, fam."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "U jednoj rupi u zemlji živio je hobit. Ne gadnu, prljavu, mokru rupu, punu vrhova crva i mirisa koji se cijedi, niti suhu, golu, pješčanu rupu u kojoj nema ičega za sjesti ili jesti: bila je to hobitska rupa, i to znači udobnost."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")