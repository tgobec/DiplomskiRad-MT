# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:44:58 2023

@author: User
"""
import nltk
from nltk.translate import IBMModel1, Alignment, AlignedSent


file = open("C:/Users/User/Desktop/Translator-MA/lang-dir/deu-hrv/HrDe50/DeHr50.txt", encoding="utf-16 LE")

entire = []
deu = []
hrv = []

for i in file.readlines():
    entire.append(i.split("\t"))
    

for x in entire:
    deu.append(x[0].lower().strip("\n"))#.split(" "))
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
source_sentences = deu
target_sentences = hrv

translation_model = train_translation_model(source_sentences, target_sentences)

source_sentence = "Versuche einige."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Viel Spaß!"
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Benimm dich."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Achtung!"
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Ich versuche."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Attacke."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Perfekt."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Das könnte weh tun."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Ich bin einen Tourist."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Er rennt."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Er rennt."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Er rennt schnell."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Er rennt sehr schnell."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Er rennt sehr schnell und springt sehr hoch."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Er läuft mit hoher Geschwindigkeit und springt manchmal sehr hoch."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "# Das männliche Subjekt führt die Aktion des Laufens aus, die aus einer schnellen Beschleunigung innerhalb der Einschränkungen der horizontalen Bewegungsachse auf festen Oberflächen besteht; sowie gleichzeitige Höhenveränderungen, sogenannte 'Sprünge'."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Das Mädchen und der Junge hängen rum."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Das Paar hängt rum, macht nicht viel, genießt einfach nur die Gesellschaft, die es bietet."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "Bruder, die Party wird endlich angezündet, Gott sei Dank, Familie."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")
source_sentence = "In einem Loch im Boden lebte ein Hobbit. Kein hässliches, schmutziges, nasses Loch, gefüllt mit Wurmenden und einem klebrigen Geruch, noch ein trockenes, kahles Sandloch, in dem es nichts gab, worauf man sich setzen oder essen konnte: Es war ein Hobbitloch, und das bedeutet Trost."
translated_sentence = translate_sentence(translation_model, source_sentence)
print(f"Source sentence: {source_sentence}")
print(f"Translated sentence: {translated_sentence}")
print("\n")

source_sentences = hrv
target_sentences = deu

translation_model = train_translation_model(source_sentences, target_sentences)


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