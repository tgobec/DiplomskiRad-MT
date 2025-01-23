# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 22:47:20 2023

@author: User
"""
import nltk
from nltk.translate import IBMModel1, Alignment, AlignedSent

#Otvara se paralelni korpus
file = open("C:/Users/User/Desktop/Translator-MA/lang-dir/deu-eng/deu2.txt", encoding="ANSI")

entire = []
eng = []
deu = []

#podijeli paralelni korpus na listu parova
for i in file.readlines():
    entire.append(i.split("\t"))
    
#Raspodijeli parove u dva odvojena popisa za englesku i njemačku verziju korpusa
for x in entire:
    eng.append(x[0].lower().strip("\n"))#.split(" "))
    deu.append(x[1].lower().strip("\n"))#.split(" "))

#kreirajte model za treniranje algoritma IBM Model 1
def train_translation_model(source_sentences, target_sentences):
    bitext = []
    for src_sent, tgt_sent in zip(source_sentences, target_sentences):
        src_words = nltk.word_tokenize(src_sent)
        tgt_words = nltk.word_tokenize(tgt_sent)
        bitext.append(AlignedSent(src_words, tgt_words))
    
    ibm_model = IBMModel1(bitext, 5)
    return ibm_model

#Model je skup povezanih tokeniziranih riječi

#Stvori se funkcija koja će prevesti rečenicu na temelju vjerojatnosti definirane unutar modela
def translate_sentence(translation_model, sentence):
    translated_sentence = []
    for word in sentence.split():
        best_translation = ''
        min_prob = 0.0
        if word in translation_model.translation_table:
            for tgt_word in translation_model.translation_table[word]:
                alignment_prob = translation_model.translation_table[word][tgt_word]
                if alignment_prob > min_prob:
                    min_prob = alignment_prob
                    best_translation = tgt_word
        translated_sentence.append(best_translation if best_translation else word)
    return " ".join(translated_sentence)

# =============================================================================
# Prijevod funkcionira tako da se ulazni niz reže i uspoređuje sa svim pojavljivanjima
# navedene riječi unutar modela za smjer jezika. Funkcija dobiva vjerojatnost na
# temelju svih mogućih prijevoda i provjerava dobivenu vjerojatnost u odnosu na
# najveću vjerojatnost. Ako je dobivena vjerojatnost veća od maksimalne vjerojatnosti,
# ta se riječ ispisuje kao prijevod, inače kod ispisuje ulaznu riječ
# (ovo omogućuje prijevod rečenice "zagađene" posebnim jezikom).
# =============================================================================

# =============================================================================
# Testing
# =============================================================================
source_sentences = eng
target_sentences = deu

translation_model = train_translation_model(source_sentences, target_sentences)


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


source_sentences = deu
target_sentences = eng

translation_model = train_translation_model(source_sentences, target_sentences)



# Er rennt.
# Er rennt.
# Er rennt schnell.
# Er rennt sehr schnell.
# Er rennt sehr schnell und springt sehr hoch.
# Er läuft mit hoher Geschwindigkeit und springt manchmal sehr hoch.
# Das männliche Subjekt führt die Aktion des Laufens aus, die aus einer schnellen Beschleunigung innerhalb der Einschränkungen der horizontalen Bewegungsachse auf festen Oberflächen besteht; sowie gleichzeitige Höhenveränderungen, sogenannte „Sprünge“.
# Das Mädchen und der Junge hängen rum.
# Das Paar hängt rum, macht nicht viel, genießt einfach nur die Gesellschaft, die es bietet.
# Bruder, die Party wird endlich angezündet, Gott sei Dank, Familie.
# In einem Loch im Boden lebte ein Hobbit. Kein hässliches, schmutziges, nasses Loch, gefüllt mit Wurmenden und einem klebrigen Geruch, noch ein trockenes, kahles Sandloch, in dem es nichts gab, worauf man sich setzen oder essen konnte: Es war ein Hobbitloch, und das bedeutet Trost.


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