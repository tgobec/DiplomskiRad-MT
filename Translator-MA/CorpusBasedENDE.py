# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:55:03 2023

@author: User
"""

file = open("C:/Users/User/Desktop/Translator-MA/lang-dir/deu-eng/deu2.txt", encoding="ANSI")

entire = []
eng = []
deu = []
HRENdict = {}
ENDEdict = {}


for i in file.readlines():
    entire.append(i.split("\t"))
    

for x in entire:
    eng.append(x[0].lower().strip("\n"))#.split(" "))
    deu.append(x[1].lower().strip("\n"))#.split(" "))



for each in range(len(eng)):
    HRENdict[eng[each]] = deu[each]

for each in range(len(deu)):
    ENDEdict[deu[each]] = eng[each]

def translateENtoDE(string):
    string = str(string).lower()
    out = ""
    if string in HRENdict.keys():
        out = out + HRENdict[string] + " "
    else:
        out = "Derzeit unbekannte Übersetzung"
    return out

def translateDEtoEN(string):
    string = str(string).lower()
    out = ""
    if string in ENDEdict.keys():
        out = out + ENDEdict[string] + " "
    else:
        out = "Currently unknown translation"
    return out


print(translateENtoDE("Try some."))
print(translateENtoDE("Have fun!"))
print(translateENtoDE("Behave yourself."))
print(translateENtoDE("Look out!"))
print(translateENtoDE("I try."))
print(translateENtoDE("Attack."))
print(translateENtoDE("Perfect."))
print(translateENtoDE("This might hurt."))
print(translateENtoDE("I'm a tourist."))
print(translateENtoDE("He runs."))
print(translateENtoDE("He is running."))
print(translateENtoDE("He is running fast."))
print(translateENtoDE("He is running very fast."))
print(translateENtoDE("He is running very fast and he is jumping very high."))
print(translateENtoDE("He is running at great speeds while sometimes jumping very high."))
print(translateENtoDE("The male subject is performing the action of running which consists of rapid acceleration within the constraints of the horizontal axis of movement on solid surfaces; as well as, at the same time, performing altitudinal shifts, referred to as 'jumping'."))
print(translateENtoDE("The girl and the boy are hanging out."))
print(translateENtoDE("The couple is hanging out, not doing much of anything, just enjoying the company provided."))
print(translateENtoDE("Bro, the party finna be lit, on god, fam."))
print(translateENtoDE("In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."))


print(translateDEtoEN("Versuche einige."))
print(translateDEtoEN("Viel Spaß!"))
print(translateDEtoEN("Benimm dich."))
print(translateDEtoEN("Achtung!"))
print(translateDEtoEN("Ich versuche."))
print(translateDEtoEN("Attacke."))
print(translateDEtoEN("Perfekt."))
print(translateDEtoEN("Das könnte weh tun."))
print(translateDEtoEN("Ich bin einen Tourist."))
print(translateDEtoEN("Er rennt."))
print(translateDEtoEN("Er rennt."))
print(translateDEtoEN("Er rennt schnell."))
print(translateDEtoEN("Er rennt sehr schnell."))
print(translateDEtoEN("Er rennt sehr schnell und springt sehr hoch."))
print(translateDEtoEN("Er läuft mit hoher Geschwindigkeit und springt manchmal sehr hoch."))
print(translateDEtoEN("Das männliche Subjekt führt die Aktion des Laufens aus, die aus einer schnellen Beschleunigung innerhalb der Einschränkungen der horizontalen Bewegungsachse auf festen Oberflächen besteht; sowie gleichzeitige Höhenveränderungen, sogenannte 'Sprünge'."))
print(translateDEtoEN("Das Mädchen und der Junge hängen rum."))
print(translateDEtoEN("Das Paar hängt rum, macht nicht viel, genießt einfach nur die Gesellschaft, die es bietet."))
print(translateDEtoEN("Bruder, die Party wird endlich angezündet, Gott sei Dank, Fam."))
print(translateDEtoEN("In einem Loch im Boden lebte ein Hobbit. Kein hässliches, schmutziges, nasses Loch, gefüllt mit Wurmenden und einem klebrigen Geruch, noch ein trockenes, kahles Sandloch, in dem es nichts gab, worauf man sich setzen oder essen konnte: Es war ein Hobbitloch, und das bedeutet Trost."))