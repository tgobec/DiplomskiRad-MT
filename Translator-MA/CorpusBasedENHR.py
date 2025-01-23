# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:42:37 2023

@author: User
"""

file = open("C:/Users/User/Desktop/Translator-MA/lang-dir/hrv-eng/hrv3.txt", encoding="utf-16 LE")

entire = []
eng = []
hrv = []
HRENdict = {}
ENHRdict = {}


for i in file.readlines():
    entire.append(i.split("\t"))
    

for x in entire:
    eng.append(x[0].lower().strip("\n"))#.split(" "))
    hrv.append(x[1].lower().strip("\n"))#.split(" "))



for each in range(len(eng)):
    HRENdict[eng[each]] = hrv[each]

for each in range(len(hrv)):
    ENHRdict[hrv[each]] = eng[each]

def translateENtoHR(string):
    string = str(string).lower()
    tempList = []
    out = ""
    if string in HRENdict.keys():
        out = out + HRENdict[string] + " "
    else:
        out = "Trenutno nepoznati prijevod"
    return out

def translateHRtoEN(string):
    string = str(string).lower()
    tempList = []
    out = ""
    if string in ENHRdict.keys():
        out = out + ENHRdict[string] + " "
    else:
        out = "Currently unknown translation"
    return out

print(translateENtoHR("Try some."))
print(translateENtoHR("Have fun!"))
print(translateENtoHR("Behave yourself."))
print(translateENtoHR("Look out!"))
print(translateENtoHR("I try."))
print(translateENtoHR("Attack."))
print(translateENtoHR("Perfect."))
print(translateENtoHR("This might hurt."))
print(translateENtoHR("I'm a tourist."))
print(translateENtoHR("He runs."))
print(translateENtoHR("He is running."))
print(translateENtoHR("He is running fast."))
print(translateENtoHR("He is running very fast."))
print(translateENtoHR("He is running very fast and he is jumping very high."))
print(translateENtoHR("He is running at great speeds while sometimes jumping very high."))
print(translateENtoHR("The male subject is performing the action of running which consists of rapid acceleration within the constraints of the horizontal axis of movement on solid surfaces; as well as, at the same time, performing altitudinal shifts, referred to as 'jumping'."))
print(translateENtoHR("The girl and the boy are hanging out."))
print(translateENtoHR("The couple is hanging out, not doing much of anything, just enjoying the company provided."))
print(translateENtoHR("Bro, the party finna be lit, on god, fam."))
print(translateENtoHR("In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."))


print(translateHRtoEN("Probaj malo."))
print(translateHRtoEN("Zabavi se!"))
print(translateHRtoEN("Lijepo se ponašaj."))
print(translateHRtoEN("Pazi!"))
print(translateHRtoEN("Pokušavam."))
print(translateHRtoEN("Napad."))
print(translateHRtoEN("Savršen."))
print(translateHRtoEN("Ovo bi moglo boljeti."))
print(translateHRtoEN("Ja sam turist."))
print(translateHRtoEN("On trči."))
print(translateHRtoEN("On trči."))
print(translateHRtoEN("On trči brzo."))
print(translateHRtoEN("On trči vrlo brzo."))
print(translateHRtoEN("Trči vrlo brzo i skače vrlo visoko."))
print(translateHRtoEN("Trči velikom brzinom dok ponekad skače vrlo visoko."))
print(translateHRtoEN("Muški subjekt izvodi radnju trčanja koja se sastoji od brzog ubrzanja unutar ograničenja horizontalne osi kretanja na čvrstim površinama; kao i, u isto vrijeme, izvođenje visinskih pomaka, koji se nazivaju 'skakanje'."))
print(translateHRtoEN("Djevojčica i dječak se druže."))
print(translateHRtoEN("Par se druži, ne radi puno, samo uživa u društvu."))
print(translateHRtoEN("Brate, zabava je konačno upaljena, zaboga, fam."))
print(translateHRtoEN("U jednoj rupi u zemlji živio je hobit. Ne gadnu, prljavu, mokru rupu, punu vrhova crva i mirisa koji se cijedi, niti suhu, golu, pješčanu rupu u kojoj nema ičega za sjesti ili jesti: bila je to hobitska rupa, i to znači udobnost."))