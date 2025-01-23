# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:10:49 2023

@author: User
"""



file = open("C:/Users/User/Desktop/Translator-MA/lang-dir/deu-hrv/deu-hrv.txt", encoding="utf-16 LE")

entire = []
deu = []
hrv = []
HRDEdict = {}
DEHRdict = {}


for i in file.readlines():
    entire.append(i.split("\t"))
    

for x in entire:
    deu.append(x[0].lower().strip("\n"))#.split(" "))
    hrv.append(x[1].lower().strip("\n"))#.split(" "))



for each in range(len(deu)):
    HRDEdict[deu[each]] = hrv[each]

for each in range(len(hrv)):
    HRDEdict[hrv[each]] = deu[each]

def translateHRtoDE(string):
    string = str(string).lower()
    tempList = []
    out = ""
    if string in HRDEdict.keys():
        out = out + HRDEdict[string] + " "
    else:
        out = "Derzeit unbekannte Übersetzung"
    return out

def translateDEtoHR(string):
    string = str(string).lower()
    tempList = []
    out = ""
    if string in DEHRdict.keys():
        out = out + DEHRdict[string] + " "
    else:
        out = "Trenutno nepoznati prijevod"
    return out

print(translateHRtoDE("Probaj malo."))
print(translateHRtoDE("Zabavi se!"))
print(translateHRtoDE("Lijepo se ponašaj."))
print(translateHRtoDE("Pazi!"))
print(translateHRtoDE("Pokušavam."))
print(translateHRtoDE("Napad."))
print(translateHRtoDE("Savršen."))
print(translateHRtoDE("Ovo bi moglo boljeti."))
print(translateHRtoDE("Ja sam turist."))
print(translateHRtoDE("On trči."))
print(translateHRtoDE("On trči."))
print(translateHRtoDE("On trči brzo."))
print(translateHRtoDE("On trči vrlo brzo."))
print(translateHRtoDE("Trči vrlo brzo i skače vrlo visoko."))
print(translateHRtoDE("Trči velikom brzinom dok ponekad skače vrlo visoko."))
print(translateHRtoDE("Muški subjekt izvodi radnju trčanja koja se sastoji od brzog ubrzanja unutar ograničenja horizontalne osi kretanja na čvrstim površinama; kao i, u isto vrijeme, izvođenje visinskih pomaka, koji se nazivaju 'skakanje'."))
print(translateHRtoDE("Djevojčica i dječak se druže."))
print(translateHRtoDE("Par se druži, ne radi puno, samo uživa u društvu."))
print(translateHRtoDE("Brate, zabava je konačno upaljena, zaboga, fam."))
print(translateHRtoDE("U jednoj rupi u zemlji živio je hobit. Ne gadnu, prljavu, mokru rupu, punu vrhova crva i mirisa koji se cijedi, niti suhu, golu, pješčanu rupu u kojoj nema ičega za sjesti ili jesti: bila je to hobitska rupa, i to znači udobnost."))

print(translateDEtoHR("Versuche einige."))
print(translateDEtoHR("Viel Spaß!"))
print(translateDEtoHR("Benimm dich."))
print(translateDEtoHR("Achtung!"))
print(translateDEtoHR("Ich versuche."))
print(translateDEtoHR("Attacke."))
print(translateDEtoHR("Perfekt."))
print(translateDEtoHR("Das könnte weh tun."))
print(translateDEtoHR("Ich bin einen Tourist."))
print(translateDEtoHR("Er rennt."))
print(translateDEtoHR("Er rennt."))
print(translateDEtoHR("Er rennt schnell."))
print(translateDEtoHR("Er rennt sehr schnell."))
print(translateDEtoHR("Er rennt sehr schnell und springt sehr hoch."))
print(translateDEtoHR("Er läuft mit hoher Geschwindigkeit und springt manchmal sehr hoch."))
print(translateDEtoHR("Das männliche Subjekt führt die Aktion des Laufens aus, die aus einer schnellen Beschleunigung innerhalb der Einschränkungen der horizontalen Bewegungsachse auf festen Oberflächen besteht; sowie gleichzeitige Höhenveränderungen, sogenannte 'Sprünge'."))
print(translateDEtoHR("Das Mädchen und der Junge hängen rum."))
print(translateDEtoHR("Das Paar hängt rum, macht nicht viel, genießt einfach nur die Gesellschaft, die es bietet."))
print(translateDEtoHR("Bruder, die Party wird endlich angezündet, Gott sei Dank, Fam."))
print(translateDEtoHR("In einem Loch im Boden lebte ein Hobbit. Kein hässliches, schmutziges, nasses Loch, gefüllt mit Wurmenden und einem klebrigen Geruch, noch ein trockenes, kahles Sandloch, in dem es nichts gab, worauf man sich setzen oder essen konnte: Es war ein Hobbitloch, und das bedeutet Trost."))