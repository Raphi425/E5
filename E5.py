import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import math

def gerade(a,b,x):
    return(a*x+b)

def mittelwert(x):
    return(np.sum(x)/np.size(x))

def sortiere(x, y):
    xsort = np.sort(x)
    yneu = np.zeros(np.size(y))
    for z in range(np.size(x)):
        for h in range(np.size(x)):
            if xsort[z] == x[h]:
                yneu[z] = y[h]
    return (xsort, yneu)

def steigunggrenz (x, y):
    x, y = sortiere(x, y)
    param = np.polyfit(x, y, 1)
    steigung = param[0]
    n = np.size(x)*0.317
    if (math.ceil(n) -n) > (n - math.floor(n)):
        n = math.floor(n)
    else:
        n = math.ceil(n)
    maxab = np.abs(0.9 * steigung)
    min = np.size(x)
    mittel = np.array([mittelwert(x), mittelwert(y)])
    m = 0
    l = 0
    for z in range(np.size(x)-1):
        if(x[z] > mittel[0]):
            l = z
            break
    for z in range(0, 1001, 1):
        counter = 0
        for k in range(l):
            a = mittel[1] + (steigung-(maxab - z * maxab/1000)) * (x[k]-mittel[0])
            b = mittel[1] + (steigung+(maxab - z * maxab/1000)) * (x[k]-mittel[0])
            if (y[k] > a or y[k] < b):
                    counter = counter + 1

        for k in range(l, np.size(x)-1, 1):
            a = mittel[1] + (steigung-(maxab - z * maxab/1000)) * (x[k] - mittel[0])
            b = mittel[1] + (steigung+(maxab - z * maxab/1000)) * (x[k] - mittel[0])
            if (y[k] < a or y[k] > b):
                    counter = counter + 1
        #print(counter)
        if (np.abs(counter - n) < min):
            min = np.abs(counter - n)
            m = z
    return np.array([steigung -(maxab - m * maxab/1000), steigung + (maxab - m * maxab/1000)])

def standardabweichung(x):
    return np.sqrt(np.sum((x-mittelwert(x))**2)/(np.size(x)-1))

def mittelwertfehler(x):
    return standardabweichung(x)/np.sqrt(np.size(x))


def P(x):
    return((x*6.576**2/(x+1.426)**2))

strom =[]
spannung = []
Rat = np.arange(0,120,1)

f = open('strom.txt')
for line in f:
    line=line.replace(",", ".")
    strom.append(float(line))


f = open('spannung.txt')
for line in f:
    line=line.replace(",", ".")
    spannung.append(float(line))

print(strom)
print(spannung)

ausgleich = np.polyfit(strom, spannung, 1)
mgrenz = steigunggrenz(strom, spannung)
schwx = mittelwert(strom)
schwy = mittelwert(spannung)

leistung = np.multiply(spannung,strom)
Ra =( (ausgleich[0]*0+ausgleich[1]) / strom ) - ausgleich[0]
print(leistung)
print(Ra)


print(str(ausgleich[0]*0+ausgleich[1]) + "   U0")
#print(str(mgrenz[0]*(0-schwx)+schwy) + "   U01")
#print(str(mgrenz[1]*(0-schwx)+schwy) + "   U02")
#print(str(np.abs(standardabweichung(spannung)))+ "         Delta ys")
#print(str(((mgrenz[0]*(0-schwx)+schwy-(mgrenz[1]*(0-schwx)+schwy))/2)+np.abs(standardabweichung(spannung)))+ "         DeltaU0")
print(str(ausgleich[0])+"          Ri")
#print(str(mgrenz[0])+"           Ri1")
#print(str(mgrenz[1])+"             Ri2")
#print(str(np.abs(standardabweichung(strom)))+ "         Delta xs")
#print(str((mgrenz[0]-mgrenz[1])/2)+"    Delta Ri")
#print(str(-ausgleich[1]/ausgleich[0]) +  "      Ik")
#print(str((-schwy/mgrenz[0])+schwx)+  "      Ik1")
#print(str((-schwy/mgrenz[1])+schwx)+  "      Ik2")
#print(str((((-schwy/mgrenz[0])+schwx-((-schwy/mgrenz[1])+schwx))/2)+np.abs(standardabweichung(strom)))+"       Delta Ik")

#plt.plot(schwx,schwy,'bD', label='Schwerpunkt')
plt.plot(Ra, leistung, '.k',label='Messwerte')
plt.plot(4.77101688,7.974096,'dr',label='Punkt der Leistungsanpassung')
#plt.plot(2.683,0.0365,'rd',label='PMax Messwerte')
#plt.plot(2.683, np.multiply(np.power(2.683,8),solar[0])+np.multiply(np.power(2.683,7),solar[1])+np.multiply(np.power(2.683,6),solar[2])+np.multiply(np.power(2.683,5),solar[3])+np.multiply(np.power(2.683,4),solar[4])+np.multiply(np.power(2.683,3),solar[5])+ np.multiply(np.power(2.683,2),solar[6])+np.multiply(solar[7],2.683)+solar[8],'gd',label='PMax Polynom')
#plt.plot(sp,np.multiply(np.power(sp,8),solar[0])+np.multiply(np.power(sp,7),solar[1])+np.multiply(np.power(sp,6),solar[2])+np.multiply(np.power(sp,5),solar[3])+np.multiply(np.power(sp,4),solar[4])+np.multiply(np.power(sp,3),solar[5])+ np.multiply(np.power(sp,2),solar[6])+np.multiply(solar[7],sp)+solar[8],'b',label='Annäherung durch Polynom 8. Grades')
#plt.plot(strom, np.multiply(ausgleich[0],strom)+ausgleich[1],'r-',label='Ausgleichsgerade')
#plt.plot(strom, mgrenz[0]*(strom-schwx)+schwy,'r--',label='Grenzgeraden')
#plt.plot(strom, mgrenz[1]*(strom-schwx)+schwy,'r--')
plt.plot(Rat, P(Rat), 'b',label='theoretisch erwarteter Verlauf')
#plt.plot(Ra, np.multiply(strom,spannung),'.k')
plt.xlabel('Ra in Ω')
plt.ylabel('P in W')
plt.grid()
plt.legend()
plt.show()