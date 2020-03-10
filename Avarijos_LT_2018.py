import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def define(zod):
    print("-" * 50)
    print(zod)

define('Duomenų pasiėmimas iš failo')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\Eimas_02-09.xls'
df = pd.read_excel(filename, sep=";", encoding="ansi")

define('Kokie duomenų tipai laikomi kiekviename stulpelyje?')
df.info()

define('Kiek eilučių ir kiek stulpelių yra duomenyse?')
print(df.shape)
print('Gauname, kad turime 2928 avarijas')

define('Kokie duomenys yra stebimi?')
print(df.columns)

define('Sutvarkymas stulpelių pavadinimų')
df.columns = df.columns.str.replace(' ', '_')
print(df.columns)

#---------------------------------------------------------------------------------------------------------------------
define('Hipotezė:'
       'PRADEDANTIEJI VAIRUOTOJAI 2018 BUVO GERESNI VAIRUOTOJAI NEGU SENJORAI-')
#---------------------------------------------------------------------------------------------------------------------

define('Koks yra vidutinis stebimų žmonių vairavimo stažas?')
print(round(df.Vairavimo_stazas_metais.mean(), 2))

define('Kiek iš viso vairuotojų su stažu? (Išmetame tuščius duomenis)')
df = df[pd.notnull(df.Vairavimo_stazas_metais)]
print(df.Vairavimo_stazas_metais.count())
print('Gauname, kad iš 2928 avarijų 368 yra dėl neužpildytų duomenų, dviračių vairuotojų ar pėsčiųjų kaltės')

define('Vairavimo stažo grupės')
print(df.Vairavimo_stazas_metais.unique())
print('Matome, kad yra grupių su minuso ženklu')

define('Patikriname, kiek avarijų šioje grupėje?')
filtered_data = df[df["Vairavimo_stazas_metais"]<0].Vairavimo_stazas_metais.count()
print(filtered_data)

define('Išmetame grupes, kuriose vairavimo stažas mažiau 0')
df = df[(pd.notnull(df.Vairavimo_stazas_metais)) & ((df.Vairavimo_stazas_metais)>= 0)]
print(df.Vairavimo_stazas_metais.unique())

define('Kiek buvo avarijų dėl vairuotojų neturinčių 2m vairavimo stažo kaltės?')
KlevoLapas = df[df.Vairavimo_stazas_metais <2].Vairavimo_stazas_metais.count()
print(KlevoLapas)

define('Kiek buvo avarijų dėl senjorų kaltės (senjorų vairavimo stažas nuo 47 metų)?')
Senjorai = df[df.Vairavimo_stazas_metais >= 47].Vairavimo_stazas_metais.count()
print(Senjorai)

define('Kiek yra skirtingų vairavimo stažo metų grupių?')
print(df.Vairavimo_stazas_metais.nunique())

define('Vairuotojų vairavimo stažo histograma')
df.Vairavimo_stazas_metais.plot.hist(bins =62, color= "#395280")
plt.xlabel('Vairavimo stažas metais')
plt.ylabel('Vairuotojų skaičius')
plt.title('Vairavimo stažo \nhistograma')
plt.grid(True)
plt.show()

#----------------------------------------------------------------------------------------------------------------------

define('Nors iš padarytų avarijų skaičiaus pagal tiriamas grupes, galima teigti, kad pradedantieji nėra geresni '
       'vairuotojai, tačiau reikia patikrinti sąlygas avarijų metu')

#----------------------------------------------------------------------------------------------------------------------

define ('Kokiomis oro sąlygomis įvyko avarijos vairuojant senjorams?')
OroSal_S = df.loc[df['Vairavimo_stazas_metais'] >=47, 'Meteorologines_salygos'].unique()
print(OroSal_S)

define ('Kokiomis oro sąlygomis įvyko avarijos vairuojant pradedantiesiems?')
OroSal_S = df.loc[df['Vairavimo_stazas_metais'] <2, 'Meteorologines_salygos'].unique()
print(OroSal_S)

define('Patikrinu, kokiomis oro sąlygomis senjorai ir pradedantieji padarė daugiausiai avarijų')
OroSal_S = df.loc[df['Vairavimo_stazas_metais'] >=47, 'Meteorologines_salygos']
OroSal_S.to_excel('SenjoruOroSalygos.xlsx')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\SenjoruOroSalygos.xlsx'
df1 = pd.read_excel(filename, sep=";", encoding="ansi")


OroSal_P = df.loc[df['Vairavimo_stazas_metais'] <2, 'Meteorologines_salygos']
OroSal_P.to_excel('PradedanciujuOroSalygos.xlsx')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\PradedanciujuOroSalygos.xlsx'
df2 = pd.read_excel(filename, sep=";", encoding="ansi")

plt.figure(1)

ax1 = plt.subplot(121)
df1.Meteorologines_salygos.value_counts().plot(kind="bar", color= "#584f75",ax=ax1)
plt.xlabel('Oro sąlygos')
plt.ylabel('Senjorų padarytų avarijų sk.')
plt.grid(True)
plt.tight_layout()

ax2=plt.subplot(122)
df2.Meteorologines_salygos.value_counts().plot(kind="bar", color= "#c6a4a8",ax=ax2)
plt.xlabel('Oro sąlygos')
plt.ylabel('Pradedančiųjų padarytų avarijų sk.')
plt.grid(True)
plt.tight_layout()

plt.show()

#----------------------------------------------------------------------------------------------------------------------

define('Kadangi daugiausiai avarijų padaryta tokiomis pačiomis oro sąlygomis, analizė vyksta toliau. '
       'Tikrinu pagal avarijų metu buvusią kelio dangą')

#----------------------------------------------------------------------------------------------------------------------

define ('Kokia buvo kelio dangos buklė vairuojant senjorams?')
KelioDanga_S = df.loc[df['Vairavimo_stazas_metais'] >=47, 'Dangos_bukle'].unique()
print(KelioDanga_S)

define ('Kokia buvo kelio dangos buklė vairuojant pradedantiesiems?')
KelioDanga_P = df.loc[df['Vairavimo_stazas_metais'] <2, 'Dangos_bukle'].unique()
print(KelioDanga_P)

define('Patikrinu, kokiai dangos buklei esant senjorai ir pradedantieji padarė daugiausiai avarijų')
KelioDanga_S = df.loc[df['Vairavimo_stazas_metais'] >=47, 'Dangos_bukle']
KelioDanga_S.to_excel('SenjoruKelioDanga.xlsx')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\SenjoruKelioDanga.xlsx'
df3 = pd.read_excel(filename, sep=";", encoding="ansi")

KelioDanga_P = df.loc[df['Vairavimo_stazas_metais'] <2, 'Dangos_bukle']
KelioDanga_P.to_excel('PradedanciujuKelioDanga.xlsx')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\PradedanciujuKelioDanga.xlsx'
df4 = pd.read_excel(filename, sep=";", encoding="ansi")

plt.figure(2)

ax3 = plt.subplot(121)
df3.Dangos_bukle.value_counts().plot(kind="bar", color= "#584f75",ax=ax3)
plt.xlabel('Dangos būklė')
plt.ylabel('Senjorų padarytų avarijų sk.')
plt.grid(True)
plt.tight_layout()

ax4 = plt.subplot(122)
df4.Dangos_bukle.value_counts().plot(kind="bar", color= "#c6a4a8",ax=ax4)
plt.xlabel('Dangos būklė')
plt.ylabel('Pradedančiųjų padarytų avarijų sk.')
plt.grid(True)
plt.tight_layout()

plt.show()

#----------------------------------------------------------------------------------------------------------------------

define('Kadangi atsakymai sutapo- ir pradedantieji ir senjorai daugiausiai avarijų padarė esant sausai kelio dangai, '
       'tiriu sekančią sąlygą')

#----------------------------------------------------------------------------------------------------------------------

define ('Koks buvo paros metas vairuojant senjorams?')
ParosMetas_S = df.loc[df['Vairavimo_stazas_metais'] >=47, 'Paros_metas'].unique()
print(ParosMetas_S)

define ('Koks buvo paros metas vairuojant pradedantiesiems?')
ParosMetas_P = df.loc[df['Vairavimo_stazas_metais'] <2, 'Paros_metas'].unique()
print(ParosMetas_P)


define('Patikrinu, kokiam paros metui esant senjorai ir pradedantieji padarė daugiausiai avarijų')
ParosMetas_S = df.loc[df['Vairavimo_stazas_metais'] >=47, 'Paros_metas']
ParosMetas_S.to_excel('SenjoruParosMetas.xlsx')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\SenjoruParosMetas.xlsx'
df5 = pd.read_excel(filename, sep=";", encoding="ansi")

ParosMetas_P = df.loc[df['Vairavimo_stazas_metais'] <2, 'Paros_metas']
ParosMetas_P.to_excel('PradedanciujuParosMetas.xlsx')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\PradedanciujuParosMetas.xlsx'
df6 = pd.read_excel(filename, sep=";", encoding="ansi")

plt.figure(3,figsize=(8,4))

Diena = (df5['Paros_metas'] == 'Diena').sum()
Sutemos_prieblanda = (df5['Paros_metas'] == 'Sutemos (prieblanda)').sum()
Tamsus_paros_metas = (df5['Paros_metas'] == 'Tamsus paros metas').sum()

proportions = [Diena, Sutemos_prieblanda, Tamsus_paros_metas]
ax5 = plt.subplot(121)
explode = (0.1, 0, 0)
spalvos=['#7FB3D5', '#1F618D','#34495E']
plt.pie(proportions, labels = ['Diena', 'Sutemos_prieblanda', 'Tamsus_paros_metas'],explode=explode,
        colors= spalvos, autopct = '%1.2f%%')
plt.title('Senjorai')


Diena = (df6['Paros_metas'] == 'Diena').sum()
Tamsus_paros_metas = (df6['Paros_metas'] == 'Tamsus paros metas').sum()
Sutemos_prieblanda = (df6['Paros_metas'] == 'Sutemos (prieblanda)').sum()

proportions = [Diena, Sutemos_prieblanda, Tamsus_paros_metas]
ax6 = plt.subplot(122)
explode = (0.1, 0, 0)
spalvos=['#7FB3D5', '#1F618D','#34495E']
plt.pie(proportions, labels = ['Diena', 'Sutemos_prieblanda', 'Tamsus_paros_metas'],explode=explode,
        colors= spalvos, autopct = '%1.2f%%')
plt.title('Pradedantieji')

plt.suptitle('Paros metas, kuomet abi grupės padarė daugiausiai avarijų', size = 12)
plt.tight_layout(pad=2)
plt.show()

#----------------------------------------------------------------------------------------------------------------------

define('Pagal tikrintas sąlygas galiu teigti, kad pradedantieji ir senjorai padarė daugiausiai avarijų tokiomis '
       'pačiomis sąlygomis- dienos metu, esant giedram orui ir sausai kelio dangai. Išvada - 2018 metais senjorai '
       'buvo geresni vairuotojai negu vairuotojai, kurių vairavimo stažas iki dviejų metų')

#----------------------------------------------------------------------------------------------------------------------
define('Su linijine regresija ir koreliacijos koeficientu patikrinu, ar galima pasitikėti gautais rezultatais?')

dfGp = df.groupby('Vairavimo_stazas_metais').count().reset_index()
dfGp.to_excel('GrupuotaPagalStaza.xlsx')
dfGp.plot.scatter(x= 'Vairavimo_stazas_metais',y= 'Laikas')

x = dfGp.Vairavimo_stazas_metais.values.reshape(-1,1)
y = dfGp.Laikas.values.reshape(-1,1)
lm = linear_model.LinearRegression()
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
lm.fit(xTrain,yTrain)
yhat = lm.predict(x)
plt.scatter(x,y,color= '#566573')
plt.plot(x, yhat,color = '#EC7063')

define('Kiek  tikslus yra grafikas?:')
print(lm.score(xTest,yTest))
# print(lm.intercept_)
# print(lm.coef_)
print('Koreliacijos koeficientas:')
print(dfGp[['Vairavimo_stazas_metais', 'Laikas']].corr())
print('-stiprus neigiamas ryšys, rezultatai ganėtinai tikslūs')

plt.xlabel('Vairavimo stažas metais')
plt.ylabel('Avarijų skaičius')
plt.title('Vairavimo stažo metais ir avarijų skaičiaus linijinė regresija')
plt.show()


#----------------------------------------------------------------------------------------------------------------------

define('FOR FUN:) '
       'Ar BMW padarė daugiausiai avarijų 2018 metais?')

# 'Kiek automobilių sukėlė avarijas? (Išmetame tuščius duomenis)'
df = df[pd.notnull(df.Marke)]
# 'Patikrinu, kokios markės automobiliai sukėlė daugiausiai avarijų.TOP5'
AutoMarke = df[['Vairavimo_stazas_metais', 'Marke']]
AutoMarke.to_excel('AutomobiliuMarkes.xlsx')
filename = r'C:\Users\Administrator\PycharmProjects\mokslai\eismas\AutomobiliuMarkes.xlsx'
dfM = pd.read_excel(filename, sep=";", encoding="ansi")

dfM.Marke.value_counts().head(5).plot(kind="bar", color= "#395280")
plt.xlabel('Markė')
plt.ylabel('Avarijų sk.')
plt.grid(True)
plt.tight_layout()
plt.show()

define('NE')