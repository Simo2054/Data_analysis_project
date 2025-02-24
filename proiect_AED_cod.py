# -*- coding: utf-8 -*-

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols

data = r"Ecommerce_Sales_Prediction_Dataset.csv"

df = pd.read_csv(data)


df.info(), df.head()

numar_inregistrari = len(df)
print(f"Numarul total de inregistrari : {numar_inregistrari}")

valori_null = df.isnull().sum()
print("Numar de valori lipsa pe fiecare coloana :\n", valori_null)

#Daca am fi avut valori lipsa am fi putut folosi functia din pandas ce se scrie in urmatorul mod
#df_curat = df.dropna()
#Acest lucru va sterge randurile cu valori lipsa.

# -------------------------------------------------------------------------------------------------

# Crearea unui datadframe cu coloana Date convertita in format datetime
df_with_datetime = df.copy()
df_with_datetime['Date'] = pd.to_datetime(df_with_datetime['Date'], format='%d-%m-%Y')


# Desfacerea datei în lună, zi, an in df-ul original
df['Date'] = df['Date'].str.split("-")
ziua = df['Date'].str[0].astype(int)
luna = df['Date'].str[1].astype(int)
anul = df['Date'].str[2].astype(int)

df.insert(0, 'Ziua', ziua, allow_duplicates=True)
df.insert(1, 'Luna', luna, allow_duplicates=True)
df.insert(2, 'Anul', anul, allow_duplicates=True)

df.drop(['Date'], axis=1, inplace=True)

# -------------------------------------------------------------------------------------------------

#Distributii:

#PRICE DISTRIBUTION
plt.figure(figsize=(8, 6))
sns.histplot(df['Price'], kde=True, bins=30, color='blue')
plt.title('Price Distribution', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

#Discount Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Discount'], kde=True, bins=30, color='green')
plt.title('Discount Distribution', fontsize=16)
plt.xlabel('Discount', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

#Marketing Spend Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Marketing_Spend'], kde=True, bins=30, color='orange')
plt.title('Marketing Spend Distribution', fontsize=16)
plt.xlabel('Marketing Spend', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

#Customer Segment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Customer_Segment', data=df, palette='Set2')
plt.title('Customer Segment Distribution', fontsize=16)
plt.xlabel('Customer Segment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Price vs Units Sold
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Price', y='Units_Sold', data=df, color='purple', alpha=0.6)
plt.title('Price vs Units Sold', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Units Sold', fontsize=12)
plt.show()

#Discount vs Units Sold
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Discount', y='Units_Sold', data=df, color='teal', alpha=0.6)
plt.title('Discount vs Units Sold', fontsize=16)
plt.xlabel('Discount', fontsize=12)
plt.ylabel('Units Sold', fontsize=12)
plt.show()

# Marketing Spend vs Units Sold
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Marketing_Spend', y='Units_Sold', data=df, color='brown', alpha=0.6)
plt.title('Marketing Spend vs Units Sold', fontsize=16)
plt.xlabel('Marketing Spend', fontsize=12)
plt.ylabel('Units Sold', fontsize=12)
plt.show()

# Pair plot pentru valori numerice
plt.figure(figsize=(12, 10))
sns.pairplot(df[['Price', 'Discount', 'Marketing_Spend', 'Units_Sold']], diag_kind='kde', palette='husl')
plt.suptitle('Pairplot of Features', y=1.02, fontsize=16)
plt.show()

#Sales Trend Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_with_datetime, x='Date', y='Units_Sold',color='green')
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Units Sold')
plt.show()


statistici_agregate = df.agg(
    {
     'Price':['sum','min','max', 'mean', 'median', 'std'],
     'Discount':['sum','min','max', 'mean', 'median', 'std'],
     'Marketing_Spend':['sum','min','max', 'mean', 'median', 'std'],
     'Units_Sold': ['sum','min','max', 'mean', 'median', 'std']
     }
    )

print("Statistici descriptive pentru fiecare coloana numerica:\n", statistici_agregate)
print("---------------------------------------------------------")

# Gruparea datelor din df dupa fiecare categorie de produs
# (Product_Category) si aplicarea unor functii agregate
# pentru mai multe coloane 
Categ_prod = df.groupby(['Product_Category']).agg(
    {
     'Price':['sum','min','max','mean','median','std'],
     'Discount':['sum','min','max','mean','median','std'],
     'Marketing_Spend':['sum','min','max','mean','median','std'],
     'Units_Sold':['sum','min','max','mean','median','std'],
     }
    )
print("Statistici descriptive pentru fiecare coloana numerica, grupate în funcție de categoria de produs", Categ_prod)
print("---------------------------------------------------------")

# Gruparea datelor din df dupa fiecare segment de client
# (Customer_Segment) si aplicarea unor functii agregate
# pentru mai multe coloane 
Segment_clienti_agg = df.groupby(['Customer_Segment']).agg(
    {
     'Price':['sum','min','max','mean','median','std'],
     'Discount':['sum','min','max','mean','median','std'],
     'Marketing_Spend':['sum','min','max','mean','median','std'],
     'Units_Sold':['sum','min','max','mean','median','std'],
     }
    )
print("Statistici descriptive pentru fiecare coloana numerica, grupate în funcție de segmentul de clienți", Segment_clienti_agg)
print("---------------------------------------------------------")

# -------------------------------------------------------------------------------------------------

#CALCULAREA CVARTILELOR PENTRU O SINGURA COLOANA (Price)
q1_price = df['Price'].quantile(0.25)
q2_price = df['Price'].quantile(0.50) #Sau df['Price'].median()
q3_price = df['Price'].quantile(0.75)
print(f"Q1: {q1_price}, Mediana: {q2_price}, Q3: {q3_price}")
print("---------------------------------------------------------")

#Shapiro 1 
print("""Aplicarea testului shapiro pe fiecare CATEGORIE DE PRODUS, 
pentru a verifica distributia valorilor din coloana Units_Sold:\n""")
for categorie in df["Product_Category"].unique():
# Obține lista valorilor unice din coloana Product_Category
    date_grup = df[  categorie == df["Product_Category"]   ]["Units_Sold"]
    stats, p_value = shapiro(date_grup)
    print ("grup:", categorie, "p_value", p_value)
    print(stats)
    sns.histplot(date_grup, label=str(categorie), kde=True)
    plt.legend()
    plt.show()
    print("---------------------------------------------------------\n")
#Deoarece pentru fiecare tip de produs din Product_Category p-value > 0,05
#putem deduce faptul ca fiecare produs are o distributie normala

#LEVENE 1
print("""Aplicarea testului Levene pentru a verifica daca variantele valorilor 
din coloana Units_Sold sunt egale intre categoriile de produse:\n""")
unit_sport = df[  df["Product_Category"] == "Sports"  ] ["Units_Sold"]
unit_toys = df[  df["Product_Category"] == "Toys"  ] ["Units_Sold"]
unit_Home_Decor = df[  df["Product_Category"] == "Home Decor"  ] ["Units_Sold"]
unit_Fashion = df[  df["Product_Category"] == "Fashion"  ] ["Units_Sold"]
unit_Electronics = df[  df["Product_Category"] == "Electronics"  ] ["Units_Sold"]
stats, p_value = levene(unit_sport, unit_toys, unit_Home_Decor, unit_Fashion, unit_Electronics)

print ("val test:", stats, "p_value:", p_value) 
print("---------------------------------------------------------\n")

# Observam faptul ca p-value > 0.05 la fiecare grup, deci testul Levene 
# de verficare a omogenitatii variantelor este indeplinit

#SHAPIRO 2
print("""Aplicarea testului shapiro pe fiecare SEGMENT DE CLIENT, 
pentru a verifica distributia valorilor din coloana Units_Sold:\n""")
for customer in df["Customer_Segment"].unique():
    date_grup = df [  customer == df["Customer_Segment"]   ]["Units_Sold"]
    stats, p_value = shapiro(date_grup)
    print("grup:", customer, "p-value", p_value)
    print(stats)
    sns.histplot(date_grup, label=str(customer), kde=True)
    plt.legend()
    plt.show()
print("---------------------------------------------------------\n")
#p_value > 0.05, distributie normala

#LEVENE 2
print("""Aplicarea testului Levene pentru a verifica daca variantele valorilor 
din coloana Units_Sold sunt egale intre segmentele de client:\n""")
unit_Occasional = df[  df["Customer_Segment"] == "Occasional"  ] ["Units_Sold"]
unit_Premium = df[  df["Customer_Segment"] == "Premium"  ] ["Units_Sold"]
unit_Regular = df[  df["Customer_Segment"] == "Regular"  ] ["Units_Sold"]
stats, p_value = levene(unit_Occasional, unit_Premium, unit_Regular)
print ("val test:", stats, "p_value:", p_value) 
print("---------------------------------------------------------\n")
#p_value < 0.05

# Anova pentru compararea mediilor intre 2 grupuri
# Cele 2 grupuri sunt unitatile vandute pentru customers regular 
# si unitatile vandute pentru customers premium
stats, p_value = f_oneway(unit_Regular, unit_Premium)
print ("val test ANOVA one-way:", stats, "p_value:", p_value)
# p_value > 0.05, de unde putem deduce faptul ca existe diferente intre 
# unitatile vandute catre clientii regular si clientii premium
print("---------------------------------------------------------")

#Deducerea var categoriale
print("Tipurile de date")
print(df.dtypes)
variabile_categoriale = df.select_dtypes(include=['object']).columns
print("Varialbile categoriale identificate :\n", variabile_categoriale)
print("---------------------------------------------------------")
#Putem obseva faptul ca variabilele categoriale sunt: Date, Product_Category si Customer_Segment
#Ele sunt de acest tip pentru ca sunt in fond niste variabile discrete, 
#iar python transforma string-urile de acest gen in tipuri object

# -------------------------------------------------------------------------------------------------
# in df-ul fara dummy, vom folosi MAJUSCULE
# in df-ul cu dummy vom folosi litere mici
# -------------------------------------------------------------------------------------------------

#Corelatie fara dummy
df_fara_dummy = df[ ["Units_Sold", "Marketing_Spend", "Discount", "Price"] ]
corelatia_fara_dummy = df_fara_dummy.corr()
plt.figure()
sns.heatmap(corelatia_fara_dummy, annot=True)

X, Y = df_fara_dummy.drop(columns='Units_Sold'), df_fara_dummy['Units_Sold']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

print("---------------------------------------------------------")
print("Verificare Multicoliniaritate pentru df-ul fara dummies: ")
#Verificare multicoliniaritate
vif_data_fara_dummy = pd.DataFrame()
vif_data_fara_dummy["Variabila"] = X.columns
vif_data_fara_dummy["VIF"] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ]
print (vif_data_fara_dummy)
print("---------------------------------------------------------")
#daca VIF < 5 --> nu exista prb. legate de multicoliniaritate

#REGRESIE IN 2 MODURI, LINEAR SI OLS PENTRU DF ** FARA ** VARIABILE DUMMMY

model_fara_dummy = LinearRegression()
model_fara_dummy.fit(X_train, Y_train)

#Predictions
Y_pred = model_fara_dummy.predict(X_test)
Y_prezis_train = model_fara_dummy.predict(X_train)

#Evaluation
MSE = mean_squared_error(Y_test, Y_pred)
R2 = r2_score(Y_test, Y_pred)

R2_train = r2_score(Y_train, Y_prezis_train)
print("---------------------------------------------------------")
print(f'Mean Squared Error: {MSE:.2f}')
print(f'R-squared: {R2:.3f}')
print("---------------------------------------------------------")

# OLS FARA VARIABILE DUMMY

X_train_sm = sm.add_constant(X_train)

lr_fara_dummy = sm.OLS(Y_train, X_train_sm).fit()

print(lr_fara_dummy.summary())


# -------------------------------------------------------------------------------------------------


#Corelatie fara dummy si cu data
df_fara_dummy_cu_data = df[ ["Units_Sold", "Marketing_Spend", "Discount", "Price", "Ziua", "Luna", "Anul"] ]
corelatia_fara_dummy_cu_data = df_fara_dummy_cu_data.corr()
plt.figure()
sns.heatmap(corelatia_fara_dummy_cu_data, annot=True)

analiza_pe_luna = df.groupby(['Product_Category','Luna']).agg(
    {
     'Price':['sum','min','max','mean','median','std'],
     'Discount':['sum','min','max','mean','median','std'],
     'Marketing_Spend':['sum','min','max','mean','median','std'],
     'Units_Sold':['sum','min','max','mean','median','std'],
     }
    )
print(analiza_pe_luna)
print("---------------------------------------------------------")


# -------------------------------------------------------------------------------------------------



#Transformarea variabilelor categoriale in variabile dummy
df_encoded = pd.get_dummies(df, columns=['Product_Category', 'Customer_Segment'], drop_first=True, dtype = int)

x = pd.DataFrame(df_encoded[ ["Price", "Discount", "Marketing_Spend","Product_Category_Fashion", "Product_Category_Home Decor", "Product_Category_Sports" ,"Product_Category_Toys", "Customer_Segment_Premium", "Customer_Segment_Regular"] ])
y = df_encoded["Units_Sold"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=100)

#Corelatia cu variabile dummy
corelatie_cu_dummy = df_encoded.corr()
plt.figure()
print (corelatie_cu_dummy)

print("---------------------------------------------------------")
print("Verificare Multicoliniaritate pentru df-ul cu dummies: ")
#Verificare multicoliniaritate
vif_data = pd.DataFrame()
vif_data["Variabila"] = x.columns
vif_data["VIF"] = [ variance_inflation_factor(x.values, i) for i in range(x.shape[1]) ]
print (vif_data)
print("---------------------------------------------------------")
#daca VIF < 5 --> nu exista prb. legate de multicoliniaritate

#REGRESIE IN 2 MODURI, LINEAR SI OLS PENTRU VARIABILE DUMMMY

# Model training
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)
y_prezis_train = model.predict(x_train)
# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

r2_train = r2_score(y_train, y_prezis_train)
print("---------------------------------------------------------")
print(f'Mean Squared Error Dummy: {mse:.2f}')
print(f'R-squared Dummy: {r2:.2f}')
print("---------------------------------------------------------")

# OLS pentru variabile DUMMY

x_train_sm = sm.add_constant(x_train)

lr = sm.OLS(y_train, x_train_sm).fit()

print(lr.summary())
print("---------------------------------------------------------")

# -------------------------------------------------------------------------------------------------



# Verificare existență vânzări duplicate în aceeași zi pentru aceeași categorie
duplicates = df.duplicated(subset=["Anul", "Luna", "Ziua", "Product_Category"], keep=False)


# Extragem doar rândurile duplicate pentru analiză
duplicates_df = df[duplicates]
print("---------------------------------------------------------")
# Afișăm rândurile duplicate
print("Vânzări duplicate găsite (aceeași zi, aceeași categorie):")
print(duplicates_df)

print("---------------------------------------------------------")
#valoarea cheltuielilor intre categorii
cheltuieli_categorii = df.groupby("Product_Category")["Marketing_Spend"].sum()

# Afișarea valorilor cheltuielilor pentru fiecare categorie
print("Cheltuieli de marketing pe categorii:")
print(cheltuieli_categorii)

print("---------------------------------------------------------")
# grupeaza datele pe baza coloanei Product_Category si calculeaza statistici
# pentru coloana Marketing_Spend
cheltuieli_extinse = df.groupby("Product_Category")["Marketing_Spend"].agg(['sum', 'mean', 'min', 'max'])
print("\nStatistici extinse pentru cheltuieli de marketing:")
print(cheltuieli_extinse)

#Vizualizare grafica
plt.figure()
cheltuieli_categorii.plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
plt.title("Cheltuieli de Marketing pe Categorii")
plt.xlabel("Categorie de Produs")
plt.ylabel("Suma Cheltuielilor (Marketing_Spend)")
plt.xticks(rotation=45)
plt.show()


model = ols('Units_Sold ~ Luna + C(Product_Category) + Luna:C(Product_Category)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
'''
Interpretarea rezultatelor

    Efecte principale:
        Verifică dacă month sau product_category au efect 
        semnificativ asupra lui units_sold (valoare p < 0.05).
    Efectul de interacțiune:
        Dacă interacțiunea este semnificativă (valoare p < 0.05), 
        înseamnă că efectul unei variabile depinde de nivelurile 
        celeilalte.

    In cazul nostru, p_value > 0.05 => 
    efectul variabilelor exogene asupra variabilelor dependente
    nu este semnificativ
'''

# dependent_variable ~ independent_variable1 + independent_variable2 + interaction_term
# units_sold ~ C(month) + C(product_category) + C(month):C(product_category)

"""
units_sold este variabila dependenta

C() indică faptul că month este tratată ca o variabilă categorică
    - ia fiecare luna la un nivel separat si nu interpreteaza ca ar fii o 
    relatie numerica intre ele

la fel si la product_category

C(month):C(product_category) 
    - reprezinta termenul de interactiune
    - interactiunea poate analiza efectul specific al fiecarei combinatii luna-categorie
    - assign-eaza coeficienti diferiti pentru fiecare combinatie

operatorul "~" este specific bibliotecii "statsmodels" 
si are rolul de a separa variabila dependenta (stanga operatorului)
de variabilele explicative (dreapta operatorului)

operatorul "+" adauga variabile explicative individuale in model



"""