#!/usr/bin/env python3
import pandas as pd
import numpy as np
import calendar
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Caricamento dei dati storici
df = pd.read_csv("../data/01_input_history.csv")

# Conversione della colonna 'Month' in datetime e creazione delle colonne 'Year' e 'Month'
df['Date'] = pd.to_datetime(df['Month'], format='%b%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df = df.drop(columns=['Date'])

# Filtra i record con Quantity diversa da 0
df = df[df['Quantity'] != 0]

# Definizione di features e target
X = df[['Country', 'Product', 'Year', 'Month']]
y = df['Quantity']

# Preprocessing: OneHotEncoder per le variabili categoriche
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), ['Country', 'Product']),
    ('num', 'passthrough', ['Year', 'Month'])
])

# Suddivisione in training e test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione della pipeline con preprocessore e modello Decision Tree
pipeline_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Addestramento del modello sul training set
pipeline_dt.fit(X_train, y_train)

# Valutazione del modello
y_pred_train = pipeline_dt.predict(X_train)
y_pred_test = pipeline_dt.predict(X_test)
train_score = r2_score(y_train, y_pred_train)
test_score = r2_score(y_test, y_pred_test)
print(f"Decision Tree Train R^2 score: {train_score:.8f}")
print(f"Decision Tree Test R^2 score: {test_score:.8f}")

# -----------------------------
# Previsione per il 2024
# -----------------------------
# Creazione del DataFrame per le previsioni del 2024
# Otteniamo le combinazioni uniche di Country e Product dai dati storici
countries = df['Country'].unique()
products = df['Product'].unique()

predictions_list = []
for country in countries:
    for product in products:
        for month in range(1, 13):
            predictions_list.append({
                'Country': country,
                'Product': product,
                'Year': 2024,
                'Month': month
            })

df_pred = pd.DataFrame(predictions_list)

# Esecuzione delle previsioni per il 2024
df_pred['Quantity'] = pipeline_dt.predict(df_pred)
df_pred['Quantity'] = df_pred['Quantity'].clip(lower=0).astype(int)

# Trasforma mese e anno nel formato "%b%Y"
df_pred['date'] = df_pred.apply(lambda x: f"{x['Month']:02d}{x['Year']}", axis=1)
df_pred['Month'] = pd.to_datetime(df_pred['date'], format='%m%Y').dt.strftime('%b%Y')
df_pred.drop(columns=['Year', 'date'], inplace=True)

# Riordina le colonne secondo il formato desiderato: Country, Product, Month, Quantity
df_pred = df_pred[['Country', 'Product', 'Month', 'Quantity']]

# Salva il file di output
output_file = "../data/output.csv"
df_pred.to_csv(output_file, index=False)
print(f"File '{output_file}' creato con successo!")