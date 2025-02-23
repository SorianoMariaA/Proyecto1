# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# %%
#file_path = "/content/datos_apartamentos_rent.xlsx"

#file_path = r"C:\Users\Santiago\OneDrive\Documentos\Universidad\Semestres\SEMESTRE 9\Analítica\Proyecto 1\datos_apartamentos_rent.xlsx"

file_path = r"datos_apartamentos_rent.xlsx"

df = pd.read_excel(file_path)

# %%
df = pd.DataFrame(df)

# Se crea una columna con el precio mensual
df["price_monthly"] = df.apply(lambda x: x["price"] * 4 if x["price_type"] == "Weekly" else x["price"], axis=1)


# %% [markdown]
# # Tarea 2

# %%
# EXPLORACIÓN DE DATOS

# Extraer las comodidades de los inmueblde de la columna "amenities"
def get_unique_amenities(df, column="amenities"):
    unique_amenities = set()
    df[column].dropna().apply(lambda x: unique_amenities.update(x.split(",")))
    return list(unique_amenities)


# %%
# Obtener lista sin duplicados
unique_amenities_list = get_unique_amenities(df)
#print(unique_amenities_list)

# %% [markdown]
# # Tarea 3

# %%
# LIMPIEZA DE DATOS

# Se eliminan las filas con valores nulos en las columnas latitude y longitude
df = df.dropna(subset=['latitude', 'longitude'])

# Se detectó que los valores de latitude y longitude se importaron mal, se procede a corregir
df['latitude'] = df['latitude']/10000
df['longitude'] = df['longitude']/10000

# Se reemplazan los datos nulos de address por una respuesta negativa
#df['address'].fillna('No adress given', inplace=True)
df['address'] = df['address'].fillna('No address given')

# Se reemplazan los valores nulos de la columna "pets_allowed" por "None" para que pandas no confunda None con null
df["pets_allowed"] = df["pets_allowed_corrected"].apply(lambda x: "None" if x == "No pets" else x)

# %%
df['cityname_corrected'] = df['cityname']

df.loc[df['cityname'].isna() & (df['latitude'] == 39.8163), 'cityname_corrected'] = 'Lebanon'
df.loc[df['cityname'].isna() & (df['latitude'] == 28.45900), 'cityname_corrected'] = 'Trilby'

# %%
# Se corrige el estado para los datos nulos
df['state_corrected'] = df['state']

df.loc[df['state'].isna() & (df['latitude'] == 39.8163), 'state_corrected'] = 'KS'
df.loc[df['state'].isna() & (df['latitude'] == 28.4590), 'state_corrected'] = 'FL'

#Se verifica que no quedan datos nulos
#null_counts = df.isnull().sum()
#print(null_counts)

# %%
# Se reemplazan los datos nulos de bedrooms por la media
#df['bedrooms'].fillna(df['bedrooms'].mean(), inplace=True)
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mean())

# %%
# Se reemplazan los datos nulos de bathrooms por una regresión lineal (price_monthly, square_feet, bedrooms)

# Imputar bathrooms
bathrooms_notnull = df[df['bathrooms'].notnull()]
bathrooms_null = df[df['bathrooms'].isnull()]
features = ['bedrooms', 'square_feet', 'price_monthly']  # Variables predictoras

model_bathrooms = LinearRegression()
model_bathrooms.fit(bathrooms_notnull[features], bathrooms_notnull['bathrooms'])
df.loc[df['bathrooms'].isnull(), 'bathrooms'] = model_bathrooms.predict(bathrooms_null[features])

#null_counts = df.isnull().sum()
#print(null_counts)

# %%
# IMPORTANTE HACER PIP INSTALL SKLEARN SI NO FUNCIONAN LOS IMPORTS

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Mapear valores de pets_allowed a números
pets_mapping = {
    "None": 0,
    "Cats": 1,
    "Dogs": 2,
    "Cats,Dogs": 3
}
df['pets_allowed'] = df['pets_allowed'].map(pets_mapping)

categorical_features = ['category', 'source']

# Se hace la codificación de las variables categóricas
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Seleccionar las variables predictoras
pets_features = ['bedrooms', 'square_feet', 'price_monthly'] + [col for col in df_encoded.columns if col.startswith('category_') or col.startswith('source_')]

# %%
# Aplicar imputación múltiple con IterativeImputer usando RandomForest
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), max_iter=10)

# Aplicar imputación asegurando que la salida sea un DataFrame
imputed_values = imputer.fit_transform(df_encoded[['pets_allowed'] + pets_features])

# %%
# Convertir el array imputado nuevamente a un DataFrame
df_imputed = pd.DataFrame(imputed_values, columns=['pets_allowed'] + pets_features, index=df_encoded.index)

# Redondear los valores imputados de pets_allowed y convertir a enteros
df_imputed['pets_allowed'] = np.ceil(df_imputed['pets_allowed']).astype(int)

# Sobrescribir los valores imputados en el df original
df['pets_allowed'] = df_imputed['pets_allowed']

# Devolver el mapeo
reverse_mapping = {0: "None", 1: "Cats", 2: "Dogs", 3: "Cats,Dogs"}
df['pets_allowed'] = df['pets_allowed'].map(reverse_mapping)

df['pets_allowed'].value_counts()

# %%
# Se corrigen los valores nulos de amenities mediante el texto de la columna body
# Función para extraer amenities de "body"
def extract_amenities_from_body(body_text, amenities_list):
    if pd.isna(body_text):  # Si el texto es NaN, devolver None
        return None
    found_amenities = [amenity for amenity in amenities_list if amenity.lower() in body_text.lower()] #.lower() devuelve todo el texto en minusc.
    return ",".join(found_amenities) if found_amenities else None #Unir las amenities como la columna original

# %%
# Funcion para completar los valores nulos en "amenities"
def completar_amenities(row):
    if pd.isna(row["amenities"]):  # Si el valor en "amenities" es NaN se extrae los amenities de "body"
        return extract_amenities_from_body(row["body"], unique_amenities_list)
    return row["amenities"]

# Aplicar la función a cada fila
df["amenities"] = df.apply(completar_amenities, axis=1)

# Reemplazar los valores nulos restantes en "amenities" por "No Amenities"
#df["amenities"].fillna("No Amenities", inplace=True)
df["amenities"] = df["amenities"].fillna("No Amenities")

#null_counts = df.isnull().sum()
#print(null_counts)

# %%
# Para descargar el excel
#df.to_excel("/content/datos_apartamentos_rent_actualizado.xlsx", index=False)

# %%
# Codificación de las variables categóricas
categorical_features = ['category', 'pets_allowed', 'cityname_corrected', 'state_corrected', 'source']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

df['year'] = pd.to_datetime(df['time'], unit='s').dt.year
df['month'] = pd.to_datetime(df['time'], unit='s').dt.month
df['day'] = pd.to_datetime(df['time'], unit='s').dt.day


# %%
# Filtrar Outliers
def filter_outliers(df, tolerance=4):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = numeric_cols.difference(['id', 'price'])
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - (std * tolerance)
        upper_bound = mean + (std * tolerance)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df_final = filter_outliers(df)
#print(df_final.describe())

# %% [markdown]
# # Tarea 4 - Revisualización

# %%
#Reondear baños a entero más cercano
df_final["bathrooms"] = np.ceil(df_final["bathrooms"])
df_final["bedrooms"] =np.ceil(df_final["bedrooms"])

# %%
# Categorizar los amenities
for i in unique_amenities_list:
    lista_amen = []
    for j in df_final["amenities"]:
        if i in j:
            lista_amen.append(1)
        else:
            lista_amen.append(0)
    df_final[i] = lista_amen


# Arreglar nulos en cities y states
for i in df_final.iterrows():
    if pd.isna(df_final.at[i[0],"cityname"]):
        df_final.at[i[0],"cityname"] = df_final.columns[np.where(i[1][26:1600] == True)[0][0]+25].split("_")[2]

for i in df_final.iterrows():
    if pd.isna(df_final.at[i[0],"state"]):
        df_final.at[i[0],"state"] = df_final.columns[np.where(i[1][1601:1651] == True)[0][0]+1600].split("_")[2]

# %% [markdown]
# # Tarea 4 -Modelos

# %%
# renombrar categorias largas
df_final = df_final.rename( columns={"category_housing/rent/home":"home","category_housing/rent/short_term":"short_term",
                                     "pets_allowed_Cats,Dogs":"pets_both","pets_allowed_Dogs":"pets_dogs","pets_allowed_None":"pets_None"})
# quitar columnas innecesarias
df_inf = df_final.drop(columns=["id","title","body","has_photo","amenities","currency","fee","price_display","price_type","address","latitude","longitude",
                                "time","pets_allowed_corrected","price_monthly"])
df_inf = df_inf.drop(columns=df_inf.columns[11:1635].values)
df_inf = df_inf.drop(columns=["cityname","year","month","day"])

# %%
# Volver enteras todas las variables incluidas las booleanas
df_inf[["home","short_term","pets_both","pets_dogs","pets_None"]] =df_inf[["home","short_term","pets_both","pets_dogs","pets_None"]].astype(int)
df_inf = pd.get_dummies(df_inf, columns=["state"],drop_first=True).astype(int)

# %% [markdown]
# ### Definición de Modelos

# %%
# Modelo 1: Todas las variables
X_var1 = [i for i in df_inf.columns if i != "price"]

# Modelo 2:Factores Generales y estado
X_var2 = ["bathrooms","bedrooms","square_feet","state_AR","state_AZ", "state_CA", "state_CO", "state_CT", "state_DC", "state_DE", "state_FL",
         "state_GA","state_IA","state_ID","state_IL","state_IN","state_KS","state_KY","state_LA","state_MA", "state_MD","state_ME","state_MI",
         "state_MN","state_MO","state_MS","state_MT","state_NC","state_ND","state_NE","state_NH","state_NJ","state_NM","state_NV","state_NY",
         "state_OH","state_OK","state_OR","state_PA","state_RI","state_SC","state_SD","state_TN","state_TX","state_UT","state_VA","state_VT",
         "state_WA","state_WI","state_WV", "state_WY"]

# Modelo 3: Aminities y caracteristicas
X_var3 = ["bathrooms","bedrooms","square_feet","Garbage Disposal", "TV", "Luxury", "Elevator", "Gym", "Fireplace","Washer Dryer","Doorman",
          "Cable or Satellite","Internet Access","Playground","Parking","Tennis","Gated","Pool","Basketball","View","Clubhouse",
          "Dishwasher","Patio/Deck", "Alarm", "Hot Tub", "Refrigerator", "Golf", "Storage", "Wood Floors", "AC"]

# Modelo 4: Platafroma y tipo de propiedad
X_var4 = ["source_Home Rentals","source_Listanza","source_ListedBuy","source_RENTCafé","source_RENTOCULAR","source_Real Estate Agent",
          "source_RealRentals","source_RentDigs.com","source_RentLingo","source_rentbits","source_tenantcloud","home","short_term"]

# Modelo 5: Mascotas y Segruidad
X_var5 = ["pets_both", "pets_dogs", "pets_None", "bathrooms", "bedrooms", "Gated", "Doorman", "Alarm"]

# Modelo 6: Regresión sencilla
X_var6 = ["bathrooms","bedrooms","square_feet"]

# %% [markdown]
# ### Regresión Lineal

# %%
Resultados_Regresion ={}
def reg_lineal(varX):
    Y_var = "price"
    X_var =varX
    X_inf = df_inf[X_var]
    Y_inf = df_inf[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)

    # agregar constante explíticamente
    X_train = sm.add_constant(X_train)

    # regresión usando mínimos cuadrados ordinarios (ordinary least squares - OLS) 
    model = sm.OLS(y_train, X_train).fit()

    return {'Variables': X_var,'R2': model.rsquared,"R2aj":model.rsquared_adj,"AIC":model.aic,"BIC":model.bic,
            'Coeficientes': model.params,"modelo":model}


# %%
Resultados_Regresion[1]= reg_lineal(X_var1)
Resultados_Regresion[2]= reg_lineal(X_var2)
Resultados_Regresion[3]= reg_lineal(X_var3)
Resultados_Regresion[4]= reg_lineal(X_var4)
Resultados_Regresion[5]= reg_lineal(X_var5)
Resultados_Regresion[6]= reg_lineal(X_var6)

# %% [markdown]
# ### K vecinos

# %%
Resultados_K ={}
def k_model(varX):
    Y_var = "price"
    X_var = varX

    X_inf = df_inf[X_var]
    Y_inf = df_inf[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=10)

    # Entrenar el modelo con los datos de entrenamiento
    knn.fit(X_train_scaled, y_train)

    # Hacer predicciones
    y_pred = knn.predict(X_test_scaled)

    return {'Variables': X_var,'R2':r2_score(y_test, y_pred),"modelo":knn}

# %%
Resultados_K[1]= k_model(X_var1)
Resultados_K[2]= k_model(X_var2)
Resultados_K[3]= k_model(X_var3)
Resultados_K[4]= k_model(X_var4)
Resultados_K[5]= k_model(X_var5)
Resultados_K[6]= k_model(X_var6)

# %% [markdown]
# ### Suport Vector Regression (SVR)

# %%
Resultados_SVR ={}
def SVR_model(varX):
    Y_var = "price"
    X_var = varX
    X_inf = df_inf[X_var]
    Y_inf = df_inf[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')

    # Entrenar el modelo con los datos de entrenamiento
    svr.fit(X_train_scaled, y_train)

    # Hacer predicciones
    y_pred = y_pred = svr.predict(X_test_scaled)

    return {'Variables': X_var,'R2':r2_score(y_test, y_pred),"modelo":svr}


# %%
Resultados_SVR[1]= SVR_model(X_var1)
Resultados_SVR[2]= SVR_model(X_var2)
Resultados_SVR[3]= SVR_model(X_var3)
Resultados_SVR[4]= SVR_model(X_var4)
Resultados_SVR[5]= SVR_model(X_var5)
Resultados_SVR[6]= SVR_model(X_var6)

# %% [markdown]
# #### Modelos Hagalo Usted Mismo

# %%
def pobar_modelos(tipo,variables):
    Y_var = "price"
    X_var =variables
    X_inf = df_inf[X_var]
    Y_inf = df_inf[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)

    if tipo == "Lineal":
        X_train = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train).fit()
        return {'Variables': X_var,'R2': model.rsquared,"R2aj":model.rsquared_adj,"AIC":model.aic,"BIC":model.bic,'Coeficientes': model.params,"modelo":model}
    
    elif tipo == "K":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        return {'Variables': X_var,'R2':r2_score(y_test, y_pred),"modelo":knn}
    
    elif tipo =="SVR":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')

        # Entrenar el modelo con los datos de entrenamiento
        svr.fit(X_train_scaled, y_train)

        # Hacer predicciones
        y_pred = y_pred = svr.predict(X_test_scaled)

        return {'Variables': X_var,'R2':r2_score(y_test, y_pred),"modelo":svr}
    
    

# %%
def pobar_lineal(variables):
    Y_var = "price"
    X_var =variables
    X_inf = df_inf[X_var]
    Y_inf = df_inf[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    return {'Variables': X_var,'R2': model.rsquared,"R2aj":model.rsquared_adj,"AIC":model.aic,"BIC":model.bic,'Coeficientes': model.params,"modelo":model}

def pobar_k(variables):
    Y_var = "price"
    X_var =variables
    X_inf = df_inf[X_var]
    Y_inf = df_inf[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    return {'Variables': X_var,'R2':r2_score(y_test, y_pred),"modelo":knn}

def probar_SVR(variables):
    Y_var = "price"
    X_var = variables
    X_inf = df_inf[X_var]
    Y_inf = df_inf[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')

    # Entrenar el modelo con los datos de entrenamiento
    svr.fit(X_train_scaled, y_train)

    # Hacer predicciones
    y_pred = y_pred = svr.predict(X_test_scaled)

    return {'Variables': X_var,'R2':r2_score(y_test, y_pred),"modelo":svr}


#import json #Para hacer mapa
import dash
from jupyter_dash import JupyterDash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import plotly.express as px
import urllib.request as urllib


# from urllib.request import urlopen
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
import plotly.figure_factory as ff
import numpy as np

app = dash.Dash(__name__)
server = app.server
#Layout de prueba
app.layout = html.Div([
    html.H1("Hola Dash"),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'NYC'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])
if __name__ == "__main__":
    app.run_server(debug=True)