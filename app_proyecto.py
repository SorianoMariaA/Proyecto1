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

# from urllib.request import urlopen
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import urllib.request as urllib
import base64
from io import BytesIO

import dash
from dash import dcc, html, Input, Output, callback_context, State, ALL
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import folium

df =pd.DataFrame(pd.read_excel(r"datos_apartamentos_rent_modelos.xlsx"))
#df_mapa =pd.DataFrame(pd.read_excel(r"datos_apartamentos_rent_actualizado.xlsx"))
df[["home","short_term","pets_both","pets_dogs","pets_None"]] =df[["home","short_term","pets_both","pets_dogs","pets_None"]].astype(int)
df = pd.get_dummies(df, columns=["state"],drop_first=True).astype(int)
df = df.rename(columns ={"source_RENTCafé":"source_RENTCafe", "source_RentDigs.com":"source_RentDigs"})

# Modelo 1: Todas las variables
X_var1 = [i for i in df.columns if i != "price"]
# Modelo 2:Factores Generales y estado
X_var2 = ["bathrooms","bedrooms","square_feet","state_AR","state_AZ", "state_CA", "state_CO", "state_CT", "state_DC", "state_DE", "state_FL",
         "state_GA","state_IA","state_ID","state_IL","state_IN","state_KS","state_KY","state_LA","state_MA", "state_MD","state_ME","state_MI",
         "state_MN","state_MO","state_MS","state_MT","state_NC","state_ND","state_NE","state_NH","state_NJ","state_NM","state_NV","state_NY",
         "state_OH","state_OK","state_OR","state_PA","state_RI","state_SC","state_SD","state_TN","state_TX","state_UT","state_VA","state_VT",
         "state_WA","state_WI","state_WV", "state_WY"]
# Modelo 3: Factores Generales y Aminities 
X_var3 = ["bathrooms","bedrooms","square_feet","Garbage Disposal", "TV", "Luxury", "Elevator", "Gym", "Fireplace","Washer Dryer","Doorman",
          "Cable or Satellite","Internet Access","Playground","Parking","Tennis","Gated","Pool","Basketball","View","Clubhouse",
          "Dishwasher","Patio/Deck", "Alarm", "Hot Tub", "Refrigerator", "Golf", "Storage", "Wood Floors", "AC"]
# Modelo 4: Platafroma y tipo de propiedad
X_var4 = ["source_Home Rentals","source_Listanza","source_ListedBuy","source_RENTCafe","source_RENTOCULAR","source_Real Estate Agent",
          "source_RealRentals","source_RentDigs","source_RentLingo","source_rentbits","source_tenantcloud","home","short_term"]
# Modelo 5: Mascotas y Segruidad
X_var5 = ["pets_both", "pets_dogs", "pets_None", "bathrooms", "bedrooms", "Gated", "Doorman", "Alarm"]
# Modelo 6: Regresión sencilla
X_var6 = ["bathrooms","bedrooms","square_feet"]

Resultados_Regresion ={}
def reg_lineal(varX):
    Y_var = "price"
    X_var =varX
    X_inf = df[X_var]
    Y_inf = df[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)

    # agregar constante explíticamente
    X_train = sm.add_constant(X_train)

    # regresión usando mínimos cuadrados ordinarios (ordinary least squares - OLS) 
    model = sm.OLS(y_train, X_train).fit()

    return {'Variables': X_var,'R2': model.rsquared,"R2aj":model.rsquared_adj,"AIC":model.aic,"BIC":model.bic,
            'Coeficientes': model.params,"pvalue":model.pvalues,"modelo":model}

Resultados_K ={}
def k_model(varX):
    Y_var = "price"
    X_var = varX

    X_inf = df[X_var]
    Y_inf = df[Y_var]
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

Resultados_SVR ={}
def SVR_model(varX):
    Y_var = "price"
    X_var = varX
    X_inf = df[X_var]
    Y_inf = df[Y_var]
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

Resultados_Regresion[1]= reg_lineal(X_var1)
Resultados_Regresion[2]= reg_lineal(X_var2)
Resultados_Regresion[3]= reg_lineal(X_var3)
Resultados_Regresion[4]= reg_lineal(X_var4)
Resultados_Regresion[5]= reg_lineal(X_var5)
Resultados_Regresion[6]= reg_lineal(X_var6)

Resultados_K[1]= k_model(X_var1)
Resultados_K[2]= k_model(X_var2)
Resultados_K[3]= k_model(X_var3)
Resultados_K[4]= k_model(X_var4)
Resultados_K[5]= k_model(X_var5)
Resultados_K[6]= k_model(X_var6)

Resultados_SVR[1]= SVR_model(X_var1)
Resultados_SVR[2]= SVR_model(X_var2)
Resultados_SVR[3]= SVR_model(X_var3)
Resultados_SVR[4]= SVR_model(X_var4)
Resultados_SVR[5]= SVR_model(X_var5)
Resultados_SVR[6]= SVR_model(X_var6)

#Mapa
# Agrupar por estado y calcular los promedios
# grouped_data = df_mapa.groupby('state').agg({
#     'latitude': 'mean',
#     'longitude': 'mean',
#     'price': 'mean'
# }).reset_index()

# # Crear mapa de Folium
# mapa = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# for index, row in grouped_data.iterrows():
#     folium.CircleMarker(
#         [row['latitude'], row['longitude']],
#         radius=10,
#         fill=True,
#         fill_opacity=0.6,
#         popup=f"Precio Promedio: ${row['price']:.2f}",
#         color='crimson'
#     ).add_to(mapa)

# # # Guardar el mapa de Folium como archivo HTML
# mapa.save("mapa.html")

#Crear controles
states_options = [
    {"label": "District of Columbia", "value": "DC"},
    {"label": "Indiana", "value": "IN"},
    {"label": "Virginia", "value": "VA"},
    {"label": "Washington", "value": "WA"},
    {"label": "New York", "value": "NY"},
    {"label": "California", "value": "CA"},
    {"label": "Arizona", "value": "AZ"},
    {"label": "North Carolina", "value": "NC"},
    {"label": "Texas", "value": "TX"},
    {"label": "Georgia", "value": "GA"},
    {"label": "Florida", "value": "FL"},
    {"label": "Alabama", "value": "AL"},
    {"label": "Maryland", "value": "MD"},
    {"label": "Colorado", "value": "CO"},
    {"label": "New Mexico", "value": "NM"},
    {"label": "Illinois", "value": "IL"},
    {"label": "Tennessee", "value": "TN"},
    {"label": "Alaska", "value": "AK"},
    {"label": "Massachusetts", "value": "MA"},
    {"label": "New Jersey", "value": "NJ"},
    {"label": "Oregon", "value": "OR"},
    {"label": "Delaware", "value": "DE"},
    {"label": "Pennsylvania", "value": "PA"},
    {"label": "Iowa", "value": "IA"},
    {"label": "South Carolina", "value": "SC"},
    {"label": "Minnesota", "value": "MN"},
    {"label": "Michigan", "value": "MI"},
    {"label": "Kentucky", "value": "KY"},
    {"label": "Wisconsin", "value": "WI"},
    {"label": "Ohio", "value": "OH"},
    {"label": "Connecticut", "value": "CT"},
    {"label": "Rhode Island", "value": "RI"},
    {"label": "Nevada", "value": "NV"},
    {"label": "Utah", "value": "UT"},
    {"label": "Missouri", "value": "MO"},
    {"label": "Oklahoma", "value": "OK"},
    {"label": "New Hampshire", "value": "NH"},
    {"label": "Nebraska", "value": "NE"},
    {"label": "Louisiana", "value": "LA"},
    {"label": "North Dakota", "value": "ND"},
    {"label": "Arkansas", "value": "AR"},
    {"label": "Kansas", "value": "KS"},
    {"label": "Idaho", "value": "ID"},
    {"label": "Hawaii", "value": "HI"},
    {"label": "Montana", "value": "MT"},
    {"label": "Vermont", "value": "VT"},
    {"label": "South Dakota", "value": "SD"},
    {"label": "West Virginia", "value": "WV"},
    {"label": "Mississippi", "value": "MS"},
    {"label": "Maine", "value": "ME"},
    {"label": "Wyoming", "value": "WY"}
]

source_options = [
    {"label": "Home Rentals",  "value": "source_Home Rentals"},
    {"label":  "Listanza",  "value": "source_Listanza"},
    {"label":  "ListedBuy",  "value": "source_ListedBuy"},
    {"label":  "RENTCafé",  "value": "source_RENTCafé"},
    {"label":  "RENTOCULAR",  "value": "source_RENTOCULAR"},
    {"label":  "Real Estate Agent",  "value": "source_Real Estate Agent"},
    {"label":  "RealRentals",  "value": "source_RealRentals"},
    {"label":  "RentDigs.com",  "value": "source_RentDigs.com"},
    {"label":  "RentLingo",  "value": "source_RentLingo"},
    {"label":  "rentbits",  "value": "source_rentbits"},
    {"label":  "tenantcloud",  "value": "source_tenantcloud"}
    
]
list_amenities = ['Tennis', 'Cable or Satellite', 'Elevator', 'Luxury', 'Gated', 'Hot Tub', 'Patio/Deck', 'Storage', 'Fireplace', 'Clubhouse', 'View', 'AC', 'Gym', 'TV', 'Golf', 'Dishwasher', 'Washer Dryer', 'Doorman', 'Wood Floors', 'Refrigerator', 'Playground', 'Basketball', 'Garbage Disposal', 'Parking', 'Internet Access', 'Alarm', 'Pool']

# Grafico de correlación
def correlation_plot():
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['price', 'bathrooms', 'bedrooms', 'square_feet']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f'data:image/png;base64,{encoded_image}'

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.config.suppress_callback_exceptions = True
server = app.server

# Layout del dashboard
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Predict Realty",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    dbc.Tabs([
        dbc.Tab(label="Panorama Inmobiliario", tab_id="tab1"),
        dbc.Tab(label="Variables de Interés", tab_id="tab5"),
        dbc.Tab(label="Insights del Mercado", tab_id="tab2"),
        dbc.Tab(label="Predicción Personalizada", tab_id="tab3"),
        dbc.Tab(label="Análisis Técnico", tab_id="tab4"),
    ], id="tabs", active_tab="tab1"),
    html.Div(id='tab-content', className='p-4')
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_content(tab):
    if tab == "tab1":
        return html.Div([
            html.H2("Panorama Inmobiliario"),
            html.P("Obtenga una visión rápida y detallada del mercado de alquileres con estadísticas clave y análisis interactivos."),
            html.H4("Estadísticas Clave"),
            dbc.Row([
                dbc.Col(dbc.Card([html.H5("Precio Promedio de Alquiler"), html.P("$1,416.75 USD")], body=True)),
                dbc.Col(dbc.Card([html.H5("Mediana del Precio de Alquiler"), html.P("$1,259 USD")], body=True)),
                dbc.Col(dbc.Card([html.H5("Mínimo Y Máximo"), html.P("$200 USD - $5,350 USD")], body=True)),
            ]),
            html.H4("Precio Promedio por Estado"),
            #dcc.Graph(figure=fig),
            html.Iframe(srcDoc=open('mapa.html', 'r').read(), width='100%', height='600')
            #dcc.Graph(figure=px.box(df_final, x='state', y='price', title="Distribución de Precios por Estado")),
        ])
    elif tab == "tab5":
        return html.Div([
            html.H3("Variables Clave en el Mercado Inmobiliario"),
            html.P("Para entender mejor los factores que influyen en los precios de renta de las propiedades, hemos seleccionado variables clave que nos permiten analizar patrones y tendencias del mercado."),
            
            html.H4("\U0001F3E0 Características de la Propiedad"),
            html.Ul([
                html.Li("\U0001F6C1 Número de Baños: Más baños suelen estar asociados con propiedades más grandes o lujosas."),
                html.Li("\U0001F6CF Número de Habitaciones: Una mayor cantidad de habitaciones puede incrementar el valor de renta, especialmente en zonas familiares o con alta demanda."),
                html.Li("📏 Tamaño (square_feet): El área total de la propiedad en pies cuadrados, una variable fundamental en la determinación del precio."),
                html.Li("🏡🏢 Tipo de Propiedad: Se diferencia entre casas y apartamentos, ya que cada uno tiene dinámicas de precios distintas."),
                html.Li("📅 Tipo de Renta: Se clasifica en corto y largo plazo. Las rentas a corto plazo suelen tener precios más altos por noche, pero varían según la demanda."),
            ]),
            
            html.H4("🏢 Comodidades y Amenidades"),
            html.P("Las amenidades disponibles pueden aumentar significativamente el precio de renta. Algunas de las más relevantes incluyen:"),
            html.Ul([
                html.Li("🏋️ Gimnasio"),
                html.Li("🏊‍♂️ Piscina"),
                html.Li("🚗 Estacionamiento"),
                html.Li("📡 Internet de Alta Velocidad"),
                html.Li("🔥 Chimenea"),
                html.Li("🔒 Seguridad y Acceso Privado"),
            ]),
            
            html.H4("📍 Ubicación y Factores Externos"),
            html.Ul([
                html.Li("📍 Estado: Diferentes estados en EE.UU. tienen costos de vida y regulaciones que afectan los precios de renta."),
                html.Li("🔗 Fuente: Indica de qué plataforma proviene la información (Zillow, Craigslist, etc.), lo que puede influir en los datos reportados."),
            ]),
            
            html.H3("🔎 ¿Cómo Usamos Estas Variables?"),
            html.P("Utilizamos estas variables para generar gráficos interactivos y modelos predictivos que ayudan a entender la evolución del mercado y a tomar mejores decisiones de inversión."),
            
        ])
    elif tab == "tab2":
        return html.Div([
            html.H2("Insights del Mercado Inmobiliario"),
            html.P("Explora las tendencias y los patrones del mercado inmobiliario."),
    
            html.Br(),
            dcc.Graph(figure=px.scatter(df, x='square_feet', y='price', title="Relación Tamaño-Precio", color ='bathrooms')),
            html.Br(),
            html.H4("Genera tu propio gráfico"),
            html.Label("Selecciona el tipo de gráfico:"),
            dcc.RadioItems(
                id='graph-type',
                options=[
                    {'label': 'Box Plot', 'value': 'box'},
                    #{'label': 'Scatter Plot', 'value': 'scatter'},
                    {'label': 'Correlation Plot', 'value': 'corr'}
                ],
                value='box',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.Div([
                html.Div([
                    html.Label(" variables categoricas :"),
                    dcc.Dropdown(
                        id='box-category',
                        options=["bathrooms","bedrooms","home", "short_term", "pets_both", "pets_dogs", "pets_None", "source_Home Rentals", "source_Listanza", "source_ListedBuy", "source_RENTCafe", "source_RENTOCULAR", "source_Real Estate Agent", "source_RealRentals", "source_RentDigs", "source_RentLingo", "source_rentbits", "source_tenantcloud", "Patio/Deck", "Doorman", "Cable or Satellite", "Hot Tub", "Washer Dryer", "AC", "Luxury", "Dishwasher", "Elevator", "Refrigerator", "Internet Access", "Tennis", "Fireplace", "View", "Wood Floors", "Basketball", "Parking", "Gym", "TV", "Clubhouse", "Storage", "Garbage Disposal", "Gated", "Pool", "Playground", "Alarm", "Golf", "state_AR", "state_AZ", "state_CA", "state_CO", "state_CT", "state_DC", "state_DE", "state_FL", "state_GA", "state_IA", "state_ID", "state_IL", "state_IN", "state_KS", "state_KY", "state_LA", "state_MA", "state_MD", "state_ME", "state_MI", "state_MN", "state_MO", "state_MS", "state_MT", "state_NC", "state_ND", "state_NE", "state_NH", "state_NJ", "state_NM", "state_NV", "state_NY", "state_OH", "state_OK", "state_OR", "state_PA", "state_RI", "state_SC", "state_SD", "state_TN", "state_TX", "state_UT", "state_VA", "state_VT", "state_WA", "state_WI", "state_WV", "state_WY"],
                        value='home'
                    )
                ], id="box-options", style={'marginTop': '10px'}),
                # html.Div([
                #     html.Label("Variables numericas:"),
                #     dcc.Dropdown(
                #         id='scatter-x',
                #         options=["bathrooms","bedrooms","square_feet"],
                #         value='bathrooms'
                #     )
                # ], id="scatter-options", style={'marginTop': '10px', 'display': 'none'}),
                html.Div("El gráfico de correlación muestra la relación entre y, bathrooms,bedrooms,square_feet", 
                        id="corr-info", style={'marginTop': '10px', 'display': 'none'})
            ]),
            dcc.Graph(id='graph-output')
        ])
    
    elif tab == "tab3":
        return html.Div([
            html.H2("🔍 Predicción Personalizada", style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '10px'}),
            html.P("Ingrese los detalles específicos de una propiedad y obtenga una predicción del precio de alquiler utilizando nuestros modelos avanzados de regresión.",
                style={'textAlign': 'center', 'color': '#7F8C8D', 'fontSize': '16px', 'marginBottom': '20px'}),

            html.Div([
                html.Label("Selecciona el modelo:", style={'fontWeight': 'bold', 'color': '#34495E'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {"label": "Regresión Lineal", "value": "Lineal"},
                        {"label": "K-Vecinos Cercanos", "value": "K"},
                        {"label": "Máquinas de Soporte Vectorial (SVR)", "value": "SVR"}
                    ],
                    value='Lineal',
                    style={'marginBottom': '15px'}
                ),
                
                html.Label("Selecciona las variables:", style={'fontWeight': 'bold', 'color': '#34495E'}),
                dcc.Dropdown(
                    id='variable-dropdown',
                    options=[{"label": var.replace("_", " ").title(), "value": var} for var in [
                        "bathrooms", "bedrooms", "square_feet", "home", "short_term", "pets_both", "pets_dogs", "pets_none", 
                        "source_Home Rentals", "source_Listanza", "source_ListedBuy", "source_RENTCafe", "source_RENTOCULAR", 
                        "source_Real Estate Agent", "source_RealRentals", "source_RentDigs", "source_RentLingo", "source_rentbits", 
                        "source_tenantcloud", "Patio/Deck", "Doorman", "Cable or Satellite", "Hot Tub", "Washer Dryer", "AC", "Luxury", 
                        "Dishwasher", "Elevator", "Refrigerator", "Internet Access", "Tennis", "Fireplace", "View", "Wood Floors", 
                        "Basketball", "Parking", "Gym", "TV", "Clubhouse", "Storage", "Garbage Disposal", "Gated", "Pool", "Playground", 
                        "Alarm", "Golf", "state_AR", "state_AZ", "state_CA", "state_CO", "state_CT", "state_DC", "state_DE", "state_FL", 
                        "state_GA", "state_IA", "state_ID", "state_IL", "state_IN", "state_KS", "state_KY", "state_LA", "state_MA", 
                        "state_MD", "state_ME", "state_MI", "state_MN", "state_MO", "state_MS", "state_MT", "state_NC", "state_ND", 
                        "state_NE", "state_NH", "state_NJ", "state_NM", "state_NV", "state_NY", "state_OH", "state_OK", "state_OR", 
                        "state_PA", "state_RI", "state_SC", "state_SD", "state_TN", "state_TX", "state_UT", "state_VA", "state_VT", 
                        "state_WA", "state_WI", "state_WV", "state_WY"
                    ]],
                    multi=True,
                    value=["square_feet"],
                    style={'marginBottom': '15px'}
                ),
            ], style={'width': '50%', 'margin': '0 auto'}),

            html.Div(id='variable-inputs', style={'marginTop': '10px'}),
            dcc.Store(id='store-input-values', data={}),
            
            html.Div(style={'textAlign': 'center', 'marginTop': '20px'}, children=[
                html.Button('📊 Predecir Precio', id='predict-button', n_clicks=0,
                            style={'backgroundColor': '#2980B9', 'color': 'white', 'borderRadius': '5px',
                                'padding': '10px 20px', 'border': 'none', 'fontSize': '16px', 'cursor': 'pointer'}),
                html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '18px'})
            ])
        ])

                
    elif tab == 'tab4':
            return html.Div([
                html.H3("Análisis Técnico", style={'color': '#1565C0'}),
                html.Label("Seleccione un modelo:"),
                dcc.Dropdown(
                    id='modelo-dropdown',
                    options=[
                        {'label': 'Modelo 1: Todas las variables', 'value': 1},
                        {'label': 'Modelo 2: Factores Generales y Estado', 'value': 2},
                        {'label': 'Modelo 3: Factores Generales y Amenities', 'value': 3},
                        {'label': 'Modelo 4: Plataforma/Source y Tipo de propiedad', 'value': 4},
                        {'label': 'Modelo 5: Mascotas y Seguridad', 'value': 5},
                        {'label': 'Modelo 6: Regresión sencilla', 'value': 6}
                    ],
                    value=1,
                    style={'marginBottom': '20px'}
                ),
                html.Label("Seleccione el tipo de modelo estadístico:"),
                dcc.RadioItems(
                    id='tipo-modelo-radio',
                    options=[
                        {'label': 'Regresión Lineal', 'value': 'lineal'},
                        {'label': 'K Vecinos', 'value': 'kvecinos'},
                        {'label': 'SVR', 'value': 'svr'}
                    ],
                    value='lineal',
                    style={'marginBottom': '20px'}
                ),
                html.Div(id='model-performance', className='mt-4')
            ])
    else:
        return html.Div([
            html.H4("Contenido de la pestaña seleccionada")
        ])
    
##Callback Tab 2
@app.callback(
    [Output("box-options", "style"),
     #Output("scatter-options", "style"),
     Output("corr-info", "style")],
    Input("graph-type", "value")
)
def update_dropdown_visibility(graph_type):
    if graph_type == 'box':
        return ({'marginTop': '10px', 'display': 'block'},
                {'marginTop': '10px', 'display': 'none'},
                {'marginTop': '10px', 'display': 'none'})
    # elif graph_type == 'scatter':
    #     return ({'marginTop': '10px', 'display': 'none'},
    #             {'marginTop': '10px', 'display': 'block'},
    #             {'marginTop': '10px', 'display': 'none'})
    elif graph_type == 'corr':
        return ({'marginTop': '10px', 'display': 'none'},
                {'marginTop': '10px', 'display': 'none'},
                {'marginTop': '10px', 'display': 'block'})
    else:
        return ({'marginTop': '10px', 'display': 'none'},
                {'marginTop': '10px', 'display': 'none'},
                {'marginTop': '10px', 'display': 'none'})

# =============================================================================
# Callback para generar el gráfico según las opciones seleccionadas
# =============================================================================
@app.callback(
    Output('graph-output', 'figure'),
    [Input('graph-type', 'value'),
     Input('box-category', 'value'),
     #Input('scatter-x', 'value')
     ]
)
#def update_graph(graph_type, box_category, scatter_x):
def update_graph(graph_type, box_category):
    if graph_type == 'box':
        if not box_category:
            box_category = 'home'
        fig = px.box(df, x=box_category, y='price', title=f"Box Plot: price vs {box_category}")
    # elif graph_type == 'scatter':
    #     if not scatter_x:
    #         scatter_x = 'bedrooms'
    #     fig = px.scatter(df, x=scatter_x, y='price', title=f"Scatter Plot: price vs {scatter_x}",color =scatter_x)
    elif graph_type == 'corr':
        
        corr_df = df[['price', 'bathrooms', 'bedrooms', 'square_feet']].corr()
        # Crear heatmap con seaborn
        plt.figure(figsize=(6, 4))
        ax = sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar=True)

        # Convertir a imagen Plotly
        fig = ff.create_annotated_heatmap(
            z=corr_df.values,
            x=list(corr_df.columns),
            y=list(corr_df.index),
            annotation_text=corr_df.round(2).values,
            colorscale="RdBu",
            showscale=True
        )
        fig.update_layout(title="Correlation Plot")
    else:
        fig = {}
    return fig

## Callbat Tab 3
# Callbacks para generar los inputs dinámicos
# =============================================================================
@app.callback(
    Output('variable-inputs', 'children'),
    [Input('variable-dropdown', 'value')]
)
def update_variable_inputs(selected_vars):
    if not selected_vars:
        return html.Div("Seleccione al menos una variable", style={'color': '#d32f2f', 'fontSize': '16px'})

    inputs = []
    for var in selected_vars:
        inputs.append(
            html.Div([
                html.Label(var.replace('_', ' ').title(), style={'fontWeight': 'bold', 'color': '#1565C0'}),
                dcc.Input(
                    id={'type': 'dynamic-input', 'index': var},  # ID dinámico
                    type='number',
                    placeholder=f"Ingrese valor para {var}",
                    value=0,  # Inicializar con un valor por defecto
                    style={
                        'width': '100%', 'padding': '10px', 'marginBottom': '10px',
                        'border': '1px solid #1565C0', 'borderRadius': '8px', 
                        'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'
                    }
                )
            ], style={'marginBottom': '15px'})
        )
    return inputs
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('variable-dropdown', 'value'),
    State({'type': 'dynamic-input', 'index': ALL}, 'value')  # 🔹 Aquí está la clave
)
def pobar_modelos(n_clicks, model_type, selected_variables, input_values):
    if n_clicks == 0:
        return ""

    if not selected_variables or not input_values:
        return html.Div("⚠️ Ingrese valores antes de predecir.", style={'color': 'red', 'fontSize': '16px'})

    # Convertir valores ingresados a DataFrame
    input_dict = {var: [val] for var, val in zip(selected_variables, input_values)}
    list_pred = [i for i,j in zip(selected_variables, input_values) ]
    prediccion = pd.DataFrame(input_dict)
    # Datos de entrenamiento
    Y_var = "price"
    X_var = selected_variables
    X_inf = df[X_var]
    Y_inf = df[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X_inf, Y_inf, random_state=1)

    if model_type == "Lineal":
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        nuevaPred = model.predict(list_pred.insert(0,1))
        return html.Div([
            html.H3(f"El precio estimado de alquiler es: ${nuevaPred[0]:,.2f} USD"),
            html.H3(f"Coeficiente de determinación (R²): {r2:.2f}")
        ])
    
    elif model_type == "K":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        prediccion_scaled = scaler.transform(prediccion)
        nuevaPred = knn.predict(prediccion_scaled)
        r2 = r2_score(y_test, y_pred)
        return html.Div([
            html.H3(f"El precio estimado de alquiler es: ${nuevaPred[0]:,.2f} USD"),
            html.H3(f"Coeficiente de determinación (R²): {r2:.2f}")
        ])

    elif model_type == "SVR":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
        svr.fit(X_train_scaled, y_train)
        y_pred = svr.predict(X_test_scaled)
        prediccion_scaled = scaler.transform(prediccion)
        nuevaPred = svr.predict(prediccion_scaled)
        r2 = r2_score(y_test, y_pred)
        return html.Div([
            html.H3(f"El precio estimado de alquiler es: ${nuevaPred[0]:,.2f} USD"),
            html.H3(f"Coeficiente de determinación (R²): {r2:.2f}")
        ])

    return ""

## Callback Tab 4
@app.callback(
    Output('model-performance', 'children'),
    [Input('modelo-dropdown', 'value'),
     Input('tipo-modelo-radio', 'value')]
)
def update_model_performance(selected_model, tipo_modelo):
    if tipo_modelo == 'lineal':
        res = Resultados_Regresion[selected_model]
        metrics_table = html.Table([
            html.Tr([html.Td("Precisión (R²):"), html.Td(f"{res['R2']*100:.0f}%")]),
            html.Tr([html.Td("R² ajustado:"), html.Td(f"{res['R2aj']*100:.0f}%")]),
            html.Tr([html.Td("AIC:"), html.Td(f"{res['AIC']}")]),
            html.Tr([html.Td("BIC:"), html.Td(f"{res['BIC']}")])
        ], style={'width': '50%', 'border': '1px solid #ccc', 'borderCollapse': 'collapse', 'marginBottom': '20px'})
        
        significant_coefs = {var: res['Coeficientes'][var] 
                            for var in res['Coeficientes'].index 
                            if var in res['pvalue'] and res['pvalue'][var] < 0.05}
        if significant_coefs:
            coef_rows = [html.Tr([
                                html.Td(var, style={'border': '1px solid #ccc', 'padding': '5px'}),
                                html.Td(f"{val}", style={'border': '1px solid #ccc', 'padding': '5px'})
                            ]) for var, val in significant_coefs.items()]
            coef_table = html.Table(
                [html.Tr([html.Th("Coeficiente"), html.Th("Valor")], style={'border': '1px solid #ccc', 'padding': '5px'})] +
                coef_rows,
                style={'width': '50%', 'border': '1px solid #ccc', 'borderCollapse': 'collapse'}
            )
        else:
            coef_table = html.Div("No se encontraron coeficientes significativos.", style={'marginTop': '10px'})

        return html.Div([
            html.H4("Resumen del Modelo Lineal", style={'color': '#1565C0'}),
            html.P(f"Variables utilizadas: {res['Variables']}"),
            metrics_table,
            html.H5("Coeficientes significativos (p < 0.05):", style={'marginTop': '20px'}),
            coef_table
        ], style={'fontSize': '16px', 'lineHeight': '1.5'})
    
    elif tipo_modelo == 'kvecinos':
        res = Resultados_K[selected_model]
        return html.Div([
            html.H4("Resumen del Modelo K Vecinos", style={'color': '#1565C0'}),
            html.P(f"Variables utilizadas: {res['Variables']}"),
            html.Table([
                html.Tr([html.Td("Precisión (R²):"), html.Td(f"{res['R2']*100:.0f}%")])
            ], style={'width': '30%', 'border': '1px solid #ccc', 'borderCollapse': 'collapse'})
        ], style={'fontSize': '16px', 'lineHeight': '1.5'})
    
    elif tipo_modelo == 'svr':
        res = Resultados_SVR[selected_model]
        return html.Div([
            html.H4("Resumen del Modelo SVR", style={'color': '#1565C0'}),
            html.P(f"Variables utilizadas: {res['Variables']}"),
            html.Table([
                html.Tr([html.Td("Precisión (R²):"), html.Td(f"{res['R2']*100:.0f}%")])
            ], style={'width': '30%', 'border': '1px solid #ccc', 'borderCollapse': 'collapse'})
        ], style={'fontSize': '16px','lineHeight': '1.5'})

# @app.callback(
#     Output('prediction-output', 'children'),
#     Input('predict-btn', 'n_clicks'),
#     [Input('bathrooms', 'value'), Input('bedrooms', 'value'), Input('square_feet', 'value')]
# )
# def predict(n_clicks, bathrooms, bedrooms, square_feet):
#     if n_clicks:
#         return html.H4(f"El precio estimado de alquiler es: $X,XXX por mes.")
#     return ""

if __name__ == '__main__':
    app.run_server(debug=True)
