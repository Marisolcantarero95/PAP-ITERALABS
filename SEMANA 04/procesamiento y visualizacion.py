#!/usr/bin/env python
# coding: utf-8

# SERIES DE TIEMPO e HISTOGRAMAS PARA FLUJOS_E_MIN

# Importando paqueterias

# In[259]:


import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import scipy.stats as st
import statsmodels.datasets
import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
pd.set_option('precision',2)
get_ipython().run_line_magic('matplotlib', 'inline')


# Importar pestaña Flujos_E_min de datos en excel

# In[260]:


data=pd.read_excel(r'C:\Users\luism\OneDrive\Escritorio\data\datos_ejemplo_1.xlsx', sheet_name='Flujos_E_Min', header=0,index_col =0)
Flujos_E_min=data.transpose()


# Acomodar archivo

# Se eliminarán las variables egresos_pub,egresos_rh y egresos_servicios ya que son gastos constantes

# In[261]:


Flujos_E_min=Flujos_E_min.drop(['egresos_pub', 'egresos_rh', 'egresos_servicios'], axis=1)
Flujos_E_min.head()


# Se realiza histograma para saber como se distribuyen los datos

# In[262]:


x_result = Flujos_E_min['ingresos_totales']
data = [go.Histogram(x=x_result)]

iplot(data, filename='basic histogram')


# Graficar serie de tiempo

# In[263]:


data1 = [go.Scatter(y=Flujos_E_min['ingresos_totales'] )]

iplot(data1, filename='pandas-time-series')


# Determinar si el modelo es estacionario o no

# In[264]:


def test_stationarity(timeseries):
#Estaditicas para la determinación de balanceo(media_movil y desviación estandar)
    rolmean=Flujos_E_min['ingresos_totales'].rolling(window=12,center=False).mean()
    rolstd=Flujos_E_min['ingresos_totales'].rolling(window=12,center=False).std()
#graficar las estadisticas de balanceo
    orig=plt.plot(timeseries,color='blue',label='Original')
    mean=plt.plot(rolmean,color='red',label='Media_movil')
    std=plt.plot(rolstd,color='black',label='Desviación estandar')
    plt.legend(loc='best')
    plt.title('Media movil y Desviación estandar')
    plt.show(block=False)
    test_stationarity(Flujos_E_min['ingresos_totales'])


# Calcular media Movil

# In[285]:


moving_avg=Flujos_E_min['ingresos_totales'].rolling(10,win_type ='triang',center=False).mean()
Flujos_E_min['ingresos_totales'].plot()
#Gráfica de la media movil
moving_avg.plot(label='Media movil',color='red')
plt.title('Media movil')
plt.legend(loc='best')


# Calculo de la varianza

# In[286]:


variacion_mensual=Flujos_E_min['ingresos_totales']/Flujos_E_min['ingresos_totales'].shift(1)-1
Flujos_E_min['var diaria']=variacion_mensual
Flujos_E_min['var diaria'][:20]
#Grafica de los datos
plot = Flujos_E_min['var diaria'].plot()
plt.title('varianza mensual')


# Calculo de la linea de tendecia de los datos aplicando el filtro Prescott

# In[287]:


ciclo,tend=sm.tsa.filters.hpfilter(Flujos_E_min['ingresos_totales'])
Flujos_E_min['Tendencia']=tend
Flujos_E_min[['ingresos_totales','Tendencia']].plot(fontsize=12)
plt.title('Tendencia')
legend=plt.legend()
legend.prop.set_size(14);


#  

# In[288]:


decompocision=sm.tsa.seasonal_decompose(Flujos_E_min['ingresos_totales'],freq=30)

trend=decompocision.trend
seasonal=decompocision.seasonal
plt.subplot(411)
plt.plot(Flujos_E_min['ingresos_totales'],label='original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Fluctuaciones')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Variaciones estacionales')
plt.legend(loc='best')
plt.tight_layout()


# In[289]:


decompocision=sm.tsa.seasonal_decompose(Flujos_E_min['ingresos_totales'],freq=30)


# In[ ]:





# In[ ]:




