# 💼 Memoria Ejecutiva: Predicción de Demanda Energética  
> **Objetivo**: Desarrollar un modelo predictivo robusto para el consumo eléctrico nacional con detección automática de eventos extraordinarios y forecasting estratégico a 3 años vista.  
> **Metodología**: Análisis de series temporales sobre 12 años de datos mensuales de Austria.

---

## 📈 Resultados del Modelo

### Evolución del Consumo Eléctrico
![Evolución Temporal](https://github.com/user-attachments/assets/29c1e4b2-abfe-49cd-ac3d-39f2220e23e1)
*Consumo eléctrico mensual en Austria mostrando patrones estacionales marcados (picos invernales) y la detección automática de eventos extraordinarios como la caída del COVID-19 en abril 2020.*

### Precisión Excepcional del Modelo
- **🎯 Precisión: 98.4%** - El modelo acierta en más del 98% de sus predicciones
- **📊 Error promedio: 1.6%** - Desviación mínima respecto a valores reales
- **🔍 Capacidad predictiva:** Detecta automáticamente eventos disruptivos

---

## 🔍 Factores que Influyen en el Consumo

### 📅 **Patrones Temporales**
#### Estacionalidad Marcada
- **❄️ Invierno:** Picos de consumo por calefacción (noviembre-febrero)
- **☀️ Verano:** Consumo reducido (junio-agosto) por ser meses festivos con menor actividad industrial
- **🔄 Persistencia:** El consumo de un mes influye fuertemente en el siguiente

#### Actividad Económica
- **📈 Días laborables:** Cada día laboral adicional aumenta el consumo un **0.34%**
- **🐣 Semana Santa:** Incremento sistemático del **1.1%** durante festividades

### ⚡ **Eventos Extraordinarios Detectados Automáticamente**

#### 🦠 Crisis COVID-19 (Marzo-Abril 2020)
- **Marzo 2020:** Reducción del **3.3%** (inicio restricciones)
- **Abril 2020:** Caída drástica del **9.9%** (confinamiento total)
- **💡 Detección:** El modelo identificó automáticamente el impacto sin conocimiento previo

#### 🏢 Otros Eventos del Sector Energético
- **Febrero 2017:** **-3.1%** (crisis empresa Care Energy)
- **Abril 2018:** **-4.2%** (nueva estrategia energética gubernamental)
- **Octubre 2022:** **-3.6%** (temperaturas récord, menor demanda calefacción)

---

## 🔮 Predicciones Estratégicas (2024-2026)

### Forecasting a 3 Años Vista
![Predicciones ARIMA](https://github.com/user-attachments/assets/ea1da166-9f16-4817-849c-8b454a9f020e)
*Comparación de predicciones: ARIMA (izquierda, rojo) mantiene estacionalidad marcada con intervalos precisos, mientras que ETS Inverso (derecha, azul) presenta patrones más suavizados con mayor incertidumbre hacia el futuro. El modelo ARIMA preserva mejor los patrones históricos estacionales.*

### Tendencias Identificadas
- **📈 Crecimiento mínimo:** Incremento muy gradual
- **🔄 Patrones estacionales:** Se mantienen los picos y valles históricos

---

## 📂 Documentación Técnica
- 💾 **[Descripción de Datos](../data/README.md)**
- 🔍 **[Análisis Completo (HTML)](../code/analisis_series_temporales.html)**  

---
