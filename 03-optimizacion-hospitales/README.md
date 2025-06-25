# 🏥 Optimización de Gestión Hospitalaria

## 🎯 Objetivo
Desarrollar modelos predictivos avanzados para optimizar la duración de estancias hospitalarias mediante análisis comparativo entre metodologías estadísticas (GLM vs GAM), identificando factores clave en 2,100 casos clínicos.

## 📂 Archivos Principales
- **[💾 Descripción del Dataset](./data/README.md)**
- **[🔍 Análisis Completo (HTML)](./code/analisis_hospitalario.html)**
- **[💼 Memoria Ejecutiva](./output/memoria_ejecutiva.md)**

## 💡 Principales Hallazgos
- ✅ **60.7% variación explicada** con modelo GAM optimizado (B-splines)
- ✅ **Relación no lineal** edad-estancia validada estadísticamente  
- ✅ **Factores críticos identificados** - Diagnóstico (+144% infartos), Sexo (-30% hombres)
- ✅ **Efectos hospitalarios** - Variabilidad significativa entre centros (15-25%)

## 🛠️ Tecnologías
`R` • `GAM (mgcv)` • `GLM/GLMER` • `Splines` • `Efectos Aleatorios`

## 🔗 Aplicaciones Potenciales
- **Sector Sanitario:** Predicción de estancias para planificación de recursos
- **Gestión Hospitalaria:** Optimización de ocupación de camas
- **Sistemas de Salud:** Benchmarking entre centros hospitalarios
- **Investigación Clínica:** Framework replicable para análisis longitudinales
