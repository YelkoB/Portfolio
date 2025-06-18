# 💼 Memoria Ejecutiva: Segmentación de Estudiantes

> 🎯 **Objetivo:** Identificar patrones de comportamiento estudiantil mediante clustering jerárquico para clasificación automática en perfiles diferenciados.

---

## 📊 Resultados Principales

| **Métrica** | **Valor** |
|-------------|-----------|
| **Grupos identificados** | 3 perfiles claramente diferenciados |
| **Método seleccionado** | Average (correlación cofenética: 0.5655) |
| **Outliers detectados** | 2 estudiantes (IDs 72 y 79) |
| **Varianza explicada** | 68% (primeras 2 componentes principales) |

---

## 📊 Perfiles Identificados

### Características por Grupo
![Perfil de Características por Grupo](https://github.com/user-attachments/assets/4be0c515-7bce-44d8-a411-d07498bfacba)
*Características promedio por grupo (valores estandarizados). La línea horizontal en 0 representa la media poblacional.*

### Segmentación Final
![Clustering PCA Final]([./output/visualizations/clustering-pca-final.png](https://github.com/user-attachments/assets/4be0c515-7bce-44d8-a411-d07498bfacba))
*Distribución de los 3 grupos en el espacio de componentes principales. Separación clara entre perfiles.*

### 🔴 **Grupo 1: Estudiantes con Bajo Rendimiento** (33%)

**📉 Rendimiento académico:** Por debajo de la media en todas las asignaturas  
**📚 Hábitos de estudio:** Menor dedicación en horas de estudio y asistencia, mayor uso de dispositivos electrónicos, menos horas de sueño  
**🎯 Patrón identificativo:** Valores negativos consistentes en CP1 (rendimiento académico)  
**📊 Dispersión:** Comportamiento variable en CP2 (aspectos físicos/emocionales)

### 🔵 **Grupo 2: Estudiantes con Rendimiento Medio** (42%)

**📊 Rendimiento académico:** Valores medios o ligeramente por debajo de la media  
**⚖️ Patrón compensatorio:** Si tienen valores altos en rendimiento, presentan patrones variables en aspectos físicos/emocionales  
**🎯 Patrón identificativo:** Posición central en CP1, tendencia ligeramente negativa en CP2  
**🔄 Equilibrio inestable:** Compensación entre diferentes aspectos del desarrollo estudiantil

### 🟢 **Grupo 3: Estudiantes de Alto Rendimiento Integral** (25%)

**🌟 Rendimiento académico:** Bastante por encima de la media en todas las áreas  
**💪 Perfil complejo:** Buen rendimiento académico con patrones diversos en condición física y estrés  
**🎯 Patrón identificativo:** Valores positivos en CP1 con tendencia positiva en CP2  
**🏆 Excelencia académica:** Destacan principalmente en rendimiento académico

---

## 🔍 Metodología y Hallazgos

**✅ Eliminación exitosa de outliers:** Los estudiantes 72 y 79 (valores extremos en Condición Física) fueron identificados y eliminados, mejorando la calidad del clustering

**✅ Método óptimo identificado:** El método Average con correlación cofenética de 0.5655 resultó ser el más adecuado tras la eliminación de outliers

**✅ Componentes principales interpretables:**
- **CP1:** Rendimiento académico general y hábitos estudiantiles y de bienestar
- **CP2:** Combinación compleja de condición física y nivel de estrés

---

## 🔧 Criterios de Clasificación

| **Criterio** | **Grupo 1** | **Grupo 2** | **Grupo 3** |
|-------------|-------------|-------------|-------------|
| **Rendimiento Académico/Hábitos (CP1)** | < -0.5 (Bajo) | -0.5 a 0.5 (Medio) | > 0.5 (Alto) |
| **Condición Física/Estrés (CP2)** | Variable | Tendencia ligeramente negativa | Tendencia positiva |
| **Posición en Gráfico PCA** | Lado izquierdo | Centro | Lado derecho |

### 📋 Regla de Clasificación Simplificada

1. **Calcular CP1** (combinación de promedios académicos y hábitos de estudio y bienestar)
2. **Evaluar CP2** (equilibrio entre condición física y nivel de estrés)  
3. **Aplicar umbrales** en el espacio bidimensional PCA
4. **Asignar grupo** según la posición en el gráfico de componentes principales

---

## 💼 Aplicaciones Empresariales

**🎓 Sector Educativo:** Identificación automática de perfiles estudiantiles para programas personalizados  
**👥 Recursos Humanos:** Clasificación de candidatos por patrones de comportamiento y rendimiento  
**📊 Segmentación de Mercado:** Metodología transferible para identificar grupos homogéneos de clientes  
**🔍 Análisis de Datos:** Sistema escalable para segmentación automática de poblaciones

---

## 📚 Documentación Técnica

- 🔍 **[Análisis Completo (HTML)](./code/analisis_clustering.html)**
- 📄 **[Reporte Técnico (PDF)](./code/analisis-clustering.pdf)**
- 💾 **[Descripción de Datos](./data/README.md)**
- 🗂️ **[Código Fuente](./code/)**

---

> **Conclusión:** El análisis de clustering identificó tres grupos distintos de estudiantes con características diferenciadas en términos de rendimiento académico y patrones físicos/emocionales, con metodología aplicable a segmentación empresarial.
