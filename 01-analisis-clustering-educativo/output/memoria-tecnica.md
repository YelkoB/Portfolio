
# 📋 Memoria Ejecutiva: Segmentación de Estudiantes

> **Objetivo:** Identificar patrones de comportamiento estudiantil mediante clustering automático para clasificación en perfiles diferenciados.

---

## 🎯 Resultados Principales

| **Métrica** | **Valor** |
|-------------|-----------|
| **Grupos identificados** | 3 perfiles diferenciados |
| **Precisión del modelo** | Correlación cofenética: 0.57 |
| **Varianza explicada** | 68% (primeras 2 componentes) |
| **Muestra analizada** | 100 estudiantes, 9 variables |

---

## 📊 Perfiles Identificados

### Características por Grupo
![Perfil de Características por Grupo](https://github.com/user-attachments/assets/4be0c515-7bce-44d8-a411-d07498bfacba)
*Características promedio por grupo (valores estandarizados). La línea horizontal en 0 representa la media poblacional.*

### Segmentación Final
![Clustering PCA Final](https://github.com/user-attachments/assets/4be0c515-7bce-44d8-a411-d07498bfacba)
*Distribución de los 3 grupos en el espacio de componentes principales. Separación clara entre perfiles.*

### 🔴 **Grupo 1: Bajo Rendimiento** (33%)
- Rendimiento académico por debajo de la media en todas las áreas
- Hábitos deficientes: menos estudio, más dispositivos, peor asistencia
- **Estrategia:** Intervención prioritaria

### 🔵 **Grupo 2: Rendimiento Medio** (42%)
- Rendimiento estándar con patrones compensatorios
- Equilibrio variable entre aspectos académicos y personales
- **Estrategia:** Mejora focalizada por áreas

### 🟢 **Grupo 3: Alto Rendimiento** (25%)
- Excelencia académica consistente
- Buenos hábitos de estudio y bienestar físico
- **Estrategia:** Modelos de referencia

---

## 🔍 Metodología

**Técnica:** Clustering jerárquico con método Average tras eliminación de outliers
**Validación:** Análisis de Componentes Principales (PCA)
**Interpretación:** 
- **CP1 (Eje X):** Rendimiento académico y hábitos de estudio
- **CP2 (Eje Y):** Balance condición física - nivel de estrés

---

## 💼 Aplicaciones Empresariales

- **Segmentación de clientes:** Metodología transferible a comportamiento de consumo
- **Recursos Humanos:** Clasificación de perfiles de empleados
- **Marketing:** Estrategias diferenciadas por grupo de comportamiento
- **Educación:** Programas personalizados por perfil estudiantil

---

## 🔧 Criterios de Clasificación

| **Factor** | **Grupo 1** | **Grupo 2** | **Grupo 3** |
|------------|-------------|-------------|-------------|
| **CP1 (Rendimiento)** | < -0.5 | -0.5 a 0.5 | > 0.5 |
| **Posición PCA** | Izquierda | Centro | Derecha |
| **Patrón** | Consistentemente bajo | Variable | Consistentemente alto |

---

## 📚 Documentación Técnica

- 🔍 **[Análisis Completo (HTML)](./code/analisis_clustering.html)**
- 📄 **[Reporte Técnico (PDF)](./code/analisis-clustering.pdf)**
- 💾 **[Descripción de Datos](./data/README.md)**
- 🗂️ **[Código Fuente](./code/)**

---

> **Conclusión:** Sistema automático que clasifica individuos en 3 grupos homogéneos basándose en patrones de comportamiento, con aplicabilidad directa en segmentación empresarial.
