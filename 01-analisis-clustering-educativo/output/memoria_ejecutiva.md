# 💼 Memoria Ejecutiva: Segmentación de Estudiantes

> **Objetivo**: Identificar patrones de comportamiento estudiantil mediante clustering jerárquico para clasificación automática en perfiles diferenciados.  
> **Muestra**: 100 estudiantes analizados mediante clustering jerárquico.

---

## 📈 Resultados

### Características por Grupo
![Perfil de Características por Grupo](https://github.com/user-attachments/assets/4be0c515-7bce-44d8-a411-d07498bfacba)
*Características promedio por grupo (valores estandarizados). La línea horizontal en 0 representa la media poblacional.*

### Segmentación Final
![Clustering PCA Final](https://github.com/user-attachments/assets/96daf296-e208-4b12-a320-1cae7679f4f4)
*Distribución de los 3 grupos en el espacio de componentes principales donde:*
-  ***Eje X (CP1):** Rendimiento académico (derecha = mejor rendimiento).*
-  ***Eje Y (CP2):** Combinación de condición física y estrés (arriba = alta condición física + alto estrés).*

## 👥 Perfiles identificados

### 🔴 **Grupo 1: Estudiantes con Bajo Rendimiento**

**📉 Rendimiento académico:** Por debajo de la media en todas las asignaturas  
**📚 Hábitos de estudio:** Menor dedicación en horas de estudio y asistencia, mayor uso de dispositivos electrónicos, menos horas de sueño  
**🎯 Patrón identificativo:** Valores negativos consistentes en CP1 (rendimiento académico)  
**📊 Dispersión:** Comportamiento variable en CP2 (aspectos físicos/emocionales)

### 🔵 **Grupo 2: Estudiantes con Rendimiento Medio**

**📊 Rendimiento académico:** Valores medios o ligeramente por debajo de la media  
**⚖️ Patrón compensatorio:** Si tienen valores altos en rendimiento, presentan patrones variables en aspectos físicos/emocionales  
**🎯 Patrón identificativo:** Posición central en CP1, tendencia ligeramente negativa en CP2  
**🔄 Equilibrio inestable:** Compensación entre diferentes aspectos del desarrollo estudiantil

### 🟢 **Grupo 3: Estudiantes de Alto Rendimiento Integral**

**🌟 Rendimiento académico:** Bastante por encima de la media en todas las áreas  
**💪 Perfil complejo:** Buen rendimiento académico con patrones diversos en condición física y estrés  
**🎯 Patrón identificativo:** Valores positivos en CP1 con tendencia positiva en CP2  
**🏆 Excelencia académica:** Destacan principalmente en rendimiento académico

---

## 📂 Documentación Técnica

- 💾 **[Descripción de Datos](./data/README.md)**
- 🔍 **[Análisis Completo (HTML)](./code/analisis_clustering.html)**
- 📄 **[Reporte Técnico (PDF)](./code/analisis-clustering.pdf)**

---
