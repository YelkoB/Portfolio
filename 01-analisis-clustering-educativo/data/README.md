---
layout: default
markdown: kramdown
---

# 📊 Descripción del Dataset: Rendimiento Académico y Hábitos

## 📈 Información General
- **Tamaño**: 100 estudiantes
- **Período**: Estudio transversal
- **Población**: Estudiantes de 15-18 años en entorno urbano
- **Centros educativos**: 9 instituciones

## 📋 Variables del Dataset

### 🔢 Identificación
| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `ID` | Numérico | 1-100 | Identificador único del estudiante |
| `Edad` | Numérico | 15-18 | Edad en años |
| `Centro` | Categórico | 1-9 | Centro educativo de procedencia |

### 📚 Rendimiento Académico
| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `Promedio_matematicas` | Numérico | 0-100 | Calificación promedio en matemáticas |
| `Promedio_ciencias` | Numérico | 0-100 | Calificación promedio en ciencias |
| `Promedio_lectura` | Numérico | 0-100 | Calificación promedio en lectura |
| `Asistencia` | Porcentaje | 0-100% | Porcentaje de asistencia a clases |

### 🏃‍♂️ Hábitos y Estilo de Vida
| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `Horas_estudio` | Numérico | - | Horas de estudio semanales |
| `Horas_sueño` | Numérico | - | Promedio de horas de sueño diarias |
| `Nivel_estres` | Escala | 1-10 | Nivel de estrés autorreportado |
| `Uso_dispositivos` | Numérico | - | Horas diarias de uso de dispositivos electrónicos |
| `Condicion_fisica` | Numérico | - | Minutos de ejercicio por semana |

## 🎯 Metodología de Análisis
- **Técnica principal**: Clustering jerárquico (Ward, Average)
- **Preprocesamiento**: Estandarización de variables
- **Validación**: Correlación cofenética y análisis de componentes principales
- **Grupos identificados**: 2-3 clusters óptimos según índices estadísticos

## 📊 Principales Hallazgos
- **Grupo 1**: Alto rendimiento académico + buenos hábitos
- **Grupo 2**: Rendimiento medio con variabilidad en hábitos  
- **Grupo 3**: Bajo rendimiento académico generalizado

## 🔍 Variables Clave para Segmentación
1. **Rendimiento académico** (matemáticas, ciencias, lectura)
2. **Hábitos de estudio** (horas de estudio, asistencia)
3. **Bienestar físico/emocional** (condición física, nivel de estrés)

---
*Dataset generado con fines educativos y de investigación académica.*
