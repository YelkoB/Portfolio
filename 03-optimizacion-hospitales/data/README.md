# 💾 Descripción del Dataset: Estancias Hospitalarias

## 📈 Información General
- **Tamaño**: 2,100 casos de hospitalización
- **Período**: Datos retrospectivos de múltiples centros
- **Población**: Pacientes con tres patologías principales
- **Centros**: 10 hospitales diferentes
- **Tipo**: Datos clínicos anonimizados

## 📋 Variables del Dataset

### 🎯 **Variable Objetivo**
- **DuracionHospitalizacion:** Numérico → horas de estancia hospitalaria
- **Rango**: 1-20 horas aproximadamente
- **Distribución**: Positiva asimétrica (Gamma)
- **Media**: 7-8 horas de estancia promedio

### 👤 **Características del Paciente**
- **Edad:** Numérico → Edad del paciente en años
- **Sexo:** Categórico → Femenino (F) / Masculino (M)

### 🏥 **Información Clínica**
- **Diagnostico:** Categórico → Tres categorías principales:
  - **Fractura:** Lesiones óseas y traumatológicas
  - **Infarto:** Eventos cardiovasculares agudos  
  - **Neumonia:** Infecciones respiratorias graves

### 🏢 **Factor Institucional**
- **HospitalID:** Categórico → Identificador del centro (1-10)
- **Variabilidad**: Diferencias en protocolos y recursos entre centros

## 📊 **Distribución de Casos**

### Por Diagnóstico
- **Fracturas**: ~35% de los casos
- **Infartos**: ~30% de los casos  
- **Neumonías**: ~35% de los casos

### Por Características
- **Edad**: Distribución amplia con concentración en mayores
- **Sexo**: Distribución equilibrada entre géneros
- **Hospitales**: Representación variable según capacidad del centro

## 🔍 **Calidad de Datos**
- **Completitud**: 100% - Sin valores faltantes
- **Consistencia**: Validación clínica de rangos y relaciones
- **Anonimización**: Cumple normativas de protección de datos sanitarios

---
*Dataset clínico para investigación en gestión hospitalaria y modelización predictiva.*
