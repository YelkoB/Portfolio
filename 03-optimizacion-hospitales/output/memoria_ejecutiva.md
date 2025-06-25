# 💼 Memoria Ejecutiva: Optimización Hospitalaria

> **Objetivo**: Desarrollar modelos predictivos para optimizar estancias hospitalarias mediante análisis comparativo de metodologías estadísticas avanzadas.  
> **Muestra**: 2,100 casos analizados con GAM vs GLM vs efectos aleatorios.

---

## 📈 Resultados del Modelo Óptimo

### Efectos No Lineales Identificados
![Efectos Suavizados del Modelo](https://github.com/user-attachments/assets/588bff63-698a-4588-babb-ed1ad0beb27e)
*Relación no lineal entre edad y duración hospitalaria. El patrón en U invertida revela que pacientes de mediana edad requieren estancias más prolongadas.*

### Predicciones por Perfil Clínico
![Predicciones por Variables](https://github.com/user-attachments/assets/56e092f2-7253-4d0d-8d2a-ce72ed0833a7)
*Diferencias sustanciales entre diagnósticos y sexos. Los infartos generan las estancias más prolongadas, seguidos por neumonías. Las mujeres consistentemente requieren más tiempo de hospitalización.*

---

## 🔍 Factores Predictivos Clave

### 👤 **Características del Paciente**

#### Efecto de la Edad (No Lineal)
**🌀 Patrón complejo:** Relación no lineal validada (p < 0.001)  
**💡 Interpretación:** Jóvenes y muy mayores tienen duraciones menores  

#### Diferencias por Sexo  
**👨 Hombres:** **30% menos** tiempo de hospitalización vs mujeres  
**📏 Impacto:** Reducción promedio de 3-5 días de estancia  

### 🏥 **Factores Clínicos - Diagnóstico Principal**

| Patología | vs Fractura | Multiplicador | Días Adicionales |
|-----------|-------------|---------------|------------------|
| **🔴 Infarto** | +144% | 2.44x más largo | +15-20 días |
| **🟡 Neumonía** | +88% | 1.88x más largo | +8-12 días |
| **🟢 Fractura** | Referencia | Base | 2-5 días |

#### Efectos Hospitalarios
**🏥 Variabilidad significativa:** 8.48 edf entre centros (p < 0.001)  
**📏 Diferencias:** 15-25% variación en duraciones entre hospitales  

---

## 🏆 Rendimiento del Modelo Final

### Capacidad Predictiva
** Deviance Explicada: 60.7%** - Captura más del 60% de factores determinantes  
** R² Ajustado: 36.2%** - Nivel robusto para datos clínicos complejos  
** Escala: 0.915** - Ajuste óptimo de distribución Gamma

---

## 🔮 **Sistema Predictivo**
- Mujer, 75 años, Neumonía → **91 días** predichos (IC: 75-107 días)
- Hombre, 65 años, Infarto → **45 días** predichos (IC: 38-53 días)  
- Hombre, 30 años, Fractura → **0.1 días** predichos (alta rápida)
- Mujer, 45 años, Fractura → **2.1 días** predichos (estancia corta)

---

## 📂 Documentación Técnica

- 💾 **[Descripción de Datos](./data/README.md)**
- 🔍 **[Análisis Completo (HTML)](./code/analisis_hospitalario.html)**

---
