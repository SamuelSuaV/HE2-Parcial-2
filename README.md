# Neural Network Implementation for Income Prediction

Neural network implementation for income prediction (>$50K) using Adult Census dataset. Features data preprocessing, hyperparameter optimization, regularization techniques, and baseline comparison with logistic regression. Built with PyTorch.

## 👥 Autores
- **Samuel Suárez**
- **Jade Manon Nicolas**

**Curso**: Inteligencia Artificial Aplicada a la Economía - Parcial 2

## 📖 Descripción del Proyecto

Este proyecto desarrolla modelos de machine learning para predecir si una persona gana más de $50,000 anuales basándose en características demográficas y socioeconómicas del dataset Adult Census Income del repositorio UCI Machine Learning.

### 🎯 Objetivos Principales
- Implementar una red neuronal (MLP) para clasificación binaria de ingresos
- Comparar rendimiento entre modelo baseline (regresión logística) y redes neuronales
- Aplicar técnicas de regularización: Dropout y Early Stopping
- Realizar experimentación sistemática con hiperparámetros
- Evaluar y prevenir overfitting/underfitting

## 📊 Dataset

**Fuente**: [UCI Adult Census Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)

- **Año**: 1994 (Censo estadounidense)
- **Registros de entrenamiento**: 32,561
- **Registros de prueba**: 16,281
- **Características**: 14 variables predictoras
- **Variable objetivo**: Binaria (≤50K / >50K)

### Variables del Dataset
| Tipo | Variables |
|------|-----------|
| **Demográficas** | age, sex, race, native-country |
| **Educativas** | education, education-num |
| **Laborales** | workclass, occupation, hours-per-week |
| **Financieras** | capital-gain, capital-loss |
| **Familiares** | marital-status, relationship |

## 🔬 Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones y valores faltantes
- Identificación de desbalances de clase
- Detección de variables correlacionadas

### 2. Procesamiento de Datos

#### Manejo de Valores Faltantes
- **Porcentaje**: Solo 0.92% de valores faltantes
- **Variables afectadas**: workclass, occupation, native-country
- **Estrategia**: Imputación por moda (valor más frecuente)
- **Prevención de data leakage**: Uso de estadísticas del conjunto de entrenamiento

#### Transformaciones de Variables
| Transformación | Justificación |
|----------------|---------------|
| `capital-gain` + `capital-loss` → `capital-net` | Reducir sparsity y combinar información relacionada |
| `native-country` → `US/Non-US` | Prevenir overfitting por categorías con pocas muestras |
| One-hot encoding | Manejo apropiado de variables categóricas |
| Estandarización | Normalización de variables continuas |

### 3. Arquitecturas Implementadas

#### Modelo Baseline
- **Algoritmo**: Regresión Logística
- **Propósito**: Establecer línea base de comparación

#### Red Neuronal (MLP)
- **Framework**: PyTorch
- **Arquitectura**: Multilayer Perceptron
- **Función de pérdida**: Binary Cross Entropy
- **Optimizador**: Adam

## 🧪 Experimentación

### Experimentos sin Regularización (5 configuraciones)

| Experimento | Variable | Configuraciones Probadas |
|-------------|----------|-------------------------|
| 1 | Neuronas por capa | 128, 256, 512 |
| 2 | Número de capas | 3, 6, 12 |
| 3 | Batch size | 512, 1024, 2048 |
| 4 | Learning rate | 0.0005, 0.001, 0.002 |
| 5 | Función activación | ReLU, LeakyReLU, ELU |

**Mejor configuración sin regularización**:
- 3 capas ocultas, 128 neuronas por capa
- Batch size: 2048, Learning rate: 0.0005
- Función de activación: ELU

### Experimentos con Regularización

#### Técnicas Implementadas
- **Dropout**: Prevención de overfitting durante entrenamiento
- **Early Stopping**: Detención automática basada en pérdida de validación

**Configuración final optimizada**:
- **Capas**: 8 capas ocultas
- **Neuronas por capa**: 256
- **Batch size**: 1024
- **Learning rate**: 0.001
- **Activación**: ELU
- **Regularización**: Dropout + Early Stopping

## 📈 Resultados

### Métricas de Evaluación
- **Accuracy**: Precisión general del modelo
- **Precision**: Exactitud en predicciones positivas
- **Recall**: Capacidad de detectar casos positivos
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC


*Nota: Métricas completas disponibles en el notebook y en el reporte de trabajo*

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **PyTorch**: Implementación de redes neuronales
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **Scikit-learn**: Métricas y modelo baseline
- **Matplotlib/Seaborn**: Visualización
- **Google Colab**: Entorno de desarrollo

## 📁 Estructura del Repositorio

```
├── README.md
├── notebook/
│   └── Income_Prediction_Neural_Networks.ipynb
├── reports/
│   ├── Reporte_Parcial_2_HE2-2.pdf
│   └── HE2_Inteligencia_Artificial_Aplicada.pdf
├── data/
    ├── adult.data
    └── adult.test

```

## 🚀 Instrucciones de Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/SamuelSuaV/income-prediction-neural-networks.git
cd income-prediction-neural-networks
```

### 2. Descargar datos
Descargar repositorio y subir a Google Drive.

### 3. Ejecutar notebook
```bash
jupyter notebook notebook/Income_Prediction_Neural_Networks.ipynb
```
Debe ejecutarse desde Drive.

## 🔍 Hallazgos Clave

### Procesamiento de Datos
- La combinación de variables financieras (`capital-net`) mejora la representación
- La simplificación geográfica previene overfitting
- La imputación por moda es efectiva con pocos valores faltantes

### Arquitectura de Red
- **ELU** supera a ReLU y LeakyReLU evitando "muerte de neuronas"
- Redes más profundas (8 capas) mejoran el log-loss sin sacrificar F1-score significativamente
- El equilibrio entre batch size y learning rate es crucial

### Regularización
- **Early Stopping** previene efectivamente el overfitting
- **Dropout** mejora la generalización
- La regularización permite arquitecturas más complejas sin degradación

## 📊 Visualizaciones Incluidas

- Distribuciones de variables categóricas y continuas
- Curvas de pérdida: entrenamiento vs validación
- Análisis de overfitting/underfitting por experimento
- Comparación de funciones de activación

## 📚 Referencias

- Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems* (2nd ed.). O'Reilly.
- UCI Machine Learning Repository: Adult Data Set


---

**Nota**: Este notebook debe ejecutarse de inicio a fin para reproducir los resultados reportados. Cualquier error en la ejecución puede afectar la reproducibilidad de los resultados.
