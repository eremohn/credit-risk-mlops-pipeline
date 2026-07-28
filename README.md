# mlops_pipeline

Pipeline de Machine Learning en producción para la predicción del
comportamiento de nuevos usuarios a partir del histórico de créditos.
Proyecto integrador — Módulo 5: Fundamentos de Nube y Ciencia de Datos en
Producción.

## 1. Contexto de negocio

La empresa financiera requiere anticipar el comportamiento de nuevos
usuarios (riesgo de default) utilizando su información histórica de
créditos. El presente repositorio contiene el pipeline completo, desde la
carga de datos hasta el monitoreo del modelo en producción, siguiendo
prácticas de MLOps y respetando la estructura fija exigida por los
pipelines automatizados de Jenkins.

## 2. Estructura del repositorio

```
mlops_pipeline/
│
├── src/
│   ├── cargar_datos.ipynb              # Carga y validación inicial de datos
│   ├── comprension_eda.ipynb           # Análisis exploratorio (EDA)
│   ├── ft_engineering.py               # Ingeniería de características
│   ├── model_training_evaluation.py    # Entrenamiento y evaluación del modelo
│   ├── model_deploy.py                 # Despliegue del modelo
│   └── model_monitoring.py             # Monitoreo en producción
│
├── base_de_datos.xlsx                  # Fuente de datos histórica
├── requirements.txt                    # Dependencias del proyecto
├── .gitignore
└── README.md
```

**Importante:** esta estructura es obligatoria y no debe modificarse, ya
que los pipelines de Jenkins dependen de estas rutas exactas.

## 3. Estado actual del proyecto

| Componente | Estado | Versión objetivo |
|---|---|---|
| Estructura del repositorio | ✅ Completo | V1.0.0 |
| `cargar_datos.ipynb` (funcional) | ⏳ Pendiente | V1.0.1 |
| `comprension_eda.ipynb` (funcional) | ⏳ Pendiente | V1.0.1 |
| `ft_engineering.py` (funcional) | ⏳ Pendiente | V1.1.0 |
| `model_training_evaluation.py` (funcional) | ⏳ Pendiente | V1.2.0 |
| `model_deploy.py` (funcional) | ⏳ Pendiente | V1.3.0 |
| `model_monitoring.py` (funcional) | ⏳ Pendiente | V1.4.0 |

## 4. Estrategia de ramas (GitFlow)

- `master`: código estable, listo para producción. Solo recibe merges
  desde `certification` mediante Pull Request aprobado.
- `certification`: rama de pre-producción / control de calidad, donde se
  valida el trabajo integrado antes de promocionarlo a `master`.
- `developer`: rama principal de desarrollo activo, donde se integran las
  features antes de pasar a certificación.

Flujo recomendado: `feature/*` → `developer` → `certification` → `master`.

## 5. Versionado semántico

- **V1.0.0**: creación de la estructura completa del repositorio.
- **V1.0.1**: implementación funcional de `cargar_datos.ipynb` y
  `comprension_eda.ipynb`.
- Versiones futuras (V1.1.0, V1.2.0, V1.3.0, V1.4.0): implementación
  funcional de feature engineering, entrenamiento/evaluación, despliegue y
  monitoreo, respectivamente.

Cada release relevante debe quedar marcada con un tag semántico
(`git tag -a vX.Y.Z -m "mensaje"`).

## 6. Instalación del entorno

### Windows (PowerShell / CMD)
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Git Bash (Windows)
```bash
python -m venv venv
source venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Calidad de código

El proyecto sigue el estándar **PEP8**. Se recomienda validar antes de
cada commit:

```bash
flake8 src/
black --check src/
```

## 8. Próximos pasos

Ver tabla de estado (sección 3). El siguiente hito es **V1.0.1**:
implementación funcional de `cargar_datos.ipynb` y `comprension_eda.ipynb`
sobre `base_de_datos.csv`.
