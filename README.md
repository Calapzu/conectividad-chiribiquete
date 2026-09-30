# Optimización de conectividad ecológica mediante algoritmo genético frente a dIIC en la zona de amortiguación de Chiribiquete, Caquetá

Código, datos de entrada y resultados que respaldan el artículo:

> Calapzu-Palma, Y. J., Millan-Rojas, E. E., & Gallego-Ruiz, L. E. (2026). *Optimización de conectividad ecológica mediante algoritmo genético frente a dIIC en la zona de amortiguación de Chiribiquete, Caquetá*. Revista de la Academia Colombiana de Ciencias Exactas, Físicas y Naturales. **[Actualizar con volumen, páginas y DOI cuando se publique]**

**Repositorio archivado (DOI):** **[DOI de Zenodo]**

---

## 1. Descripción

El estudio compara un algoritmo genético (AG), con el Índice Integral de Conectividad (IIC) como función de aptitud, frente a la priorización por dIIC, una heurística por área (largest-first) y 1.000 construcciones aleatorias, bajo presupuestos de área del 10 %, 30 % y 50 %. El área de estudio es la zona de amortiguación de 20 km del PNN Serranía de Chiribiquete (Caquetá, Colombia).

Datos clave del análisis:

| Elemento | Valor |
|---|---|
| Fragmentos analizados | 514 (de 738 antes del filtro de 1,0 ha) |
| Enlaces de la red | 8.912 |
| Umbral de dispersión | 7.500 m |
| IIC normalizado total | 0,275873 |
| Sistema de referencia | MAGNA-SIRGAS / Origen Nacional (EPSG:9377) |
| Corridas del AG por escenario | 30 |
| Población / generaciones máximas | 200 / 500 |
| Presupuestos | 10 %, 30 % y 50 % del área total |

## 2. Estructura del repositorio

```
conectividad-chiribiquete/
├── README.md
├── LICENSE
├── requirements.txt
├── codigo/
│   └── script_maestro_final.py
├── datos_entrada/
│   ├── fragmentos_filtrados_v2.*
│   ├── centroides_fragmentos_final.*
│   ├── area_estudio_final.*
│   ├── nodos_conefor.txt
│   └── distancias_conefor_unica.txt
├── validacion_conefor/
│   ├── overall_indices.txt
│   └── node_importances_limpio.txt
└── resultados/
    ├── escenario_10/
    ├── escenario_30/
    ├── escenario_50/
    └── resumen_final_todos_escenarios.csv
```

`.*` indica que se incluyen todos los archivos del shapefile (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`).

## 3. Descripción de los archivos

### datos_entrada/

| Archivo | Contenido |
|---|---|
| `fragmentos_filtrados_v2.*` | 514 fragmentos forestales finales, con los campos `area_ha`, `perim_m`, `coord_x`, `coord_y`, `costo_ha` y `tipo_nodo` |
| `centroides_fragmentos_final.*` | Centroides de los 514 fragmentos |
| `area_estudio_final.*` | Polígono del área de estudio (zona de amortiguación de 20 km, delimitada según el artículo) |
| `nodos_conefor.txt` | Identificador y área de cada uno de los 514 nodos (formato de Conefor) |
| `distancias_conefor_unica.txt` | Distancias entre fragmentos (131.841 pares, sin duplicados direccionales) |

### validacion_conefor/

| Archivo | Contenido |
|---|---|
| `overall_indices.txt` | Índices globales calculados con Conefor 2.6 (validación del IIC) |
| `node_importances_limpio.txt` | dIIC por fragmento calculado con Conefor 2.6 |

### resultados/

Para cada escenario `{10|30|50}`:

| Archivo | Contenido |
|---|---|
| `resultados_ga_{N}_final.csv` | Resultados de las 30 corridas individuales del AG |
| `resumen_{N}.csv` | Estadística resumen del escenario |
| `composicion_ga_{N}.csv` | Frecuencia de selección de cada fragmento en las 30 corridas |
| `parametros_{N}.txt` | Parámetros y resultados en texto |

Además, `resumen_final_todos_escenarios.csv` consolida los tres escenarios.

### Correspondencia con el artículo

| Elemento del artículo | Archivo |
|---|---|
| Tabla 1 (parámetros del AG) | `parametros_{N}.txt` |
| Tabla 2 (escenario del 30 %) | `resumen_30.csv` |
| Tabla 3 (tres presupuestos) | `resumen_final_todos_escenarios.csv` |
| Tabla 4 (robustez, subgrafo inducido) | `robustez_subgrafo_inducido_resultados.csv`  |
| Tabla 5 y Figura 3 (frecuencia de selección) | `composicion_ga_30.csv` |
| Figura 2 (aptitud por estrategia y presupuesto) | `resumen_final_todos_escenarios.csv` |


## 4. Fuentes de datos originales

Los datos originales no se incluyen por su tamaño y porque son públicos. Se obtuvieron de:

| Dato | Fuente | Versión | Fecha de descarga | Enlace |
|---|---|---|---|---|
| Cobertura forestal: Bosque-No Bosque 10K  | SINCHI / SIAT-AC | Última versión  | 13/09/2026 | https://hub.arcgis.com/datasets/9627a4ced7cb41f5802fd601ef6bf79a/about |
| Registro Único Nacional de Áreas Protegidas (RUNAP) | Parques Nacionales Naturales de Colombia | Última versión disponible a la fecha | **[fecha]** | https://runap.parquesnacionales.gov.co/cifras
https://storage.googleapis.com/pnn_geodatabase/runap/latest.zip
 |
| Marco Geoestadístico Nacional 2024 (MGN), municipios | DANE / IGAC | 2024 | 13/09/2026 | https://geoportal.dane.gov.co/descargas/mgn_2024/MGN2024_00_COLOMBIA.zip |

Preprocesamiento en QGIS **[versión]**: corrección de geometrías, reproyección a EPSG:9377, recorte al área de estudio, selección de la clase de bosque, disolución, separación en fragmentos individuales y filtro de tamaño mínimo (1,0 ha), que dejó 514 fragmentos.

Los términos de uso de los datos originales corresponden a sus proveedores.

## 5. Requisitos

- Python 3.13.5
- Librerías de Python (ver `requirements.txt`):

```
deap==1.4.4
networkx==3.6.1
numpy==2.4.3
pandas==3.0.3
scipy==1.17.1
```

- Software externo, usado solo para preparar y validar los datos (no es necesario para ejecutar el script):
  - QGIS **[versión]**
  - Conefor **2.6**

## 6. Instalación y ejecución

1. Clonar o descargar el repositorio.
2. Crear un entorno virtual e instalar las dependencias:

```
python -m venv entorno
entorno\Scripts\activate        # Windows
# source entorno/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

3. Ejecutar el script desde la raíz del repositorio:

```
python codigo/script_maestro_final.py
```

El script carga los nodos y las distancias, construye el grafo, calcula el IIC y el dIIC, y ejecuta el AG, la priorización por dIIC, la heurística por área y la construcción aleatoria para los tres escenarios, en el orden 30 %, 10 % y 50 %. Los resultados se escriben en `resultados/`.

- Semilla(s) aleatoria(s): **[indicar]**
- Tiempo aproximado de ejecución:40 min, Acer, Aspire

## 7. Reproducibilidad

El AG es determinista con las semillas fijadas: en las 90 corridas (30 por escenario), la diferencia absoluta entre la aptitud reportada y la reproducida fue menor a 7,1 × 10⁻¹⁵. La validación con Conefor 2.6 confirmó el valor del IIC calculado por el script.

## 8. Licencia

- Código: **[MIT]**
- Datos derivados y resultados: **[CC BY 4.0]**

Ver el archivo `LICENSE`.

## 9. Cómo citar

Si usa este código o estos datos, cite el artículo indicado arriba y el repositorio archivado (DOI en el encabezado).

## 10. Autores y contacto

| Autor | Correo | ORCID |
|---|---|---|
| Yamson Jhoan Calapzu-Palma | y.calapzu@udla.edu.co | 0009-0000-0712-1903 |
| Edwin Eduardo Millan-Rojas (correspondencia) | e.millan@udla.edu.co | 0000-0002-4258-4601 |
| Lubeimar Eduardo Gallego-Ruiz | lu.gallego@udla.edu.co | 0000-0001-9117-6420 |

Programa de Ingeniería de Sistemas, Universidad de la Amazonia, Florencia, Caquetá, Colombia.