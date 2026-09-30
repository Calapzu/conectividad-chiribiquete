# -*- coding: utf-8 -*-
"""
SCRIPT MAESTRO CORREGIDO
Optimización de conectividad ecológica mediante algoritmo genético
vs. dIIC en la zona de amortiguación de Chiribiquete, Caquetá.

CORRECCIÓN CENTRAL RESPECTO A LA VERSIÓN ANTERIOR:
- Esta copia conserva la lógica del GA; las secciones 28–29 solo se ejecutan cuando el archivo se ejecuta directamente, no al importarlo.
- Se utiliza IIC convencional normalizado:
      IIC = sum(ai * aj / (1 + nlij)) / A_L^2
- dIIC se calcula sobre IIC, no sobre sqrt(IIC), y se recalculan los caminos mínimos después de eliminar cada nodo.
- La función de aptitud es:
      (IIC(X) / IIC_TOTAL) * 100
- Se mantiene el resto de la metodología: red, umbral, presupuestos,
  GA, semillas, operadores, elitismo, control aleatorio, bootstrap
  y pruebas estadísticas.
"""

import subprocess
import sys

# Dependencias, igual que en el script original.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "deap", "networkx"],
    check=False
)

import os
import math
import random
import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import shapiro, wilcoxon, ttest_1samp
from deap import base, creator, tools


# ============================================================
# 2. CONFIGURACIÓN GENERAL
# ============================================================

ARCHIVO_NODOS = "nodos_conefor.txt"
ARCHIVO_DISTANCIAS = "distancias_conefor_unica.txt"

UMBRAL = 7500.0

# Se conserva el orden de ejecución original.
PRESUPUESTOS = [0.30, 0.10, 0.50]

N_RUNS = 30
POBLACION_SIZE = 200
MAX_GENERACIONES = 500

CX_PROB = 0.80
MUT_PROB = 0.20
UNIFORM_INDPB = 0.50
MUTATION_INDPB = 0.04
TOURNAMENT_SIZE = 3
ELITISMO = 10

FRACCION_LARGEST_FIRST = 0.20
FRACCION_RANDOM = 0.80

LAMBDA = 200.0

PACIENCIA = 100
TOL_CONVERGENCIA = 1e-6

N_RANDOM = 1000
SEED_RANDOM = 42

N_BOOTSTRAP = 10000
SEED_BOOTSTRAP = 42

SEMILLAS = list(range(1, N_RUNS + 1))


# ============================================================
# 3. DIRECTORIO DE SALIDA
# ============================================================

if os.path.exists("/content"):
    DIRECTORIO_SALIDA = "/content"
else:
    DIRECTORIO_SALIDA = os.path.join(
        os.getcwd(), "resultados_conectividad_IIC_corregido"
    )
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)


# ============================================================
# 4. INFORMACIÓN INICIAL
# ============================================================

print("=" * 75)
print("SCRIPT MAESTRO CORREGIDO — IIC CONVENCIONAL")
print("CONECTIVIDAD ECOLÓGICA")
print("=" * 75)
print(f"Umbral de distancia: {UMBRAL:,.0f} m")
print("Escenarios: 30% -> 10% -> 50%")
print(f"Corridas GA por escenario: {N_RUNS}")
print(f"Población: {POBLACION_SIZE}")
print(f"Generaciones máximas: {MAX_GENERACIONES}")
print(f"Lambda: {LAMBDA}")
print(f"Directorio salida: {DIRECTORIO_SALIDA}")
print("=" * 75)


# ============================================================
# 5. CARGAR NODOS
# ============================================================

print("\nCargando nodos...")

AREAS = {}

with open(ARCHIVO_NODOS, "r", encoding="utf-8") as f:
    for linea in f:
        partes = linea.strip().split()
        if len(partes) < 2:
            continue

        try:
            nodo = int(float(partes[0]))
            area = float(partes[1])
        except (ValueError, TypeError):
            continue

        AREAS[nodo] = area

NODOS = sorted(AREAS.keys())
N = len(NODOS)
idx_de_nodo = {nodo: i for i, nodo in enumerate(NODOS)}
AREAS_ARR = np.array([AREAS[n] for n in NODOS], dtype=float)
AREA_TOTAL = AREAS_ARR.sum()

print(f"Fragmentos: {N}")
print(f"Área total: {AREA_TOTAL:,.6f} ha")


# ============================================================
# 6. CONSTRUIR GRAFO BASE
# ============================================================

print("\nConstruyendo grafo base...")

G_BASE = nx.Graph()
G_BASE.add_nodes_from(NODOS)

with open(ARCHIVO_DISTANCIAS, "r", encoding="utf-8") as f:
    for linea in f:
        partes = linea.strip().split()
        if len(partes) < 3:
            continue

        try:
            a = int(float(partes[0]))
            b = int(float(partes[1]))
            d = float(partes[2])
        except (ValueError, TypeError):
            continue

        if d <= UMBRAL:
            G_BASE.add_edge(a, b)

print(f"Nodos en grafo: {G_BASE.number_of_nodes()}")
print(f"Aristas <= {UMBRAL:.0f} m: {G_BASE.number_of_edges()}")


# ============================================================
# 7. CAMINOS MÍNIMOS TOPOLOGÓGICOS
# ============================================================

print("\nCalculando caminos mínimos topológicos...")

CAMINOS = dict(nx.all_pairs_shortest_path_length(G_BASE))


# ============================================================
# 8. CONSTRUIR MATRIZ NL
# ============================================================

print("Construyendo matriz NL...")

NL = [[None] * N for _ in range(N)]

for i, ni in enumerate(NODOS):
    NL[i][i] = 0
    caminos_i = CAMINOS.get(ni, {})

    for nj, saltos in caminos_i.items():
        if nj not in idx_de_nodo:
            continue
        j = idx_de_nodo[nj]
        NL[i][j] = saltos


# ============================================================
# 9. CONSTRUIR MATRIZ M
# ============================================================

print("\nCalculando matriz M...")

M = np.zeros((N, N), dtype=float)

for i in range(N):
    for j in range(N):
        if NL[i][j] is not None:
            M[i, j] = (
                AREAS_ARR[i] * AREAS_ARR[j]
            ) / (1.0 + NL[i][j])


# ============================================================
# 10. IIC TOTAL CONVENCIONAL
# ============================================================

sum_all = M.sum()

# IIC convencional normalizado por el área total del paisaje.
IIC_TOTAL = sum_all / (AREA_TOTAL ** 2)

print("\n" + "=" * 75)
print("VALIDACIÓN DEL IIC TOTAL")
print("=" * 75)
print(f"sum_all = {sum_all:.10f}")
print(f"Área total^2 = {AREA_TOTAL ** 2:.10f}")
print(f"IIC_TOTAL = {IIC_TOTAL:.12f}")
print("=" * 75)


# ============================================================
# 11. FUNCIÓN IIC DE UN INDIVIDUO
# ============================================================

def calcular_iic(individuo):
    """Calcula el IIC convencional del subconjunto seleccionado."""

    seleccionados = np.flatnonzero(np.asarray(individuo) == 1)

    if len(seleccionados) == 0:
        return 0.0

    suma_iic = M[np.ix_(seleccionados, seleccionados)].sum()

    if suma_iic <= 0:
        return 0.0

    return suma_iic / (AREA_TOTAL ** 2)


# ============================================================
# 12. PRUEBA DE CONSISTENCIA DEL IIC TOTAL
# ============================================================

ind_todos = np.ones(N, dtype=np.int8)
iic_prueba = calcular_iic(ind_todos)

diferencia_iic = (
    abs(iic_prueba - IIC_TOTAL) / IIC_TOTAL
) * 100.0

print("\nComprobación interna:")
print(f"IIC matriz M: {IIC_TOTAL:.12f}")
print(f"IIC función: {iic_prueba:.12f}")
print(f"Diferencia: {diferencia_iic:.12f}%")


# ============================================================
# 13. FUNCIONES DE COSTO Y FITNESS
# ============================================================

def calcular_costo(individuo):
    return float(np.dot(individuo, AREAS_ARR))


def evaluar_fitness(individuo, presupuesto):
    """
    Fitness = porcentaje del IIC total retenido por el subconjunto.

    Se conserva la penalización del script original para individuos
    temporalmente infeasibles durante la evolución.
    """
    iic = calcular_iic(individuo)
    costo = calcular_costo(individuo)

    fitness = (iic / IIC_TOTAL) * 100.0

    if costo > presupuesto:
        fitness -= LAMBDA * (costo - presupuesto) / 1000.0

    return fitness


# ============================================================
# 14. dIIC INDIVIDUAL CONVENCIONAL — ELIMINACIÓN REAL DEL NODO
# ============================================================

print(f"\nCalculando dIIC convencional para {N} fragmentos...")

# IMPORTANTE:
# Al eliminar un fragmento no basta con retirar su fila y columna de M.
# El fragmento eliminado puede actuar como intermediario en caminos
# mínimos entre otros fragmentos. Por ello, para cada nodo se elimina
# realmente de la red y se recalculan los caminos mínimos topológicos
# de la red restante.
#
# Se usa scipy.sparse.csgraph.shortest_path para efectuar esta auditoría
# de forma eficiente. El resultado se compara posteriormente con Conefor.

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# Matriz de adyacencia no ponderada de la red base.
_filas = []
_columnas = []

for a, b in G_BASE.edges():
    ia = idx_de_nodo[a]
    ib = idx_de_nodo[b]
    _filas.extend([ia, ib])
    _columnas.extend([ib, ia])

_ADYACENCIA = csr_matrix(
    (np.ones(len(_filas), dtype=np.int8),
     (_filas, _columnas)),
    shape=(N, N)
)

dIIC = {}

for k, nodo in enumerate(NODOS):
    conservar = np.ones(N, dtype=bool)
    conservar[k] = False

    ady_sin = _ADYACENCIA[conservar][:, conservar]

    distancias_sin = shortest_path(
        ady_sin,
        directed=False,
        unweighted=True
    )

    areas_sin = AREAS_ARR[conservar]

    finitas = np.isfinite(distancias_sin)

    M_sin = np.where(
        finitas,
        np.outer(areas_sin, areas_sin) / (1.0 + distancias_sin),
        0.0
    )

    sum_without_k = M_sin.sum()

    if sum_without_k > 0:
        iic_without_k = sum_without_k / (AREA_TOTAL ** 2)
    else:
        iic_without_k = 0.0

    dIIC[nodo] = (
        (IIC_TOTAL - iic_without_k) / IIC_TOTAL
    ) * 100.0

print("dIIC calculado mediante eliminación real y recálculo de caminos mínimos.")

print("\nTop 10 dIIC:")
for fid, valor in sorted(dIIC.items(), key=lambda x: -x[1])[:10]:
    print(
        f"  ID {fid}: dIIC = {valor:.6f}% | "
        f"área = {AREAS[fid]:,.2f} ha"
    )

print("\nTop 10 dIIC:")
for fid, valor in sorted(dIIC.items(), key=lambda x: -x[1])[:10]:
    print(
        f"  ID {fid}: dIIC = {valor:.6f}% | "
        f"área = {AREAS[fid]:,.2f} ha"
    )


# ============================================================
# 15. ÓRDENES PARA HEURÍSTICOS
# ============================================================

orden_area = np.argsort(-AREAS_ARR)
orden_dIIC = np.argsort(
    [dIIC[nodo] for nodo in NODOS]
)[::-1]


# ============================================================
# 16. HEURÍSTICO DE ÁREA
# ============================================================

def solucion_heuristica(presupuesto):
    individuo = np.zeros(N, dtype=np.int8)
    costo = 0.0

    for i in orden_area:
        nuevo_costo = costo + AREAS_ARR[i]

        if nuevo_costo <= presupuesto:
            individuo[i] = 1
            costo = nuevo_costo

    return individuo, costo


# ============================================================
# 17. dIIC — MÉTODO B
# ============================================================

def solucion_dIIC(presupuesto):
    individuo = np.zeros(N, dtype=np.int8)
    costo = 0.0

    for i in orden_dIIC:
        nuevo_costo = costo + AREAS_ARR[i]

        if nuevo_costo <= presupuesto:
            individuo[i] = 1
            costo = nuevo_costo

    return individuo, costo


# ============================================================
# 18. SOLUCIÓN ALEATORIA FACTIBLE
# ============================================================

def solucion_aleatoria_factible(presupuesto, rng):
    indices = np.arange(N)
    rng.shuffle(indices)

    individuo = np.zeros(N, dtype=np.int8)
    costo = 0.0

    for i in indices:
        nuevo_costo = costo + AREAS_ARR[i]

        if nuevo_costo <= presupuesto:
            individuo[i] = 1
            costo = nuevo_costo

    return individuo, costo


# ============================================================
# 19. CONFIGURACIÓN DE DEAP
# ============================================================

if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax

if hasattr(creator, "Individual"):
    del creator.Individual

creator.create(
    "FitnessMax",
    base.Fitness,
    weights=(1.0,)
)

creator.create(
    "Individual",
    list,
    fitness=creator.FitnessMax
)

toolbox = base.Toolbox()

toolbox.register(
    "mate",
    tools.cxUniform,
    indpb=UNIFORM_INDPB
)

toolbox.register(
    "mutate",
    tools.mutFlipBit,
    indpb=MUTATION_INDPB
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=TOURNAMENT_SIZE
)


# ============================================================
# 20. INDIVIDUO LARGEST-FIRST
# ============================================================

def individuo_largest_first(presupuesto):
    individuo = np.zeros(N, dtype=np.int8)
    costo = 0.0

    for i in orden_area:
        nuevo_costo = costo + AREAS_ARR[i]

        if nuevo_costo <= presupuesto:
            individuo[i] = 1
            costo = nuevo_costo

    return creator.Individual(individuo.tolist())


# ============================================================
# 21. INDIVIDUO ALEATORIO FACTIBLE
# ============================================================

def individuo_random_factible(presupuesto, rng):
    individuo, _ = solucion_aleatoria_factible(
        presupuesto,
        rng
    )
    return creator.Individual(individuo.tolist())


# ============================================================
# 22. POBLACIÓN INICIAL
# ============================================================

def crear_poblacion_inicial(presupuesto, rng):
    poblacion = []

    n_largest = int(
        POBLACION_SIZE * FRACCION_LARGEST_FIRST
    )

    n_random = POBLACION_SIZE - n_largest

    # Se conserva exactamente la metodología original:
    # 40 copias de la solución largest-first + 160 aleatorias
    # para una población de 200 individuos.
    for _ in range(n_largest):
        poblacion.append(
            individuo_largest_first(presupuesto)
        )

    for _ in range(n_random):
        poblacion.append(
            individuo_random_factible(
                presupuesto,
                rng
            )
        )

    return poblacion


# ============================================================
# 23. EJECUTAR UNA CORRIDA DEL GA
# ============================================================

def ejecutar_ga(presupuesto, seed):

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    poblacion = crear_poblacion_inicial(
        presupuesto,
        rng
    )

    for individuo in poblacion:
        individuo.fitness.values = (
            evaluar_fitness(
                individuo,
                presupuesto
            ),
        )

    mejor = tools.selBest(
        poblacion,
        1
    )[0]

    mejor = toolbox.clone(mejor)
    mejor_fitness = mejor.fitness.values[0]

    historial_mejor = [mejor_fitness]
    gen_convergencia = None

    for gen in range(
        1,
        MAX_GENERACIONES + 1
    ):

        seleccionados = toolbox.select(
            poblacion,
            POBLACION_SIZE - ELITISMO
        )

        descendientes = list(
            map(
                toolbox.clone,
                seleccionados
            )
        )

        for i in range(
            0,
            len(descendientes) - 1,
            2
        ):

            if random.random() < CX_PROB:

                toolbox.mate(
                    descendientes[i],
                    descendientes[i + 1]
                )

                if descendientes[i].fitness.valid:
                    del descendientes[
                        i
                    ].fitness.values

                if descendientes[
                    i + 1
                ].fitness.valid:
                    del descendientes[
                        i + 1
                    ].fitness.values

        for individuo in descendientes:

            if random.random() < MUT_PROB:

                toolbox.mutate(individuo)

                if individuo.fitness.valid:
                    del individuo.fitness.values

        individuos_invalidos = [
            ind
            for ind in descendientes
            if not ind.fitness.valid
        ]

        for individuo in individuos_invalidos:

            individuo.fitness.values = (
                evaluar_fitness(
                    individuo,
                    presupuesto
                ),
            )

        elite = tools.selBest(
            poblacion,
            ELITISMO
        )

        elite = list(
            map(
                toolbox.clone,
                elite
            )
        )

        poblacion = elite + descendientes

        mejor_actual = tools.selBest(
            poblacion,
            1
        )[0]

        fitness_actual = (
            mejor_actual.fitness.values[0]
        )

        if fitness_actual > mejor_fitness:

            mejor = toolbox.clone(
                mejor_actual
            )

            mejor_fitness = fitness_actual

        historial_mejor.append(
            mejor_fitness
        )

        if len(historial_mejor) >= PACIENCIA:

            ultimos = historial_mejor[
                -PACIENCIA:
            ]

            rango = (
                max(ultimos)
                - min(ultimos)
            )

            if rango < TOL_CONVERGENCIA:

                gen_convergencia = gen
                break

    individuo_final = np.array(
        mejor,
        dtype=np.int8
    )

    costo_final = calcular_costo(
        individuo_final
    )

    iic_final = calcular_iic(
        individuo_final
    )

    fitness_final = (
        iic_final / IIC_TOTAL
    ) * 100.0

    # Importante: se recalcula el fitness final sin penalización.
    # Las soluciones finales reportadas son evaluadas con la métrica
    # IIC real, independientemente de la penalización usada durante
    # la evolución.
    return {
        "seed": seed,
        "fitness": fitness_final,
        "IIC": iic_final,
        "costo": costo_final,
        "n_fragmentos": int(
            individuo_final.sum()
        ),
        "generacion": (
            gen_convergencia
            if gen_convergencia is not None
            else MAX_GENERACIONES
        ),
        "convergencia": (
            gen_convergencia is not None
        ),
        "individuo": individuo_final
    }


# ============================================================
# 24. BOOTSTRAP
# ============================================================

def bootstrap_media(
    valores,
    n_bootstrap=10000,
    seed=42
):
    valores = np.asarray(
        valores,
        dtype=float
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(valores)

    medias = np.empty(
        n_bootstrap
    )

    for i in range(
        n_bootstrap
    ):

        muestra = rng.choice(
            valores,
            size=n,
            replace=True
        )

        medias[i] = muestra.mean()

    low = np.percentile(
        medias,
        2.5
    )

    high = np.percentile(
        medias,
        97.5
    )

    return low, high


# ============================================================
# 25. ANÁLISIS ESTADÍSTICO
# ============================================================

def analizar_estadistica(
    resultados_ga,
    valor_dIIC,
    valor_heuristico
):

    valores = np.array(
        [
            r["fitness"]
            for r in resultados_ga
        ],
        dtype=float
    )

    media = valores.mean()
    sd = valores.std(ddof=1)
    minimo = valores.min()
    maximo = valores.max()

    W, p_shapiro = shapiro(
        valores
    )

    diferencias_dIIC = (
        valores - valor_dIIC
    )

    if p_shapiro >= 0.05:

        estadistico_dIIC, p_dIIC = (
            ttest_1samp(
                valores,
                valor_dIIC,
                alternative="greater"
            )
        )

        prueba_dIIC = (
            "t de una muestra"
        )

    else:

        estadistico_dIIC, p_dIIC = (
            wilcoxon(
                diferencias_dIIC,
                alternative="greater"
            )
        )

        prueba_dIIC = (
            "Wilcoxon de una muestra"
        )

    diferencias_heuristico = (
        valores - valor_heuristico
    )

    if p_shapiro >= 0.05:

        (
            estadistico_heuristico,
            p_heuristico
        ) = ttest_1samp(
            valores,
            valor_heuristico,
            alternative="greater"
        )

        prueba_heuristico = (
            "t de una muestra"
        )

    else:

        (
            estadistico_heuristico,
            p_heuristico
        ) = wilcoxon(
            diferencias_heuristico,
            alternative="greater"
        )

        prueba_heuristico = (
            "Wilcoxon de una muestra"
        )

    bootstrap_low, bootstrap_high = (
        bootstrap_media(
            valores,
            N_BOOTSTRAP,
            SEED_BOOTSTRAP
        )
    )

    if sd > 0:

        diferencia_est_dIIC = (
            media - valor_dIIC
        ) / sd

        diferencia_est_heuristico = (
            media - valor_heuristico
        ) / sd

    else:

        diferencia_est_dIIC = np.nan
        diferencia_est_heuristico = np.nan

    return {
        "n": len(valores),
        "media": media,
        "sd": sd,
        "minimo": minimo,
        "maximo": maximo,

        "shapiro_W": W,
        "shapiro_p": p_shapiro,

        "prueba_dIIC": prueba_dIIC,
        "estadistico_dIIC": (
            estadistico_dIIC
        ),
        "p_dIIC": p_dIIC,

        "prueba_heuristico": (
            prueba_heuristico
        ),
        "estadistico_heuristico": (
            estadistico_heuristico
        ),
        "p_heuristico": p_heuristico,

        "bootstrap_low": bootstrap_low,
        "bootstrap_high": bootstrap_high,

        "diferencia_pp_dIIC": (
            media - valor_dIIC
        ),
        "diferencia_pp_heuristico": (
            media - valor_heuristico
        ),

        "diferencia_estandarizada_dIIC": (
            diferencia_est_dIIC
        ),
        "diferencia_estandarizada_heuristico": (
            diferencia_est_heuristico
        )
    }


# ============================================================
# 26. GUARDAR RESULTADOS
# ============================================================

def guardar_resultados_escenario(
    porcentaje,
    resultados_ga,
    valor_dIIC,
    valor_heuristico,
    valor_random_mean,
    valor_random_sd,
    valor_random_interval_low,
    valor_random_interval_high,
    estadistica
):

    etiqueta = int(
        porcentaje * 100
    )

    filas = []

    for r in resultados_ga:

        filas.append({
            "presupuesto_porcentaje":
                porcentaje * 100,

            "presupuesto_ha":
                porcentaje * AREA_TOTAL,

            "seed":
                r["seed"],

            "fitness":
                r["fitness"],

            "IIC":
                r["IIC"],

            "costo_ha":
                r["costo"],

            "n_fragmentos":
                r["n_fragmentos"],

            "generacion":
                r["generacion"],

            "convergencia":
                r["convergencia"]
        })

    df_ga = pd.DataFrame(
        filas
    )

    archivo_ga = os.path.join(
        DIRECTORIO_SALIDA,
        f"resultados_ga_{etiqueta}_final.csv"
    )

    df_ga.to_csv(
        archivo_ga,
        index=False
    )

    fila_resumen = {

        "presupuesto_porcentaje":
            porcentaje * 100,

        "presupuesto_ha":
            porcentaje * AREA_TOTAL,

        "GA_media":
            estadistica["media"],

        "GA_SD":
            estadistica["sd"],

        "GA_min":
            estadistica["minimo"],

        "GA_max":
            estadistica["maximo"],

        "dIIC":
            valor_dIIC,

        "heuristico":
            valor_heuristico,

        "aleatorio_media":
            valor_random_mean,

        "aleatorio_SD":
            valor_random_sd,

        "aleatorio_intervalo_empirico_low":
            valor_random_interval_low,

        "aleatorio_intervalo_empirico_high":
            valor_random_interval_high,

        "GA_menos_dIIC_pp":
            estadistica[
                "diferencia_pp_dIIC"
            ],

        "GA_menos_heuristico_pp":
            estadistica[
                "diferencia_pp_heuristico"
            ],

        "diferencia_estandarizada_dIIC":
            estadistica[
                "diferencia_estandarizada_dIIC"
            ],

        "diferencia_estandarizada_heuristico":
            estadistica[
                "diferencia_estandarizada_heuristico"
            ],

        "Shapiro_W":
            estadistica["shapiro_W"],

        "Shapiro_p":
            estadistica["shapiro_p"],

        "prueba_dIIC":
            estadistica["prueba_dIIC"],

        "estadistico_dIIC":
            estadistica[
                "estadistico_dIIC"
            ],

        "p_dIIC":
            estadistica["p_dIIC"],

        "prueba_heuristico":
            estadistica[
                "prueba_heuristico"
            ],

        "estadistico_heuristico":
            estadistica[
                "estadistico_heuristico"
            ],

        "p_heuristico":
            estadistica["p_heuristico"],

        "bootstrap_CI95_low":
            estadistica["bootstrap_low"],

        "bootstrap_CI95_high":
            estadistica["bootstrap_high"]
    }

    df_resumen = pd.DataFrame(
        [fila_resumen]
    )

    archivo_resumen = os.path.join(
        DIRECTORIO_SALIDA,
        f"resumen_{etiqueta}.csv"
    )

    df_resumen.to_csv(
        archivo_resumen,
        index=False
    )

    frecuencia = np.zeros(
        N,
        dtype=int
    )

    for r in resultados_ga:
        frecuencia += r["individuo"]

    df_composicion = pd.DataFrame({

        "Node":
            NODOS,

        "Area_ha":
            AREAS_ARR,

        "dIIC":
            [
                dIIC[nodo]
                for nodo in NODOS
            ],

        "frecuencia_GA":
            frecuencia,

        "frecuencia_relativa":
            frecuencia / N_RUNS
    })

    archivo_composicion = os.path.join(
        DIRECTORIO_SALIDA,
        f"composicion_ga_{etiqueta}.csv"
    )

    df_composicion.to_csv(
        archivo_composicion,
        index=False
    )

    archivo_txt = os.path.join(
        DIRECTORIO_SALIDA,
        f"parametros_{etiqueta}.txt"
    )

    with open(
        archivo_txt,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ANALISIS DE CONECTIVIDAD ECOLOGICA\n"
        )
        f.write(
            "===================================\n\n"
        )

        f.write(
            "METODO DE CONECTIVIDAD\n"
        )
        f.write(
            "----------------------\n"
        )
        f.write(
            "IIC convencional normalizado basado "
            "en caminos mínimos topológicos (NL)\n"
        )
        f.write(
            "IIC = sum(ai*aj/(1+nlij)) / A_L^2\n\n"
        )

        f.write(
            f"IIC_TOTAL = {IIC_TOTAL:.12f}\n"
        )
        f.write(
            f"sum_all = {sum_all:.12f}\n"
        )
        f.write(
            f"Umbral = {UMBRAL:.0f} m\n"
        )
        f.write(
            f"Nodos = {N}\n"
        )
        f.write(
            f"Area total = {AREA_TOTAL:.12f} ha\n\n"
        )

        f.write(
            "PARAMETROS GA\n"
        )
        f.write(
            "-------------\n"
        )
        f.write(
            f"Corridas = {N_RUNS}\n"
        )
        f.write(
            f"Poblacion = {POBLACION_SIZE}\n"
        )
        f.write(
            f"Generaciones maximas = "
            f"{MAX_GENERACIONES}\n"
        )
        f.write(
            f"Crossover probability = "
            f"{CX_PROB}\n"
        )
        f.write(
            f"Mutation probability = "
            f"{MUT_PROB}\n"
        )
        f.write(
            f"Uniform indpb = "
            f"{UNIFORM_INDPB}\n"
        )
        f.write(
            f"Mutation indpb = "
            f"{MUTATION_INDPB}\n"
        )
        f.write(
            f"Tournament size = "
            f"{TOURNAMENT_SIZE}\n"
        )
        f.write(
            f"Elitismo = {ELITISMO}\n"
        )
        f.write(
            f"Lambda = {LAMBDA}\n"
        )
        f.write(
            f"Paciencia = {PACIENCIA}\n"
        )
        f.write(
            f"Tolerancia = "
            f"{TOL_CONVERGENCIA}\n"
        )
        f.write(
            f"Largest-first = "
            f"{FRACCION_LARGEST_FIRST * 100:.0f}%\n"
        )
        f.write(
            f"Random feasible = "
            f"{FRACCION_RANDOM * 100:.0f}%\n\n"
        )

        f.write(
            "RESULTADOS\n"
        )
        f.write(
            "----------\n"
        )
        f.write(
            "Fitness = (IIC(X) / IIC_TOTAL) * 100\n"
        )
        f.write(
            f"GA media = "
            f"{estadistica['media']:.10f}%\n"
        )
        f.write(
            f"GA SD = "
            f"{estadistica['sd']:.10f} pp\n"
        )
        f.write(
            f"GA minimo = "
            f"{estadistica['minimo']:.10f}%\n"
        )
        f.write(
            f"GA maximo = "
            f"{estadistica['maximo']:.10f}%\n"
        )
        f.write(
            f"dIIC = "
            f"{valor_dIIC:.10f}%\n"
        )
        f.write(
            f"Heuristico = "
            f"{valor_heuristico:.10f}%\n"
        )
        f.write(
            f"Aleatorio media = "
            f"{valor_random_mean:.10f}%\n"
        )
        f.write(
            f"Aleatorio SD = "
            f"{valor_random_sd:.10f} pp\n"
        )
        f.write(
            f"Aleatorio intervalo empirico 95% = "
            f"{valor_random_interval_low:.10f} - "
            f"{valor_random_interval_high:.10f}%\n"
        )
        f.write(
            f"GA - dIIC = "
            f"{estadistica['diferencia_pp_dIIC']:.10f} pp\n"
        )
        f.write(
            f"GA - heuristico = "
            f"{estadistica['diferencia_pp_heuristico']:.10f} pp\n"
        )
        f.write(
            f"Bootstrap IC95 de la media GA = "
            f"{estadistica['bootstrap_low']:.10f} - "
            f"{estadistica['bootstrap_high']:.10f}%\n"
        )

    print("\nArchivos guardados:")
    print(f"  {archivo_ga}")
    print(f"  {archivo_resumen}")
    print(f"  {archivo_composicion}")
    print(f"  {archivo_txt}")


# ============================================================
# 27. ANALIZAR UN ESCENARIO
# ============================================================

def analizar_escenario(porcentaje):

    presupuesto = (
        porcentaje * AREA_TOTAL
    )

    etiqueta = int(
        porcentaje * 100
    )

    print(
        "\n\n" + "=" * 75
    )
    print(
        f"ESCENARIO {etiqueta}%"
    )
    print(
        "=" * 75
    )
    print(
        f"Presupuesto = "
        f"{presupuesto:,.6f} ha"
    )

    # --------------------------------------------------------
    # Heurística largest-first
    # --------------------------------------------------------

    ind_heur, costo_heur = (
        solucion_heuristica(
            presupuesto
        )
    )

    iic_heur = calcular_iic(
        ind_heur
    )

    fitness_heur = (
        iic_heur / IIC_TOTAL
    ) * 100.0

    print(
        f"\nHeurístico: "
        f"{fitness_heur:.6f}%"
    )
    print(
        f"Costo: "
        f"{costo_heur:.6f} ha"
    )

    # --------------------------------------------------------
    # dIIC
    # --------------------------------------------------------

    ind_dIIC, costo_dIIC = (
        solucion_dIIC(
            presupuesto
        )
    )

    iic_dIIC = calcular_iic(
        ind_dIIC
    )

    fitness_dIIC = (
        iic_dIIC / IIC_TOTAL
    ) * 100.0

    print(
        f"\ndIIC Método B: "
        f"{fitness_dIIC:.6f}%"
    )
    print(
        f"Costo: "
        f"{costo_dIIC:.6f} ha"
    )

    # --------------------------------------------------------
    # Control aleatorio
    # --------------------------------------------------------

    print(
        f"\nEjecutando "
        f"{N_RANDOM:,} soluciones aleatorias..."
    )

    rng_random = np.random.default_rng(
        SEED_RANDOM + etiqueta
    )

    valores_random = []

    for _ in range(
        N_RANDOM
    ):

        ind_random, _ = (
            solucion_aleatoria_factible(
                presupuesto,
                rng_random
            )
        )

        iic_random = calcular_iic(
            ind_random
        )

        fitness_random = (
            iic_random / IIC_TOTAL
        ) * 100.0

        valores_random.append(
            fitness_random
        )

    valores_random = np.array(
        valores_random,
        dtype=float
    )

    random_mean = (
        valores_random.mean()
    )

    random_sd = (
        valores_random.std(ddof=1)
    )

    # Se conserva como intervalo empírico de los valores
    # aleatorios, NO como IC de la media.
    random_interval_low = (
        np.percentile(
            valores_random,
            2.5
        )
    )

    random_interval_high = (
        np.percentile(
            valores_random,
            97.5
        )
    )

    print(
        f"Aleatorio media: "
        f"{random_mean:.6f}%"
    )
    print(
        f"Aleatorio SD: "
        f"{random_sd:.6f} pp"
    )
    print(
        f"Intervalo empírico 95%: "
        f"{random_interval_low:.6f} - "
        f"{random_interval_high:.6f}%"
    )

    # --------------------------------------------------------
    # GA
    # --------------------------------------------------------

    print(
        f"\nIniciando "
        f"{N_RUNS} corridas GA..."
    )

    resultados_ga = []

    for numero_corrida, seed in enumerate(
        SEMILLAS,
        start=1
    ):

        print(
            f"Corrida "
            f"{numero_corrida:02d}/"
            f"{N_RUNS} "
            f"(seed={seed})... ",
            end=""
        )

        resultado = ejecutar_ga(
            presupuesto,
            seed
        )

        resultados_ga.append(
            resultado
        )

        print(
            f"fitness="
            f"{resultado['fitness']:.6f}% | "
            f"n="
            f"{resultado['n_fragmentos']} | "
            f"costo="
            f"{resultado['costo']:.2f} | "
            f"gen="
            f"{resultado['generacion']}"
        )

    # --------------------------------------------------------
    # Estadística
    # --------------------------------------------------------

    estadistica = analizar_estadistica(
        resultados_ga,
        fitness_dIIC,
        fitness_heur
    )

    print(
        "\n" + "-" * 75
    )
    print(
        f"RESULTADOS {etiqueta}%"
    )
    print(
        "-" * 75
    )
    print(
        f"GA media: "
        f"{estadistica['media']:.6f}%"
    )
    print(
        f"GA SD: "
        f"{estadistica['sd']:.6f} pp"
    )
    print(
        f"GA mínimo: "
        f"{estadistica['minimo']:.6f}%"
    )
    print(
        f"GA máximo: "
        f"{estadistica['maximo']:.6f}%"
    )
    print(
        f"dIIC: "
        f"{fitness_dIIC:.6f}%"
    )
    print(
        f"Heurístico: "
        f"{fitness_heur:.6f}%"
    )
    print(
        f"Aleatorio: "
        f"{random_mean:.6f}% "
        f"(SD={random_sd:.6f})"
    )
    print(
        f"GA - dIIC: "
        f"{estadistica['diferencia_pp_dIIC']:.6f} pp"
    )
    print(
        f"GA - heurístico: "
        f"{estadistica['diferencia_pp_heuristico']:.6f} pp"
    )
    print(
        f"Shapiro-Wilk: "
        f"W={estadistica['shapiro_W']:.6f}, "
        f"p={estadistica['shapiro_p']:.8g}"
    )
    print(
        f"{estadistica['prueba_dIIC']}: "
        f"estadístico="
        f"{estadistica['estadistico_dIIC']:.6f}, "
        f"p="
        f"{estadistica['p_dIIC']:.8g}"
    )
    print(
        f"{estadistica['prueba_heuristico']}: "
        f"estadístico="
        f"{estadistica['estadistico_heuristico']:.6f}, "
        f"p="
        f"{estadistica['p_heuristico']:.8g}"
    )
    print(
        f"Bootstrap 95% CI de la media GA: "
        f"{estadistica['bootstrap_low']:.6f} - "
        f"{estadistica['bootstrap_high']:.6f}%"
    )
    print(
        f"Diferencia estandarizada vs dIIC: "
        f"{estadistica['diferencia_estandarizada_dIIC']:.6f}"
    )
    print(
        f"Diferencia estandarizada vs heurístico: "
        f"{estadistica['diferencia_estandarizada_heuristico']:.6f}"
    )

    print(
        f"\nGuardando escenario "
        f"{etiqueta}%..."
    )

    guardar_resultados_escenario(
        porcentaje=porcentaje,
        resultados_ga=resultados_ga,
        valor_dIIC=fitness_dIIC,
        valor_heuristico=fitness_heur,
        valor_random_mean=random_mean,
        valor_random_sd=random_sd,
        valor_random_interval_low=(
            random_interval_low
        ),
        valor_random_interval_high=(
            random_interval_high
        ),
        estadistica=estadistica
    )

    print(
        "\n" + "=" * 75
    )
    print(
        f"ESCENARIO {etiqueta}% "
        "COMPLETADO Y GUARDADO"
    )
    print(
        "=" * 75
    )

    return {
        "porcentaje": porcentaje,
        "presupuesto": presupuesto,
        "resultados_ga": resultados_ga,
        "dIIC": fitness_dIIC,
        "heuristico": fitness_heur,
        "aleatorio_media": random_mean,
        "aleatorio_sd": random_sd,
        "aleatorio_intervalo_low": (
            random_interval_low
        ),
        "aleatorio_intervalo_high": (
            random_interval_high
        ),
        "estadistica": estadistica
    }


# ============================================================
# 28-29. EJECUCIÓN COMPLETA SOLO AL EJECUTAR DIRECTAMENTE
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # 28. EJECUTAR LOS TRES ESCENARIOS
    # ============================================================

    resultados_totales = {}

    for porcentaje in PRESUPUESTOS:

        resultado = analizar_escenario(
            porcentaje
        )

        resultados_totales[
            porcentaje
        ] = resultado


    # ============================================================
    # 29. RESUMEN FINAL
    # ============================================================

    print(
        "\n\n" + "=" * 75
    )
    print(
        "RESUMEN FINAL DE TODOS LOS ESCENARIOS"
    )
    print(
        "=" * 75
    )

    tabla_final = []

    for porcentaje in PRESUPUESTOS:

        resultado = resultados_totales[
            porcentaje
        ]

        e = resultado[
            "estadistica"
        ]

        tabla_final.append({

            "Presupuesto_%":
                porcentaje * 100,

            "Presupuesto_ha":
                resultado["presupuesto"],

            "GA_media_%":
                e["media"],

            "GA_SD_pp":
                e["sd"],

            "GA_min_%":
                e["minimo"],

            "GA_max_%":
                e["maximo"],

            "dIIC_%":
                resultado["dIIC"],

            "Heuristico_%":
                resultado["heuristico"],

            "Aleatorio_%":
                resultado["aleatorio_media"],

            "Aleatorio_SD_pp":
                resultado["aleatorio_sd"],

            "GA-dIIC_pp":
                e["diferencia_pp_dIIC"],

            "GA-Heuristico_pp":
                e["diferencia_pp_heuristico"],

            "Bootstrap_low_%":
                e["bootstrap_low"],

            "Bootstrap_high_%":
                e["bootstrap_high"]
        })

    df_final = pd.DataFrame(
        tabla_final
    )

    print(
        df_final.to_string(
            index=False
        )
    )

    archivo_final = os.path.join(
        DIRECTORIO_SALIDA,
        "resumen_final_todos_escenarios.csv"
    )

    df_final.to_csv(
        archivo_final,
        index=False
    )

    print(
        "\n" + "=" * 75
    )
    print(
        "ANÁLISIS COMPLETADO"
    )
    print(
        "=" * 75
    )
    print(
        "Archivo resumen final:"
    )
    print(
        archivo_final
    )
    print(
        "\nOrden ejecutado:"
    )
    print(
        "30% -> 10% -> 50%"
    )
    print(
        "\nCada escenario fue guardado inmediatamente "
        "después de completar sus 30 corridas."
    )
    print(
        "\nMétodo:"
    )
    print(
        "IIC convencional normalizado"
    )
    print(
        "Shortest path topológico (NL)"
    )
    print(
        f"\nIIC_TOTAL = "
        f"{IIC_TOTAL:.12f}"
    )
    print(
        f"Área total = "
        f"{AREA_TOTAL:.12f} ha"
    )
    print("=" * 75)
