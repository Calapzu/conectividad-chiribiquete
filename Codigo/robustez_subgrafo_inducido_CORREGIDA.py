# -*- coding: utf-8 -*-
"""
Prueba de robustez: recalculo del IIC sobre el subgrafo inducido.

No modifica ni ejecuta los 90 GA del maestro. Reproduce solamente 6 corridas:
10%, 30% y 50%: corrida mas cercana a la mediana + mejor corrida.
Primero verifica que cada corrida reproducida coincide con el CSV original.
"""
from pathlib import Path
import os
import importlib.util
import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
PROTECTED = BASE_DIR / "script_conectividad_IIC_maestro_CORREGIDO_dIIC_solo_funciones.py"
MASTER_CANDIDATES = [
    BASE_DIR / "script_conectividad_IIC_maestro_CORREGIDO_dIIC.py",
    BASE_DIR / "script_conectividad_IIC_maestro_CORREGIDO_dIIC(1).py",
]


def crear_protegido(origen):
    texto = origen.read_text(encoding="utf-8")
    marker = "# 28. EJECUTAR LOS TRES ESCENARIOS"
    pos = texto.find(marker)
    if pos < 0:
        raise RuntimeError(f"No se encontro la seccion 28 en {origen}")
    prefijo = texto[:pos]
    ejecucion = texto[pos:]
    protegido = prefijo + '\nif __name__ == "__main__":\n'
    protegido += ''.join(
        ("    " + linea if linea.strip() else linea)
        for linea in ejecucion.splitlines(True)
    )
    PROTECTED.write_text(protegido, encoding="utf-8")


def cargar_master():
    if not PROTECTED.exists():
        origen = next((p for p in MASTER_CANDIDATES if p.exists()), None)
        if origen is None:
            raise FileNotFoundError(
                "No se encontro el maestro. Busque el archivo "
                "script_conectividad_IIC_maestro_CORREGIDO_dIIC.py "
                "en esta misma carpeta."
            )
        crear_protegido(origen)
    spec = importlib.util.spec_from_file_location("master_protegido", PROTECTED)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el maestro protegido.")
    master = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(master)
    return master


def csv_original(master, porcentaje):
    etiqueta = int(porcentaje * 100)
    ruta = BASE_DIR / "resultados_conectividad_IIC_corregido" / f"resultados_ga_{etiqueta}_final.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el CSV original de {etiqueta}%:\n{ruta}\n\n"
            "Copia la carpeta resultados_conectividad_IIC_corregido "
            "que contiene tus resultados originales junto a este script."
        )
    df = pd.read_csv(ruta)
    if not {"seed", "fitness"}.issubset(df.columns):
        raise ValueError(f"El CSV {ruta} no contiene seed y fitness.")
    if len(df) != 30:
        raise ValueError(f"El CSV {ruta} tiene {len(df)} filas; se esperaban 30.")
    df["seed"] = df["seed"].astype(int)
    df["fitness"] = df["fitness"].astype(float)
    return df, ruta


def seleccionar(df):
    vals = df["fitness"].to_numpy(float)
    med = float(np.median(vals))
    i_med = int(np.argmin(np.abs(vals - med)))
    i_best = int(np.argmax(vals))
    return {
        "mediana": (int(df.iloc[i_med].seed), float(df.iloc[i_med].fitness), med),
        "mejor": (int(df.iloc[i_best].seed), float(df.iloc[i_best].fitness), med),
    }


def iic_inducido(master, individuo):
    individuo = np.asarray(individuo, dtype=np.int8)
    ids = np.flatnonzero(individuo == 1)
    nodos = [master.NODOS[i] for i in ids]
    if not nodos:
        return 0.0, 0
    G = master.G_BASE.subgraph(nodos)
    suma = 0.0
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        dist = dict(nx.all_pairs_shortest_path_length(sub))
        for ni, drow in dist.items():
            ai = float(master.AREAS[ni])
            for nj, nl in drow.items():
                suma += ai * float(master.AREAS[nj]) / (1.0 + nl)
    return suma / (float(master.AREA_TOTAL) ** 2), len(nodos)


def main():
    print("=" * 78)
    print("PRUEBA DE ROBUSTEZ — SUBGRAFO INDUCIDO")
    print("=" * 78)

    master = cargar_master()
    print(f"Maestro protegido: {PROTECTED}")
    print(f"Nodos: {master.N} | Aristas: {master.G_BASE.number_of_edges()}")
    print(f"IIC_TOTAL: {master.IIC_TOTAL:.12f}")

    filas = []

    for porcentaje in (0.10, 0.30, 0.50):
        etiqueta = int(porcentaje * 100)
        presupuesto = porcentaje * master.AREA_TOTAL
        df, ruta = csv_original(master, porcentaje)
        sel = seleccionar(df)

        ind_h, _ = master.solucion_heuristica(presupuesto)
        fit_h = master.calcular_iic(ind_h) / master.IIC_TOTAL * 100.0
        ind_d, _ = master.solucion_dIIC(presupuesto)
        fit_d = master.calcular_iic(ind_d) / master.IIC_TOTAL * 100.0

        print(f"\n--- {etiqueta}% ---")
        print(f"CSV: {ruta}")
        print(f"dIIC: {fit_d:.10f}% | Heuristico: {fit_h:.10f}%")
        print(f"Mediana de 30: {sel['mediana'][2]:.15f}%")

        for tipo in ("mediana", "mejor"):
            seed, fit_original, _ = sel[tipo]
            resultado = master.ejecutar_ga(presupuesto, seed)
            fit_repro = float(resultado["fitness"])
            delta = abs(fit_repro - fit_original)
            if delta > 1e-9:
                raise RuntimeError(
                    f"REPRODUCCION FALLIDA: {etiqueta}%, seed={seed}. "
                    f"CSV={fit_original:.15f}, reproducido={fit_repro:.15f}, "
                    f"delta={delta:.3e}. Se detiene el analisis."
                )

            iic_ind, nsel = iic_inducido(master, resultado["individuo"])
            fit_ind = iic_ind / master.IIC_TOTAL * 100.0
            ret = fit_ind / fit_original * 100.0 if fit_original else np.nan

            filas.append({
                "presupuesto_%": etiqueta,
                "tipo_corrida": tipo,
                "seed": seed,
                "fitness_base_GA_%": fit_original,
                "fitness_reproducido_%": fit_repro,
                "delta_reproduccion": delta,
                "IIC_inducido_%": fit_ind,
                "retencion_inducido_vs_base_%": ret,
                "dIIC_%": fit_d,
                "heuristico_%": fit_h,
                "inducido_menos_dIIC_pp": fit_ind - fit_d,
                "inducido_menos_heuristico_pp": fit_ind - fit_h,
                "n_fragmentos": nsel,
                "costo_ha": float(resultado["costo"]),
            })

            print(
                f"{tipo:8s} seed={seed:2d} | reproduccion OK | "
                f"base={fit_original:.10f}% | inducido={fit_ind:.10f}% | "
                f"retencion={ret:.6f}% | n={nsel}"
            )

    out = pd.DataFrame(filas)
    csv_out = BASE_DIR / "robustez_subgrafo_inducido_resultados.csv"
    txt_out = BASE_DIR / "robustez_subgrafo_inducido_resultados.txt"
    out.to_csv(csv_out, index=False, encoding="utf-8-sig")
    txt_out.write_text(
        "PRUEBA DE ROBUSTEZ — SUBGRAFO INDUCIDO\n" + "=" * 78 + "\n\n" +
        out.to_string(index=False) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("COMPLETADO")
    print("=" * 78)
    print(out.to_string(index=False))
    print(f"\nCSV: {csv_out}")
    print(f"TXT: {txt_out}")


if __name__ == "__main__":
    main()
