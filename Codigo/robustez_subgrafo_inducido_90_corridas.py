# -*- coding: utf-8 -*-
"""
PRUEBA DE ROBUSTEZ — SUBGRAFO INDUCIDO, 90 CORRIDAS

Recalcula el IIC de las 30 soluciones GA originales de cada presupuesto
(10%, 30%, 50%) sobre el subgrafo inducido.

IMPORTANTE:
- No vuelve a optimizar con semillas nuevas.
- Los CSV originales contienen seed y fitness, pero no guardan el vector
  binario del individuo. Por ello, cada seed se reproduce de forma
  determinista mediante ejecutar_ga() del maestro protegido.
- La reproducción se verifica contra el fitness guardado en el CSV.
- Si una reproducción no coincide dentro de 1e-9, el análisis se detiene.
"""

from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).resolve().parent

MASTER_CANDIDATES = [
    BASE_DIR / "script_conectividad_IIC_maestro_CORREGIDO_dIIC_solo_funciones.py",
    BASE_DIR / "maestro_GA_para_robustez.py",
    BASE_DIR / "script_conectividad_IIC_maestro_CORREGIDO_dIIC.py",
    BASE_DIR / "script_conectividad_IIC_maestro_CORREGIDO_dIIC(1).py",
]

OUT_DIR = BASE_DIR / "resultados_conectividad_IIC_corregido"

def cargar_master():
    for path in MASTER_CANDIDATES:
        if path.exists():
            spec = importlib.util.spec_from_file_location("master_robustez_90", path)
            master = importlib.util.module_from_spec(spec)
            sys.modules["master_robustez_90"] = master
            spec.loader.exec_module(master)
            return master, path
    raise FileNotFoundError(
        "No se encontró el maestro. Se buscó:\n" +
        "\n".join(str(p) for p in MASTER_CANDIDATES)
    )

def csv_original(porcentaje):
    etiqueta = int(round(porcentaje * 100))
    path = OUT_DIR / f"resultados_ga_{etiqueta}_final.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV requerido: {path}")
    df = pd.read_csv(path)
    return df, path

def iic_inducido(master, individuo):
    seleccionados = [
        master.NODOS[i]
        for i, bit in enumerate(individuo)
        if int(bit) == 1
    ]

    if not seleccionados:
        return 0.0, 0

    sub = master.G_BASE.subgraph(seleccionados).copy()
    dist = dict(nx.all_pairs_shortest_path_length(sub))

    idx = {nodo: i for i, nodo in enumerate(master.NODOS)}
    total = 0.0

    for ni in seleccionados:
        ai = master.AREAS[ni]
        di = dist.get(ni, {})
        for nj in seleccionados:
            if nj in di:
                aj = master.AREAS[nj]
                total += (ai * aj) / (1.0 + di[nj])

    iic = total / (master.AREA_TOTAL ** 2)
    return float(iic), len(seleccionados)

def wilcoxon_one_sample(values, reference):
    differences = np.asarray(values, dtype=float) - float(reference)
    nonzero = differences[np.abs(differences) > 0]

    if len(nonzero) == 0:
        return 0.0, 1.0, 0, "todas las diferencias son cero"

    stat, p = wilcoxon(
        nonzero,
        alternative="greater",
        method="auto"
    )
    return float(stat), float(p), int(len(nonzero)), "Wilcoxon signed-rank, H1: inducido > referencia"

def analizar_presupuesto(master, porcentaje):
    etiqueta = int(round(porcentaje * 100))
    presupuesto = porcentaje * master.AREA_TOTAL

    df, csv_path = csv_original(porcentaje)

    requeridas = {"seed", "fitness"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"{csv_path} no contiene las columnas requeridas: {sorted(faltantes)}"
        )

    if len(df) != 30:
        raise ValueError(
            f"{csv_path} contiene {len(df)} filas; se esperaban exactamente 30."
        )

    df = df.sort_values("seed").reset_index(drop=True)

    ind_h, _ = master.solucion_heuristica(presupuesto)
    iic_h_ind, _ = iic_inducido(master, ind_h)
    fit_h = iic_h_ind / master.IIC_TOTAL * 100.0

    ind_d, _ = master.solucion_dIIC(presupuesto)
    iic_d_ind, _ = iic_inducido(master, ind_d)
    fit_d = iic_d_ind / master.IIC_TOTAL * 100.0

    print("\n" + "-" * 78)
    print(f"ESCENARIO {etiqueta}%")
    print("-" * 78)
    print(f"CSV: {csv_path}")
    print(f"dIIC:       {fit_d:.10f}%")
    print(f"Heurístico: {fit_h:.10f}%")
    print("Reproduciendo 30 seeds y calculando IIC inducido...")

    filas = []

    for _, row in df.iterrows():
        seed = int(row["seed"])
        fitness_original = float(row["fitness"])

        resultado = master.ejecutar_ga(presupuesto, seed)
        fitness_repro = float(resultado["fitness"])
        delta = abs(fitness_repro - fitness_original)

        if delta > 1e-9:
            raise RuntimeError(
                f"REPRODUCCIÓN FALLIDA: escenario {etiqueta}%, seed={seed}. "
                f"CSV={fitness_original:.15f}; "
                f"reproducido={fitness_repro:.15f}; delta={delta:.3e}"
            )

        iic_ind, nsel = iic_inducido(master, resultado["individuo"])
        fitness_ind = iic_ind / master.IIC_TOTAL * 100.0
        retencion = (
            fitness_ind / fitness_original * 100.0
            if fitness_original != 0 else np.nan
        )

        filas.append({
            "presupuesto_%": etiqueta,
            "seed": seed,
            "fitness_base_GA_%": fitness_original,
            "fitness_reproducido_%": fitness_repro,
            "delta_reproduccion": delta,
            "IIC_inducido_%": fitness_ind,
            "retencion_%": retencion,
            "dIIC_%": fit_d,
            "heuristico_%": fit_h,
            "inducido_menos_dIIC_pp": fitness_ind - fit_d,
            "inducido_menos_heuristico_pp": fitness_ind - fit_h,
            "n_fragmentos": nsel,
            "costo_ha": float(resultado["costo"]),
        })

        print(
            f"seed={seed:2d} | OK | base={fitness_original:.8f}% | "
            f"inducido={fitness_ind:.8f}% | ret={retencion:.4f}% | n={nsel}"
        )

    out = pd.DataFrame(filas)

    # Estadística descriptiva
    vals_ind = out["IIC_inducido_%"].to_numpy(float)
    vals_base = out["fitness_base_GA_%"].to_numpy(float)
    vals_ret = out["retencion_%"].to_numpy(float)

    w_d, p_d, n_d, test_d = wilcoxon_one_sample(vals_ind, fit_d)
    w_h, p_h, n_h, test_h = wilcoxon_one_sample(vals_ind, fit_h)

    summary = {
        "presupuesto_%": etiqueta,
        "n": len(out),
        "GA_base_media_%": vals_base.mean(),
        "GA_base_SD_pp": vals_base.std(ddof=1),
        "IIC_inducido_media_%": vals_ind.mean(),
        "IIC_inducido_SD_pp": vals_ind.std(ddof=1),
        "IIC_inducido_mediana_%": np.median(vals_ind),
        "IIC_inducido_min_%": vals_ind.min(),
        "IIC_inducido_max_%": vals_ind.max(),
        "retencion_media_%": vals_ret.mean(),
        "retencion_SD_pp": vals_ret.std(ddof=1),
        "retencion_mediana_%": np.median(vals_ret),
        "retencion_min_%": vals_ret.min(),
        "retencion_max_%": vals_ret.max(),
        "dIIC_%": fit_d,
        "heuristico_%": fit_h,
        "media_inducido_menos_dIIC_pp": vals_ind.mean() - fit_d,
        "media_inducido_menos_heuristico_pp": vals_ind.mean() - fit_h,
        "wilcoxon_dIIC_W": w_d,
        "wilcoxon_dIIC_p": p_d,
        "wilcoxon_dIIC_n_no_cero": n_d,
        "wilcoxon_heuristico_W": w_h,
        "wilcoxon_heuristico_p": p_h,
        "wilcoxon_heuristico_n_no_cero": n_h,
    }

    return out, summary

def main():
    master, master_path = cargar_master()

    print("=" * 78)
    print("ROBUSTEZ — SUBGRAFO INDUCIDO: 90 CORRIDAS")
    print("=" * 78)
    print(f"Maestro: {master_path}")
    print(f"Nodos: {master.N} | Aristas: {master.G_BASE.number_of_edges()}")
    print(f"IIC_TOTAL: {master.IIC_TOTAL:.12f}")
    print(f"Área total: {master.AREA_TOTAL:.12f} ha")

    todas = []
    resumenes = []

    for porcentaje in (0.10, 0.30, 0.50):
        detalle, resumen = analizar_presupuesto(master, porcentaje)
        todas.append(detalle)
        resumenes.append(resumen)

    detalle_final = pd.concat(todas, ignore_index=True)
    resumen_final = pd.DataFrame(resumenes)

    csv_detalle = BASE_DIR / "robustez_subgrafo_inducido_90_corridas.csv"
    csv_resumen = BASE_DIR / "robustez_subgrafo_inducido_90_resumen.csv"
    txt_resumen = BASE_DIR / "robustez_subgrafo_inducido_90_resumen.txt"

    detalle_final.to_csv(csv_detalle, index=False, encoding="utf-8-sig")
    resumen_final.to_csv(csv_resumen, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("PRUEBA DE ROBUSTEZ — SUBGRAFO INDUCIDO")
    lines.append("=" * 78)
    lines.append("")
    lines.append("90 corridas: 30 por presupuesto.")
    lines.append("El individuo de cada seed se reproduce determinísticamente")
    lines.append("con el maestro y se verifica contra el fitness del CSV.")
    lines.append("")
    lines.append("Wilcoxon: IIC inducido de las 30 corridas vs. referencia fija,")
    lines.append("alternative='greater' (H1: IIC inducido > referencia).")
    lines.append("")
    lines.append("RESUMEN")
    lines.append("-" * 78)

    for _, r in resumen_final.iterrows():
        lines.append(f"\nPresupuesto {int(r['presupuesto_%'])}%")
        lines.append(
            f"  GA base:              {r['GA_base_media_%']:.6f} ± "
            f"{r['GA_base_SD_pp']:.6f}%"
        )
        lines.append(
            f"  IIC inducido:         {r['IIC_inducido_media_%']:.6f} ± "
            f"{r['IIC_inducido_SD_pp']:.6f}%"
        )
        lines.append(
            f"  IIC inducido mediana: {r['IIC_inducido_mediana_%']:.6f}%"
        )
        lines.append(
            f"  IIC inducido rango:   {r['IIC_inducido_min_%']:.6f} – "
            f"{r['IIC_inducido_max_%']:.6f}%"
        )
        lines.append(
            f"  Retención:            {r['retencion_media_%']:.6f} ± "
            f"{r['retencion_SD_pp']:.6f}%"
        )
        lines.append(
            f"  Retención mediana:    {r['retencion_mediana_%']:.6f}%"
        )
        lines.append(
            f"  Retención rango:      {r['retencion_min_%']:.6f} – "
            f"{r['retencion_max_%']:.6f}%"
        )
        lines.append(f"  dIIC:                 {r['dIIC_%']:.6f}%")
        lines.append(f"  Heurístico:           {r['heuristico_%']:.6f}%")
        lines.append(
            f"  Inducido - dIIC:      {r['media_inducido_menos_dIIC_pp']:.6f} pp"
        )
        lines.append(
            f"  Inducido - heurístico: {r['media_inducido_menos_heuristico_pp']:.6f} pp"
        )
        lines.append(
            f"  Wilcoxon vs dIIC:     W={r['wilcoxon_dIIC_W']:.6f}, "
            f"p={r['wilcoxon_dIIC_p']:.10g}, "
            f"n={int(r['wilcoxon_dIIC_n_no_cero'])}"
        )
        lines.append(
            f"  Wilcoxon vs heur.:    W={r['wilcoxon_heuristico_W']:.6f}, "
            f"p={r['wilcoxon_heuristico_p']:.10g}, "
            f"n={int(r['wilcoxon_heuristico_n_no_cero'])}"
        )

    lines.append("\n")
    lines.append("DETALLE DE LAS 90 CORRIDAS")
    lines.append("-" * 78)
    lines.append(detalle_final.to_string(index=False))

    txt_resumen.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print("COMPLETADO")
    print("=" * 78)
    print(resumen_final.to_string(index=False))
    print(f"\nCSV detalle: {csv_detalle}")
    print(f"CSV resumen: {csv_resumen}")
    print(f"TXT resumen: {txt_resumen}")

if __name__ == "__main__":
    main()
