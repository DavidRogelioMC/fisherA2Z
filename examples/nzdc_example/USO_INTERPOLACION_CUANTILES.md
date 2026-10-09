# Elegir el orden de la reconstrucción por cuantiles

El número corresponde al grado de la PDF **n(z)**, según lo acordado:

| Número | n(z) | Construcción |
| --- | --- | --- |
| 0 | Constante por tramos | CDF lineal; reproduce la reconstrucción escalonada. |
| 1 | Lineal por tramos, continua | CDF cuadrática por tramos, con un punto intermedio en cada intervalo de cuantiles. |
| 2 | Cuadrática por tramos, continua | Derivada de la CDF cúbica PCHIP; conserva el método anterior de v1. |
| 3 | Cúbica por tramos | Derivada de un spline de grado 4 para la CDF. |
| 4 | Cuártica por tramos | Derivada de un spline de grado 5 para la CDF. |
| k >= 3 | Grado k | Spline interpolante para la CDF de grado k+1. |

Con diez niveles, los órdenes disponibles van de 0 a 8. Esto no significa
que todos produzcan una PDF válida para cualquier conjunto de cuantiles.
Los grados se entienden como grados máximos: por ejemplo, unos cuantiles
uniformes reconstruyen una PDF constante para todos los órdenes.

## Número de niveles: cinco en bias_example-main-v2.ipynb

El número de niveles se configura por instancia y es independiente del orden
de interpolación y del número de bins tomográficos. La libreta v2 utiliza:

```python
N_QUANTILES = 5
N_QPARAMS = N_QUANTILES - 2  # tres parámetros por bin
```

Usa `levels = np.linspace(0., 1., N_QUANTILES)`, `np.zeros(N_QPARAMS)` en
la gráfica y `n_quantiles=N_QUANTILES` en FisherFlex. No asignes valores a
`nzq.N_QUANTILES` o `nzq.N_PARAMS`: son valores predeterminados del módulo,
y Python no recalcula uno al cambiar el otro. El módulo activo v1 ahora
admite un número configurable de niveles, como v2.

Con cinco bins y cinco niveles, las posiciones tienen forma `(5, 5)`, el
prior `(5, 3, 3)` y hay 15 desplazamientos libres de cuantiles en total.
Al cambiar el número de niveles, vuelve a medir el prior de las realizaciones;
no reutilices el prior de ocho posiciones libres ni recortes su matriz.
Con cinco niveles se admiten los órdenes 0, 1, 2 y 3; el 3 puede producir
una PDF negativa y será rechazado si ocurre.

## En FisherFlex

Reinicia el kernel de Jupyter después de actualizar los módulos. Crea una
nueva instancia y vuelve a calcular las derivadas al cambiar de orden:

```python
from fisherA2Z.fisher_flex import FisherFlex

N_QUANTILES = 5
ORDEN_NZ = 1  # 0 constante; 1 lineal; 2 cuadrática; 3 cúbica; 4 cuártica...

flex = FisherFlex(
    nz_source=nz_central,
    nz_realizations=samples,
    realizations_axis=1,  # samples tiene forma (bins, realizaciones, z)
    z_grid=z_grid,
    photoz_prior_cov=None,
    neff_source=neff,
    fsky=0.5,
    sigma_e=0.26,
    mode="cosmic_shear",
    nz_model="quantiles",
    n_quantiles=N_QUANTILES,
    quantile_interpolation_order=ORDEN_NZ,
)
flex.compute(parallel=True, save=f"fisher_quantiles_orden{ORDEN_NZ}.npz")
res = flex.forecast(ell_min_cs=300, ell_max_cs=1800)
```

El orden viaja con el estado que reciben los procesos de cálculo. Se guarda
en el NPZ junto con una identificación del método de interpolación. Se
rechaza reutilizar derivadas de otro orden. Los NPZ antiguos de Fisher que
no identifican el método deben recalcularse; esto no afecta a los archivos
de datos de entrada ni a los archivos que contienen solamente el prior.

## En las gráficas y llamadas directas

Actualmente `fisher_flex.py` importa **nz_quantiles_v1.py**. Ese es el módulo
que debes importar para usar el mismo modelo:

```python
from fisherA2Z import nz_quantiles_v1 as nzq
import numpy as np

nz_quant = nzq.evaluate_quantile_nz(
    z_plot, flex.quantile_levels, flex.quantile_locs[i],
    np.zeros(flex.n_pz_per_bin),
    interpolation_order=flex.quantile_interpolation_order,
)
```

`z_plot` debe tener los mismos extremos que la malla usada para calcular las
posiciones de los cuantiles. También puedes pasar `interpolation_order=1`
directamente sin una instancia de FisherFlex.

Si prefieres cambiar el valor predeterminado en el módulo, edita una línea
cerca del principio de **nz_quantiles_v1.py**:

```python
INTERPOLATION_ORDER = 1
```

El valor instalado por defecto es `2`, para conservar el PCHIP que ya usaba
v1. `nz_quantiles_v2.py` también admite el nuevo argumento; conserva su
valor predeterminado anterior, `0`. FisherFlex sigue importando v1.
Un argumento explícito tiene prioridad sobre el valor predeterminado.
Cambiar el predeterminado no modifica instancias de FisherFlex ya creadas.

## Conservación de los cuantiles y PDFs negativas

El interpolador construye una CDF F(z) tal que **F(q_j) = levels[j]**.
La densidad continua correspondiente es **n(z) = dF/dz**. Por tanto,
la integral entre dos cuantiles consecutivos conserva su masa exacta.
Para entregar datos a CCL se mantienen los promedios de densidad en celdas
del código anterior; su integral trapezoidal en la malla de salida es uno.
La interpolación posterior de CCL aproxima la curva continua: hay que
comprobar la convergencia con la resolución en z para un análisis científico.

Para el orden 1, una única recta por intervalo no siempre permite conservar
las masas y mantener n(z) >= 0. Se añade un punto intermedio de reconstrucción
por intervalo. Su altura se calcula para conservar la masa, sin añadir
cuantiles, parámetros libres ni elementos al prior.

Los órdenes 0, 1 y 2 garantizan una reconstrucción no negativa para cuantiles
válidos. Para órdenes >= 3, un spline ordinario puede oscilar: se examinan
sus mínimos, incluso entre los nodos de evaluación. Si produce valores
negativos significativos, se lanza ValueError. No se recorta la PDF ni se
sustituye silenciosamente el orden elegido.

En las cinco distribuciones centrales de `source_year1_outliers_OFF.npz`
conservadas en el archivo del prior de momentos, usando diez niveles, los órdenes 0, 1 y 2 son
válidos; los órdenes 3 a 8 producen valores negativos. En distribuciones
suaves de prueba sí funcionan los órdenes 3 a 8. Un orden alto puede ser
válido en el fiducial y fallar tras desplazar un cuantíl al calcular una
derivada; el mismo control se aplica a cada evaluación.

La extracción de posiciones de cuantiles y la covarianza estimada a partir
de las realizaciones no dependen de este orden. Con diez niveles siguen
existiendo ocho posiciones interiores libres por bin y dos extremos fijos.

## Comprobaciones realizadas y precisión de las derivadas

Pasaron 41 pruebas específicas: grados de interpolación, conservación de
probabilidad, positividad, prior, serialización y regresiones de momentos.
Además se completó un cálculo Fisher/CCL con las cinco PDFs centrales,
orden 1, 1249 nodos en z, 61 parámetros y dos procesos. El guardado/carga y
el control de bias cero funcionaron; las dos derivadas comprobadas
coincidieron exactamente entre serial y paralelo. Esa prueba empleó neff y
prior sintéticos, y no representa el forecast científico del sondeo.

Al variar el paso de diferencias finitas de 0.001 a la mitad y al doble,
`qshift1_1` cambió como máximo un 0.141%, pero `qshift8_5` cambió un 13.6%
y un 4.52%, respectivamente. Esa sensibilidad también aparece con el
PCHIP anterior (orden 2: 13.9% y 3.34%) y con el orden 0 (3.58% y 10.1%).
Refinar la malla a 4993 nodos en el orden 1 no la eliminó: 4.17% y 5.86%.
Estos porcentajes comparan las derivadas, no los errores de parámetros.
No se modificaron los pasos de Fisher ni las tolerancias internas de CCL
como parte del cambio de interpolación.

Antes de usar los contornos como resultados científicos, comprueba la
estabilidad de las derivadas y del forecast frente al paso y a la malla.
El diagnóstico existente se puede ejecutar, por ejemplo, así:

```python
flex.check_step_convergence("qshift8_5", factors=(0.5, 1.0, 2.0))
```
