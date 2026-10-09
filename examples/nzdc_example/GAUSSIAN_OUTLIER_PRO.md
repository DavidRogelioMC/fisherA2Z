# Gaussian convolution + tomographic outliers

Modelo nuevo: `nz_model="gaussian_outlier_pro"`.
Módulo: `src/fisherA2Z/nz_gaussian_outlier_pro.py`.
Libreta: `examples/nzdc_example/bias_example-main-gaussian-outlier-pro.ipynb`.

## Actualización: `nz_realizations=samples` ya está implementado

La libreta acepta ahora las realizaciones y **no requiere pasar
`photoz_prior_cov`**. Añade `z_realizations=z_native` cuando las realizaciones
y Fisher estén representados sobre mallas diferentes:

```python
flex = FisherFlex(
    nz_source=nz_central, z_grid=z_grid,
    nz_realizations=samples,
    z_realizations=z_native, realizations_axis=1,
    neff_source=neff, fsky=0.5, sigma_e=0.26,
    mode="cosmic_shear", nz_model="gaussian_outlier_pro",
    gaussian_outlier_pro_kwargs=gaussian_config,
    prior_mode="full", deriv_method="central",
)
```

La calibración requiere `gaussian_config["vary_mixing"] = False`.
La matriz de mezcla queda fija, se invierte para obtener las PDFs objetivo
de cada componente, y se ajustan sus desplazamientos/anchos por mínimos
cuadrados integrados en ese espacio. Las PDFs desmezcladas pueden contener
valores negativos: se mantienen como objetivos del ajuste, sin recortarlos.
Se supone que el índice de realización identifica un conjunto tomográfico
de PDFs. No se modifica el modelo de convolución ni se sustituye por gaussianas.

Los fiduciales se reemplazan por **la media de los parámetros ajustados**.
Esto NO es el ajuste a la PDF media ni equivale necesariamente a conservarla.
El prior es la covarianza muestral de los ajustes (ddof=1), por bloques de
dos parámetros por bin; no se divide entre el número de realizaciones.
No se incluyen correlaciones de prior entre bins. El prior calculado está
en `flex.pz_prior_cov`, los ajustes en `flex.realization_params`, y el
informe en `flex.gpro_calibration`.

Con las plantillas actuales, muchos ajustes se sitúan en el ancho mínimo
resuelto por la malla. Se incluyen todos, sin descartar realizaciones ni
inventar un suelo de incertidumbre. Por eso la covarianza es una aproximación
gaussiana a ajustes limitados por una frontera y puede depender de la malla.
El código avisa de la fracción afectada; si el ancho carece de dispersión,
rechaza el prior en lugar de inventar una incertidumbre. La ejecución
correcta NO garantiza que el prior sea una buena aproximación científica.

En `gaussian_config`, `calibration_n_jobs=4` controla los procesos de ajuste.
`calibration_max_realizations=None` usa todas las realizaciones por defecto;
un entero como `1024` selecciona explícitamente una submuestra espaciada y
reproducible, cuya cantidad se informa. La validación del recorrido completo
se hizo con 1024 realizaciones de calibración, usando todas para la media
central. La libreta final queda configurada para todas, con las salidas
antiguas borradas para no atribuirlas a la nueva configuración.

Las secciones siguientes describen también la modalidad de fiduciales/prior
manuales. Los resultados de std_b=1 pertenecen a la ejecución original, antes
de esta calibración, y no son resultados del nuevo prior automático.

## Comparación con el modelo incluido en FisherA2Z

El modelo `gaussian_outlier` existente delega la descomposición en
`nz_decomposition.py`. Para cada bin ajusta un núcleo gaussiano y obtiene
una plantilla de outliers a partir del residuo no negativo de la PDF:

\[
n_i(z)=(1-f_i)\,G(z;\mu_i+\delta z_i,\sigma_i)+f_i\,n_{\mathrm{out},i}(z).
\]

El ajuste perfila la amplitud de la gaussiana y optimiza su centro y ancho;
el residuo determina la fracción y la forma de outliers. Fisher varía tres
parámetros por bin: `zbias`, `zvariance` y `zoutlier`. Pese a su nombre,
`zvariance` representa una **desviación estándar**, no una varianza.
La forma del residuo se mantiene fija mientras cambian esos parámetros.
Con realizaciones, este modelo ajusta cada PDF y mide la covarianza de los
parámetros ajustados.

Tu `Gaussian_with_outliers.py` propone otra construcción: convolucionar
las PDFs de entrada con un núcleo gaussiano y mezclar contribuciones de
otros bins. No sustituye cada PDF por una gaussiana pura.

| Aspecto | `gaussian_outlier` existente | `gaussian_outlier_pro` nuevo |
|---|---|---|
| Distribución de partida | Gaussiana ajustada + residuo del mismo bin | PDF de entrada completa |
| Operación gaussiana | Reconstrucción del núcleo | Convolución de la plantilla |
| Ancho | Ancho del núcleo gaussiano | Dispersión adicional del núcleo de convolución |
| Outliers | Residuo fijo del propio bin | Otras plantillas tomográficas convolucionadas |
| Fracciones | Una `f_out` por bin | Una fila de `c[i,j]` por bin |
| Interacción entre bins | No mezcla explícita | Una plantilla puede contribuir a varios bins |
| Parámetros libres | Tres por bin | Dos, o dos más `n_bins-1` fracciones por bin |
| Prior desde `samples` | Ajuste de cada realización | Ajuste de realizaciones con mezcla fija |

Ambos modelos producen PDFs normalizadas, permiten variaciones de centro/
anchura y describen contaminación, pero sus parámetros y priors no son
intercambiables. El modelo existente y `nz_decomposition.py` no se reemplazan.

## Qué calcula el modelo nuevo

Para cada plantilla normalizada \(n_j(t)\):

\[
\widetilde g_j(z)=\int_{z_{\min}}^{z_{\max}}
n_j(t)\frac{\exp[-(z-t-\Delta z_j)^2/(2s_j^2)]}{\sqrt{2\pi}\,s_j}\,dt,
\qquad
g_j(z)=\frac{\widetilde g_j(z)}{\int_{z_{\min}}^{z_{\max}}\widetilde g_j(u)\,du}.
\]

Después:

\[
n_i^{\rm model}(z)=\left(1-\sum_{j\ne i}c_{ij}\right)g_i(z)
                    +\sum_{j\ne i}c_{ij}g_j(z).
\]

- `dz`: desplazamiento del núcleo, positivo hacia redshift mayor.
- `std_b`: \(s_j>0\), desviación estándar del núcleo en unidades de redshift.
  **`std_b=1` no conserva la anchura de la PDF: añade una convolución muy ancha.**
- `c[i,j]`: fracción de la PDF de salida `i` suministrada por la plantilla `j`.
  La diagonal es cero, los elementos son no negativos y la suma por fila no
  supera uno. Para variar independientemente las fracciones, el fiducial debe
  tener suma de fila estrictamente menor que uno.
- `neff`: entrada independiente del sondeo. Esta convención de mezcla cambia
  las PDFs condicionales por bin; no traslada conteos ni recalcula densidades.

La normalización se impone en el soporte finito suministrado. La PDF es cero
fuera de ese soporte. Por ello los desplazamientos/anchos también pueden
alterar la normalización previa al recorte: no se deben interpretar como
cambios idénticos en la media/desviación estándar final cerca de los extremos.

La integración usa trapecios con la malla real, incluidos nodos no uniformes.
La malla debe resolver el núcleo; el código rechaza anchuras inferiores al
umbral de resolución indicado en el mensaje de error. El refinamiento de la
malla y la estabilidad de las derivadas siguen siendo comprobaciones necesarias.

## Fidelidad al archivo original y correcciones

Se conserva la secuencia y los nombres útiles: `gaussian_int`, `get_dnGdz`,
`reconstructed_pdf`, `outlier_fraction` y `run`. `run(nzs, zs, Nbins, dz,
std_b, c)` conserva su interfaz y devuelve una malla y funciones interpoladas.
Las firmas internas se simplificaron al eliminar argumentos sin uso.
No se copian `get_moments`, el generador aleatorio ni imports que no intervenían
en el cálculo. Se usa NumPy/SciPy, sin requerir JAX.

No es una copia numérica literal: el archivo suministrado tenía estos problemas:

1. `get_dnGdz` rellenaba solo una fila, pero la mezcla leía otras filas que no
   habían sido calculadas. Ahora se calculan todas antes de mezclar.
2. `reconstructed_pdf` sumaba los coeficientes de toda `c` para cada salida.
   Ahora usa exclusivamente la fila correspondiente al bin reconstruido.
3. El exponente usaba `std_b**2`, pero el prefactor era
   `1/sqrt(2*pi*std_b)`. Para una desviación estándar debe ser
   `1/(sqrt(2*pi)*std_b)`.
4. `gaussian_int` sumaba valores sin pesos de redshift. Ahora calcula la
   integral. Se normaliza cada componente antes de la mezcla para que `c`
   represente fracciones de probabilidad en el soporte elegido.
5. `outlier_fraction` devolvía una suma global aunque decía “per bin”. Ahora
   devuelve una fracción por fila.
6. La interpolación original extrapolaba fuera del soporte. Ahora devuelve cero.

El archivo original cita [Schaan, Ferraro y Seljak (2020)](https://arxiv.org/abs/2007.12795).
Esta adaptación conserva y corrige la estructura del código suministrado;
no pretende reproducir todos los detalles y resultados de ese artículo.

## Uso en Fisher

```python
import numpy as np
from fisherA2Z.fisher_flex import FisherFlex

c = np.full((n_bins, n_bins), 0.01 / (n_bins - 1))
np.fill_diagonal(c, 0)

flex = FisherFlex(
    nz_source=nz_central, z_grid=z_grid,
    neff_source=neff, fsky=0.5, sigma_e=0.26,
    nz_realizations=None, photoz_prior_cov=None,
    mode="cosmic_shear", nz_model="gaussian_outlier_pro",
    gaussian_outlier_pro_kwargs=dict(
        dz=0.001, std_b=1.0, c=c, vary_mixing=False,
    ),
    deriv_method="central",
)
flex.compute(parallel=True, n_proc=2)
res = flex.forecast(ell_min_cs=300, ell_max_cs=1800)
```

El ejemplo de construcción de `c` supone al menos dos bins; para uno usa
`np.zeros((1, 1))`. Los escalares `dz` y `std_b` se repiten como fiduciales
en todos los bins, pero Fisher varía los parámetros de cada plantilla por separado.

Por bin, el orden del prior es el de `parameter_names(i, n_bins, vary_mixing)`:
`gdz1`, `gsigma1`, y, si se liberan, `gc1_2`, `gc1_3`, ... para el primer bin.
Con cinco bins, `photoz_prior_cov` tiene forma `(5, 2, 2)` con mezcla fija o
`(5, 6, 6)` con mezcla libre. Los bloques pueden ser correlacionados y deben ser
positivos definidos. No representan correlaciones de prior entre distintos bins.

`photoz_prior_cov=None` significa **sin información foto-z externa**: los
desplazamientos/anchos siguen siendo libres. Los priors cosmológicos y de IA
habituales siguen activos. Si se proporcionan `nz_realizations`, se activa la calibración automática
descrita al inicio. Si se omiten tanto las realizaciones como el prior,
los parámetros permanecen libres sin información foto-z externa.

`nz_source` debe contener las plantillas antes de la convolución. La PDF
fiducial que realmente entra en los espectros se encuentra en
`flex.nz_source_fid`. `forecast_bias` compara la verdad contra esa PDF.
Guardar/cargar conserva las plantillas originales, los fiduciales, la matriz
de mezcla y la elección de parámetros libres. Se rechaza reutilizar derivadas
de otro modelo o de una configuración distinta.

## Configuración comprobada en la libreta

- Archivo: `source_year1_outliers_OFF.npz`, 199499 realizaciones por cada uno
  de cinco bins. Plantillas: media aritmética normalizada del ensemble completo.
- Fiduciales del ejemplo original: `dz=0.001`, `std_b=1`, mezcla total de 1% por fila.
- Punto de partida: `VARY_MIXING=False`; diez parámetros foto-z libres y sin
  prior foto-z externo. `c` sigue mezclando los bins, pero queda fija.
- Con los 30 parámetros foto-z libres y sin prior, la prueba dio una matriz
  escalada de condición aproximada `6.9e16`; el código ahora rechaza ese
  forecast numéricamente degenerado. Con mezcla fija, la condición fue `7.6e5`.
- Se mantienen la densidad Cardinal 1yr y las entradas del sondeo de las
  libretas previas; deben corresponder a la selección científica que se estudie.
- Se mantienen `predicted bias`, `bias_2d`, `corner_arrows` y `contour` + `arrow`.

Con estos valores, el residuo medio fraccional de los espectros es aproximadamente
60% y el bias lineal predice Omega_m negativa. Esto muestra que ese punto fiducial
no es adecuado para interpretar el bias como un ajuste cosmológico cuantitativo.
Las flechas se conservan como diagnóstico y se explica por qué no se calcula
un S8 desplazado físicamente inadmisible. Calibrar valores/priores requiere
información adicional; el código no los sustituye automáticamente por supuestos.

## Archivos y validación

Se añadió el módulo y su libreta; se extendió `fisher_flex.py` con una rama
independiente, parámetros, estado de los procesos, priors, persistencia y control
de degeneraciones. El formato pasa a versión 5 y sigue leyendo versiones 2–4.
En pruebas de momentos solo se actualizó la comparación del número de versión.

La validación incluye casos analíticos de convolución, mezcla asimétrica,
normalización, priors, límites físicos de las derivadas, guardado/carga,
compatibilidad de cuantiles/momentos y comprobación del modelo original sin cambios.
También se ejecutó la libreta completa y se comprobó la convergencia del paso
para un desplazamiento y un ancho, la coincidencia serial/paralela de esas
derivadas y el bias nulo cuando verdad y fiducial coinciden.
