# Modelo continuo de momentos en FisherA2Z

Actualización del 9 de octubre de 2026. Reinicia el kernel de Jupyter para
cargar esta versión de `nz_moments.py` antes de volver a ejecutar el cálculo.

## Reconstrucción de máxima entropía

La distribución ahora tiene la misma forma continua que en `pymaxent_jax.py`:

```text
n(z) = exp(lambda_0 + lambda_1*z + ... + lambda_K*z**K)
```

Las funciones `integrand`, `residual_c`, `maxent_reconstruct_c` y `reconstruct`
siguen el camino continuo de ese módulo. `scipy.optimize.fsolve` calcula los
multiplicadores imponiendo los momentos. Las integrales del solucionador usan
trapecios en 1000 puntos del soporte, igual que la referencia.

Se utiliza NumPy en lugar de JAX, sin añadir dependencias. Se aporta el Jacobiano
analítico a fsolve y se comprueban los residuos antes de aceptar la solución.
La implementación anterior de entropía discretizada y su base de Legendre
se eliminaron. La cuadratura de Gauss-Legendre que aún aparece en
`_moment_weights` sirve SOLO para medir los momentos de las PDFs de entrada;
no es una base ni un solucionador de la reconstrucción de máxima entropía.

## Diferencias deliberadas respecto a pdf_maxent.py

1. Fisher mantiene m0=1 y varía únicamente los momentos de orden positivo.
2. Se conserva `raw_moments`: integra exactamente la PDF de entrada normalizada
   y lineal entre nodos. Así los momentos y el prior no cambian al refinar esa
   PDF por interpolación lineal. No se copia la suma rectangular `(b-a)/len(z)`
   de pdf_maxent.py. La fidelidad de la reconstrucción se compara suministrando
   los mismos momentos y soporte a ambos solucionadores.
3. Se reutilizan los multiplicadores fiduciales como punto inicial para pequeñas
   variaciones. Las soluciones que no satisfacen las restricciones se rechazan.
4. `run` recibe un vector de desplazamientos por bin y usa cero fuera del soporte
   en los interpoladores. Corrige `xs.append(xs)` a `xs.append(x_0)`.

La normalización y las restricciones se comprueban con la cuadratura interna
del solucionador. Una evaluación con pocos puntos de salida puede dar una
integral trapezoidal distinta de uno: es error de muestreo de la función continua.
No se renormaliza esa salida artificialmente, porque se alterarían los momentos.
Usa una malla suficientemente fina para CCL y comprueba convergencia.

## Ver los multiplicadores

```python
import numpy as np
from fisherA2Z import nz_moments as nzm

modelo = nzm.build_moment_model(
    z_grid, nz_central, n_moments=[7, 8, 4, 4, 4]
)
print(modelo.lambdas[0])  # lambda_0,...,lambda_6 del primer bin

b = 0
nz, lambdas = nzm.evaluate_moment_nz(
    z_grid, modelo.moments[b], np.zeros(modelo.n_moments[b] - 1),
    return_lambdas=True,
)

# La misma interfaz principal que pymaxent_jax, si ya tienes los momentos:
sol, lambdas = nzm.reconstruct(
    modelo.moments[b], bnds=[z_grid[0], z_grid[-1]]
)
nz = sol(z_grid)
```

`lambdas` son coeficientes de potencias del redshift físico z, incluida la
constante lambda_0. Los parámetros que Fisher marginaliza continúan siendo los
desplazamientos de momentos `mshift`, no los multiplicadores.

## Uso con Fisher

La integración está en `fisher_flex.py`. Se mantienen los 7, 8, 4, 4 y 4 momentos
totales de los cinco bins: 6, 7, 3, 3 y 3 parámetros libres, 22 en total.
El segundo momento es E[z**2], no la varianza; la varianza es m2-m1**2.

```python
from fisherA2Z.fisher_flex import FisherFlex

flex = FisherFlex(
    nz_source=nz_central, z_grid=z_grid,
    nz_realizations=samples, realizations_axis=1,
    neff_source=neff, fsky=0.5, sigma_e=0.26,
    mode="cosmic_shear", nz_model="moments",
    n_moments=[7, 8, 4, 4, 4], prior_mode="full",
)
flex.compute(parallel=True, save="fisher_moments_continuo.npz")
res = flex.forecast(ell_min_cs=300, ell_max_cs=1800)
```

Aquí `samples` tiene forma `(bins, realizaciones, z)` y debe estar en la misma
malla que `z_grid`. Si la libreta libera samples y refina la malla, usa en su
lugar `nz_realizations=None, photoz_prior_cov=photoz_prior_cov`, como ya hace
`Prueba_momentos_Fisher.ipynb`. No es necesario interpolar el ensemble completo.
El prior sigue midiendo la dispersión de momentos por bin, con correlaciones
entre órdenes y sin correlaciones entre bins. No se divide por el número de
realizaciones. Las densidades efectivas y el área del sondeo son entradas
independientes de la reconstrucción de n(z).

## Resultados anteriores y comprobaciones

El prior puede reutilizarse si corresponden los mismos datos, momentos y
convención de integración, que se conservaron. Debes recalcular los espectros,
derivadas, Fisher y bias: los archivos Fisher del modelo anterior ya no son
compatibles. `fisher_flex.py` guarda `moment_model_version` y rechaza mezclarlos.
Las curvas y salidas guardadas anteriormente en la libreta pertenecen al modelo
anterior hasta ejecutar nuevamente sus celdas. Las notas antiguas que describen
una reconstrucción discretizada quedan sustituidas por esta explicación.

Pasaron 25 pruebas de reconstrucción, conservación de momentos, comparación
con una traducción NumPy de la rama continua original, derivadas, priors,
serialización e integración. No se ejecutó JAX porque falta jaxlib en fisher_env.
Se comprueban además los cinco bins de datos reales y su integración con CCL;
el informe técnico de la ejecución se conserva en la carpeta de trabajo de
esta conversación (`continuous_moments_20261008/fisher_validation.json`).

Los pasos numéricos ponderan la respuesta de log n(z) por la propia PDF para
evitar que colas diminutas produzcan pasos demasiado pequeños. `step` controla
esa escala local; no es un desplazamiento común de 0.001 en todos los momentos.
Comprueba los pasos y la malla antes de interpretar nuevos datos u órdenes.
Con pocos momentos, máxima entropía no reproduce necesariamente un pico estrecho.

Las copias anteriores de los archivos modificados terminan en
`.before_continuous_maxent_20261008.bak`. Los módulos de referencia permanecen
intactos y la atribución del código adaptado está en `MAXENT_NOTICE.txt`.
