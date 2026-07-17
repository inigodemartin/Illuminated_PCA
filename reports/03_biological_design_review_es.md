# Revisión de Diseño Biológico y Direcciones Futuras

Memorando científico confidencial — Herramienta FANTASIA PCA

**De:** Dra. S. Lindström (Investigadora postdoctoral, Genómica Evolutiva) | **Rol:** Interpretación Biológica e Innovación | **Fecha:** 30 de junio de 2026

## Resumen ejecutivo

La herramienta visualiza con éxito la diversidad funcional del proteoma entre especies, pero se apoya en tres supuestos estadísticos biológicamente cuestionables y uno demostrablemente incorrecto (tratar datos composicionales como multivariantes estándar). Estos problemas afectan a la interpretabilidad de los ejes y las cargas (loadings) del PCA. Se proponen cinco nuevas funcionalidades ordenadas por impacto científico, junto con visualizaciones alternativas y una hoja de ruta para el análisis evolutivo.

## 1. Deficiencias en el diseño biológico

**CRÍTICO — Sesgo por datos composicionales (Aitchison 1986)**

Los datos de abundancia relativa (count / total_prots) son composicionales: todas las variables suman una constante. El PCA estándar sobre datos composicionales produce correlaciones de Pearson espurias — si la especie A está enriquecida en la función X, aparece matemáticamente empobrecida en todo lo demás, generando anticorrelaciones artefactuales. La transformación correcta es el Log-Ratio Centrado (CLR): log(x_i / media_geométrica(x)). Es un cambio de ~3 líneas y haría que las cargas del PCA fueran interpretables como contrastes funcionales reales, en lugar de artefactos de la composición.

**ALTO — El filtro de IC está sesgado hacia organismos modelo**

Los valores de Contenido de Información (IC) de GO se derivan de la frecuencia de anotación en todas las especies de la base de datos GO, la cual está masivamente sesgada hacia humano, ratón, levadura y Arabidopsis. Un término GO específico de invertebrados marinos tendrá un IC artificialmente alto (pocas anotaciones) pese a ser funcionalmente ubicuo dentro de ese clado. Solución: calcular el IC empírico a partir de la propia matriz de datos del usuario: IC_empírico(GO) = -log2(fracción de especies con ese GO). Esto es centrado en los datos y filogenéticamente imparcial.

**MEDIO — La normalización por total_prots confunde abundancia con número de copias**

count_GO / total_prots mide la abundancia por número de copias de los productos génicos anotados, no la presencia de funciones biológicas. Las especies con expansiones masivas de familias génicas (p. ej., plantas) parecen funcionalmente "más ricas" en esos términos GO aunque la actividad biológica sea idéntica. El PCA de presencia/ausencia (ya implementado) es robusto frente a esto; el PCA de abundancia debería ofrecer un tope `--max-copies-per-go N`.

**BAJO — StandardScaler amplifica los términos GO raros**

Dividir por la desviación estándar amplifica los términos GO presentes en muy pocas especies con alta varianza, lo que puede hacer que términos raros dominen los primeros componentes principales. RobustScaler (mediana/IQR) o una transformación log1p antes del escalado reducirían este efecto sin depender del actual filtro rígido de suma>5.

## 2. Nuevas funcionalidades propuestas (por impacto científico)

**1. Superposición de árbol filogenético**

Si se proporciona un árbol Newick de las ~1200 especies, las ramas pueden proyectarse sobre el gráfico de dispersión del PCA: los nodos internos se posicionan como la media ponderada de sus descendientes, y las aristas se dibujan como líneas. Esto convierte el PCA en un mapa macroevolutivo, permitiendo visualizar directamente la congruencia filo-funcional, la compacidad de los clados y las convergencias (ramas filogenéticamente distantes que se encuentran en el espacio funcional). Habilita el análisis posterior de la K de Blomberg sobre la señal filogenética en PC1/PC2.

**2. Análisis estadístico diferencial de GOs entre taxones**

Extiende `-t/--taxa`: un test de Mann-Whitney U por cada término GO entre dos grupos taxonómicos seleccionados, con corrección FDR de Benjamini-Hochberg. Salida: tabla de GOs diferencialmente abundantes con valor p, fold-change y su carga en PC1/PC2. Convierte el PCA de una herramienta exploratoria a una generadora de hipótesis.

**3. Biplot interactivo (vectores de carga sobre el gráfico)**

Superponer los N términos GO con mayor carga como flechas que parten del origen del PCA, escaladas según la magnitud de la carga. Al pasar el cursor sobre una flecha se muestra el término GO y su IC; al hacer clic se iluminan esas especies. Esta es la interpretación biológica estándar del PCA y haría que el significado de los ejes fuera inmediatamente legible sin necesidad de la barra lateral.

**4. Detección automatizada de evolución convergente**

Dado un árbol filogenético (funcionalidad 1): marcar pares de especies con distancia filogenética > umbral pero distancia en el PCA < umbral. Estos son candidatos de convergencia funcional. Para cada par, reportar los enriquecimientos de GO compartidos en relación con sus respectivos clados.

**5. Modo de embedding UMAP / t-SNE**

El PCA captura varianza lineal; la reducción de dimensionalidad no lineal revela estructura de clústeres que el PCA aplana. Añadir `--method [pca|umap|tsne]` usando la misma matriz de entrada normalizada y la misma plantilla HTML requeriría un código adicional mínimo. Advertencia crítica: las distancias de UMAP no son interpretables globalmente — esto debe comunicarse claramente en el título de la salida.

## 3. Hoja de ruta de análisis evolutivo

Los siguientes análisis se vuelven posibles una vez implementada la superposición del árbol filogenético (funcionalidad 1):

- K de Blomberg sobre PC1/PC2: cuantificar la señal filogenética frente a la convergencia adaptativa.
- Sinapomorfías funcionales por clado: términos GO presentes en ≥95% de los miembros del clado pero en ≤10% fuera de él.
- Dispersión del PCA dentro de cada clado como indicador de la tasa de evolución funcional (rápida en parásitos, lenta en cianobacterias).
- Distancia al centroide como "índice de derivación funcional" — cuánto se ha desviado cada especie del proteoma ancestral.
- Detección de reducción genómica: especies a las que les faltan términos GO presentes en ≥80% de su clado (endosimbiontes, parásitos obligados).

## 4. Visualizaciones alternativas

- **Gráfico de volcán de cargas**: X = carga en PC1, Y = valor de IC. Los GOs en el cuadrante superior derecho (carga alta + IC alto) son los más discriminantes y específicos. Revela de inmediato si el umbral de IC elegido es adecuado.
- **Mapa de calor de la matriz de GOs ordenado por PCA**: filas (especies) ordenadas por puntuación de PC1, columnas (términos GO) ordenadas por carga en PC1. Revela de un vistazo gradientes de covariación y clústeres de especies. Exportable como SVG.
- **Red de similitud funcional**: grafo dirigido por fuerzas donde los nodos son especies y las aristas conectan pares con distancia euclídea del PCA por debajo de un umbral. Las aristas entre linajes distintos son candidatas a convergencia evolutiva.
- **PCA 3D (PC1 × PC2 × PC3)**: TruncatedSVD(n_components=3) es trivial; una rotación interactiva en WebGL en la plantilla revelaría estructura oculta en las proyecciones 2D.

## 5. Comunicar los resultados a personas no bioinformáticas

El modal de contribución de especies actualmente muestra valores numéricos brutos (abundancia_normalizada × carga) que carecen de un significado biológico intuitivo. Tres mejoras harían los resultados accesibles a colaboradores sin formación cuantitativa:

- Sustituir el valor de contribución por el percentil de abundancia: "Esta especie está en el percentil 94 para este término GO entre todas las especies."
- Añadir una comparación con la media del grupo taxonómico: "Por encima / En / Por debajo de la media de su grupo taxonómico."
- Codificación visual tipo semáforo: punto verde (25% superior), amarillo (50% intermedio), rojo (25% inferior) — legible de un vistazo.
