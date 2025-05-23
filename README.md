# Lego-Maker
Aplicación de python diseñada para transformar imágenes en rompecabezas LEGO, con fines educativos y recreativos.

En este proyecto, presento el desarrollo de una aplicación de software diseñada para transformar imágenes en rompecabezas LEGO, con fines educativos y recreativos. El proyecto busca fomentar la creatividad al permitir a los usuarios visualizar y construir sus propios rompecabezas LEGO a partir de imágenes personales. Detallo los pasos del procesamiento de imágenes, incluyendo la inicialización del tamaño, la simplificación del color y la conversión de imágenes a una paleta de colores LEGO compatible. Además, analizo cómo se identifican las áreas de color significativas y se segmentan en regiones definidas por LEGO, lo cual es crucial para mantener la precisión del rompecabezas. El proceso de construcción implica transformar estas áreas en formas básicas y optimizar el número de piezas para su ensamblaje. Se incluye un análisis de costes para estimar el presupuesto para la creación del rompecabezas, junto con un listado de las piezas necesitadas.

Palabras clave: Procesamiento de imágenes, rompecabezas LEGO, desarrollo de software, análisis de color, estimación de costos.

El código implementa una serie de algoritmos y técnicas de procesamiento de imágenes para transformar una imagen dada en un puzzle LEGO. La elección de estos algoritmos es crucial para la precisión, eficiencia y calidad del resultado final. A continuación, se detallan las decisiones:

Para la simplificación del Color nos hemos basado en el Algoritmo K-means:

Este algoritmo se utiliza para reducir la cantidad de colores en la imagen original. K-means es eficaz para agrupar píxeles con colores similares en un número predefinido de clústeres. El algoritmo K-means es computacionalmente eficiente y simple de implementar. Permite una buena reducción de la paleta de colores, lo cual es necesario para mapear los colores de la imagen a los colores limitados de LEGO.

Mapeo a Colores LEGO, para el mapeo nos hemos basado en el mapeo por distancia en el espacio de color HSV: El código convierte los colores simplificados y los colores LEGO al espacio de color HSV (Tono, Saturación, Valor) y luego mapea cada color simplificado al color LEGO más cercano, ponderando la importancia de cada componente (H, S, V).

He usado este método ya que el espacio HSV es más intuitivo para la comparación de colores que el espacio RGB, ya que separa la información del color (tono y saturación) de la información de la luminosidad (valor). La ponderación permite ajustar la sensibilidad del mapeo a diferentes aspectos del color.

Para la detección de regiones de color hemos utilizado la Umbralización Adaptativa y Watershed. Esta utiliza la umbralización adaptativa (método de Otsu) para separar el objeto principal del fondo, seguida de operaciones para refinar la segmentación. Luego, se aplica el algoritmo Watershed para una mejor separación de las regiones de color. La umbralización adaptativa es robusta a las variaciones de iluminación en la imagen. Watershed es eficaz para separar objetos que se tocan o superponen, lo cual es común en imágenes complejas.

La transformación a Formas Básicas la hemos basado en descomposición en rectángulos: Las regiones de color se descomponen en rectángulos para representar las formas básicas que se llenarán con piezas LEGO. Se utiliza un enfoque de "dividir y conquistar" para descomponer regiones complejas en rectángulos más pequeños.

Justificación: Los rectángulos son la forma básica de las piezas LEGO, por lo que esta descomposición es natural para el problema.

Para el relleno con Piezas LEGO usamos un algoritmo de colocación de piezas optimizado: Se utiliza un algoritmo personalizado para colocar piezas LEGO dentro de los rectángulos, priorizando piezas más grandes para cubrir más área y piezas más pequeñas para los detalles.

Para la agrupación de áreas contiguas hemos implementado una función para optimizar el uso de piezas LEGO agrupando áreas contiguas del mismo color en la imagen. Se ha añadido un borde negro en cada una de las piezas para tener mayor facilidad para montar el puzle.
