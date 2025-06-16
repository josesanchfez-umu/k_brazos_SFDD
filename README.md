# Problema del Bandido de k-Brazos

## Información
- **Alumnos:** Sánchez Fernández, Jose Antonio; Díaz Díaz, Tomás;
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** SFDD

## Descripción
Esta práctica está enfocada al aprendizaje reforzado, donde se analizan los algoritmos del bandido de k-brazos, comparando su rendimiento en entornos estacionarios.

## Estructura
El repositorio se estructura de la siguiente forma:

- En la carpeta raíz del proyecto se encuentran los siguientes ficheros:
    - Notebooks correspondientes a los estudios del gradiente en sus versiones para Bernoulli, Binomial y Normal, denominados `gradiente_{variante}.ipynb`.
    - Idem para los notebooks de UCB, denominados `ucb_{variante}.ipynb`.
    - Notebook con el estudio $\epsilon$-greedy en `greedy_experiment.ipynb`.
    - `main.ipynb`, en el que se encuentra la descripción, estructura y explicación del proyecto, junto a una serie de enlaces a cada uno de los notebooks y un enlace para abrir el fichero en Colab.
    - `requirements.txt` utilizado para la creación del entorno virtual y ejecutar la práctica localmente.
    - `requirementsColab.txt`, fichero necesario para la ejecución básica del proyecto en el entorno de Colab.

- src/: Directorio que almacena los ficheros .py incluidos en el repositorio base. Se compone de las siguientes carpetas:
    - algorithms/: Ficheros Python con los algoritmos a implementar en los estudios.
    - arms/: Ficheros Python que definen las clases de objetos utilizados en los algoritmos.
    - plotting: Ficheros correspondientes a la representación de resultados de ejecución de los algoritmos y posterior dibujo de gráficas.

Cabe destacar que en la carpeta `\docs` fue suprimida debido a que la consideramos innecesaria para la entrega, pero dicha mención permanece en el notebok main del proyecto.

## Instalación y Uso
No es necesaria ninguna instrucción de instalación, más allá de la creación de un entorno virtual con el fichero requirements correspondiente al entorno en el que se quiera ejecutar en caso de no querer instalar de forma directa las librerías necesarias.

Hemos requerido modificar las referencias al entorno Github, y hemos cambiado la preparación y duplicación de este entorno a cada notebook de manera individual, ya que cuando queríamos entrar a cada uno de ellos desde el main, Colab nos creaba una instancia nueva y se perdía cualquier otra importación en el main. Por este motivo, hemos optado por realizar la importación y clonación correspondiente de forma individual para cada estudio.
En el caso de querer utilizar un requirements u otro, sería necesario cambiar la referencia dentro de cada notebook en la celda de instalación, sustituyendo las referencias de todas las librerías por el fichero en cuestión.

## Tecnologías Utilizadas
Las herramientas y tecnologías empleadas en el proyecto han sido:

- Python 3.11.11
- Visual Studio Code
- Google Colab