##
# Modelado de tópicos en Python
# Objetivo: Determinar los temas a partir de una lista de palabras y encontrar qué temas usa cada documento
##

#Importar corpora y models
from gensim import corpora, models
import matplotlib.pyplot as plt
import numpy as np

#Cargar los archivos de datos en la variable corpus. La variable almacena todos los documentos de texto en una forma fácil de procesar
corpus = corpora.BleiCorpus('../datos/ap.dat', '../datos/vocab.txt')

#Se crea un modelo de temas usando corpus cómo base de información. Información sobre LdaModel se puede encontrar en https://radimrehurek.com/gensim/models/ldamodel.html
#La llamada de esta función inferirá qué temas están presentes en el corpus (colección de artículos o escritos sobre un tema particular)
modeloTemas = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)

#El modelo puede explorarse de diferentes formas. Utilizando model[doc] se puede ver la lista de temas a las que se refiere un documento.
#Se visualizan como el Id del tema y la probabilidad que tiene asignada (ningún tema tiene probabilidad 0 pero por aproximación, algunos quedan con 0 y no se muestran)
doc=corpus.docbyoffset(0)
temas=modeloTemas[doc]
print(temas)

#En el histograma del número de temas usados en el documento, puede visualizarse mejor este hecho
len(corpus)													#2246 documentos
numTemas = [len(modeloTemas[doc]) for doc in corpus]		#Formar un array con los números de temas de cada documento
plt.hist(numTemas)
plt.ylabel('Número documentos')
plt.xlabel('Número temas')
plt.show()

#El parámetro alpha del constructor de LdaModel define cómo se distribuyen cuántos temas pueden tratarse por documento, tiene un valor entre 0 y 1 y por defecto tiene el valor de 1/num_topics
modeloTemasAlpha=models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1)

#Al hacer el histograma nuevamente, se encontrará que hay una distribución de temas por documento diferente
numTemasAlpha = [len(modeloTemasAlpha[doc]) for doc in corpus]		#Formar un array con los números de temas de cada documento
plt.hist(numTemasAlpha)
plt.ylabel('Número documentos')
plt.xlabel('Número temas')
plt.show()

#Los temas son distribuciones multinomiales que se hacen sobre las palabras (cada palabra tiene una probabilidad para cada tema y aquellas palabras con mayor probabilidad pertenencen al tema)
#Mostrar las palabras que conforman el primer tema:
palabras=modeloTemas.show_topic(1, 64)
print(words)

#Comparación de documentos por temas

#Se genera vectores de NumPy con los temas de todos los documentos en el corpus para calcular las distancias entre cada documento
from gensim import matutils
topics = matutils.corpus2dense(modeloTemas[corpus], num_terms=modeloTemas.num_topics)

#topics es una matriz y podemos usar la función pdist de SciPy para calcular las distancias por parejas. Con un llamado a la función, se están calculando todos los valores de sum((topics[ti] – topics[tj])**2):
from scipy.spatial import distance
pairwise = distance.squareform(distance.pdist(topics))

#Ahora se ajusta un poco la matriz de distancias modificando los elementos e la diagonal para que sean más grandes que la distancia más grande. Sin este ajuste, la función a continuación siempre retornaría el mismo documento como el más cercano
largest = pairwise.max()
for ti in range(len(topics)):
	pairwise[ti,ti] = largest+1

#Por último construímos un clasificador de temas por proximidad (distancia más baja). Documentación de NumPy.argmin(): http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.argmin.html
def closest_to(doc_id):
	return pairwise[doc_id].argmin()

#Obtener el documento más parecido a aquel con ID 1:
closest_to(1)


#Fuentes:
#https://radimrehurek.com/gensim/tut1.html
#https://github.com/RMDK/IPython-Notebooks/blob/master/Machine%20Learning/TopicModels.ipynb
#Building Machine Learning Systems with Python. Luis Pedro Coelho. Willi Richert