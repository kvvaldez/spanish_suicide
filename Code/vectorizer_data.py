from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import numpy as np
from joblib import dump, load


# Ubicacion del modelo word2vec pre-entrenado en español
path_tfidf_model='./modelo/'
# DIMENCION VECTOR
dim_vec=100

# Usa frecuencia de palabras
def create_tf_idf(list_text):
	#X_train, X_test, y_train, y_test = train_test_split(vectores, clases, test_size=0.2) # prueba imprimir test

	vectorizer = TfidfVectorizer(use_idf=False, norm=None, ngram_range=(1, 1)) # use_idf=true ponderacion de frecuancia de palabras en documentos
	tfidf = vectorizer.fit(list(list_text))
	vectores_tfidf=vectorizer.transform(list_text)
	dump(tfidf, path_tfidf_model+'tfidf_model.joblib')
	print('Fue creado modelo TF-IDF')
	#vectores_tfidf=list(matrix)
	#print(len(vectores_tfidf))
	return vectores_tfidf

# Retorna un vector TF-IDF 
def get_vector_tfidf(list_text):
	tfidf=load(path_tfidf_model+'tfidf_model.joblib')# Ojo cuidado, solo cargar solo una vez
	words_ntf=[x for x in list_text.split() if x not in tfidf.get_feature_names()] # Palabras que no estan en el vocabulario del modelo
	return tfidf.transform([list_text]).toarray(), words_ntf


# Carga en modelo word2vec en memoria
def load_w2v(path_model): 
	print('Cargando modelo pre-entrenado WORD2VEC...')
	model_w2v = KeyedVectors.load_word2vec_format(path_model, binary=True)
	return model_w2v

# Convierte una frase en un vector, usando sumatoria de vectores word2vec
def words_to_vec(frase,model_w2v=None):
	#model_w2v=load_w2v(path_w2v)
	list_words=frase.split()
	words_nw2v=[]
	v=np.array([0]*dim_vec) # se crea vectores de dimensiones iguales al modelo entrenado de word2vec
	for w in list_words: #Para la vectorizacion de las frases se estan sumando los vectores de cada palabra
		if w in model_w2v.vocab:
			v=v+model_w2v[w]
		else:
			words_nw2v.append(w) # Palabras que no estan en el vocabulario del modelo
	return v, words_nw2v

# Vectoriza todas una lista de frases 
def w2v_vec(list_text, model_w2v): 
	return [words_to_vec(f,model_w2v) for f in list_text]

