import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from gensim.models import KeyedVectors #borrar prueba
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from vectorizer_data import get_vector_tfidf, create_tf_idf, w2v_vec, words_to_vec


# DIMENCION VECTOR
dim_vec=100

#kidssas
path_save_movel="./modelo/" #Directorio donde se guardan los modelos entranados
url_data_csv='./modelo/suicidio_notacion.csv'
url_sw='./modelo/stopwords.txt'

#Stopwords
f = open(url_sw) # Open file on read mode
stopwords = f.read().split("\n") # Create a list containing all lines
f.close() # Close file
#stopwords= open(url_sw).readlines()
#print(stopwords)
# Elimina stopwords           
def delete_sw(frase):
	sin=[word for word in frase.split() if word not in stopwords]
	#print('Stopwords')
	if len(sin)==0:
		return frase
	else:
 		return ' '.join(sin)


# Lee el archivo csv y carga las listas que seran usadas.
def load_data(data_csv=url_data_csv):
	data_g = pd.read_csv(data_csv) # Convierte csv en formato pandas
	frase_sw=[]
	for i in list(data_g.tweet_clean):
		frase_sw.append(delete_sw(i))
	return frase_sw, list(data_g.suicidio)


# Retorna un dataframe con la data seleccionada
def get_random_data():
	texto, clase = load_data(url_data_csv)
	#print(texto) 
	data_select = pd.DataFrame({'text': texto,'clase': clase})
	print(data_select.shape)
	balanced = data_select.groupby('clase').apply(sampling_k_elements).reset_index(drop=True)
	return balanced

# Toma 'k' datos random del dataset
def sampling_k_elements(group, k=500): 
    if len(group) < k:
        return group
    return group.sample(k)


# Separa los datos, para el entrenamiento y las pruebas
def split_data(vectores, clases): 
	# tipo_vec='tf-idf', 'w2v'
	#print('vec',len(vectores),len(list(balanced['text'])))
	X_train, X_test, y_train, y_test = train_test_split(vectores, clases, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test

def split_data2(frases, clases): 
	k_train, k_test, l_train, l_test = train_test_split(frases, clases, test_size=0.2, random_state=42)
	for i,j in zip(k_test,l_test):
		print(j,i)
	

# Clasificador Suport Vector Machine sin Kernel
def SVM(X_train, X_test, y_train, y_test,tipo_vec): 
	#X_train, X_test, y_train, y_test=split_data(tipo_vec)
	clf = LinearSVC(random_state=0, tol=1e-5)
	clf.fit(X_train, y_train)  
	LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
	     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
	     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
	#scores = cross_val_score(clf, X_train, y_train, cv=5)
	#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	pred = clf.predict(X_test)
	print(tipo_vec+'_SVM',' Accuracy: ' +str(clf.score(X_test, y_test)))

	print(confusion_matrix(pred, y_test))
	print(classification_report(pred, y_test))
	dump(clf, path_save_movel+tipo_vec+'_SVM.joblib')


# Clasificador Regresion Logistica (distribucion de probabilidades)
def RL(X_train, X_test, y_train, y_test, tipo_vec):
	#X_train, X_test, y_train, y_test=split_data(tipo_vec)
	print(tipo_vec+'_RL')
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
	#y_pre = cross_val_predict(clf, X_train, y_train, cv=5)
	#print(classification_report(y_train, y_pre))
	#scores = cross_val_score(clf, X_train, y_train, cv=5)
	#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	pred = clf.predict(X_test)
	print(tipo_vec+'_RL',' Accuracy: ' +str(clf.score(X_test, y_test)))
	print(confusion_matrix(pred, y_test))
	print(classification_report(pred, y_test))
	dump(clf, path_save_movel+tipo_vec+'_RL.joblib')


# Carga el modelo del clasificador
def load_classifier(tipo_model):
	if tipo_model=='tf-idf':
		return load(path_save_movel+'tf-idf_RL.joblib') 
	elif tipo_model=='w2v':
		return load(path_save_movel+'w2v_RL.joblib') 
	else:
		return 'Seleccione tipo de modelo'


# Retorna una intencion de la frase, para respuesta del bot
def get_intent(frase,tipo_model='w2v',modelo_w2v=None):
	
	v=np.array([0]*dim_vec)#E
	modelo_cl=load_classifier(tipo_model)
	if tipo_model=='w2v':
		frase_vec=words_to_vec(frase,modelo_w2v)
		return 4 if (frase_vec==v).all() else modelo_cl.predict(frase_vec.reshape(1,-1))[0]#E
	else:
		return modelo_cl.predict(get_vector_tfidf(frase))[0]

# Retorna una intencion de la frase para respuesta del bot, junto con la distribucion de probabilidades de pertenecer a una clase
def get_intent_prob(frase,tipo_model='w2v',modelo_w2v=None):
	
	modelo_cl=load_classifier(tipo_model)
	if tipo_model=='w2v':
		v=np.array([0]*dim_vec) # DIMENCION
		frase_vec, words_nw2v=words_to_vec(frase,modelo_w2v)
		prop=modelo_cl.predict_proba(frase_vec.reshape(1,-1))[0]
		dis_prob=[round(x*100,1) for x in prop]
		return ([], words_nw2v) if (frase_vec==v).all() else (dis_prob, words_nw2v)
	else:
		frase_vec, words_ntf=get_vector_tfidf(frase)# retorna el unico vector
		print('SHAPE TF-IDF',frase_vec.shape)
		frase_vec=frase_vec[0].reshape(1,-1)
		l,d=frase_vec.shape
		v=np.array([0]*d)
		prop=modelo_cl.predict_proba(frase_vec)[0]
		dis_prob=[round(x*100,1) for x in prop]
		return ([], words_ntf) if (frase_vec==v).all() else (dis_prob, words_ntf)


# Entrena modelos de clasificacion con los mismos datos
def training_models(model_w2v):
	#path_w2v="/root/w2v/SBW-vectors-300-min5.bin" 
	balanced=get_random_data()#random y seleccion 'x' datos de cada clase
	vectores=[]
	vectores_tf=create_tf_idf(list(balanced['text']))
	vec_words=w2v_vec(list(balanced['text']),model_w2v)
	#test
	split_data2(list(balanced['text']), list(balanced['clase']))
	vectores_w2v=[a[0] for a in vec_words]
	X_train, X_test, y_train, y_test=split_data(vectores_tf, list(balanced['clase']))
	X_train2, X_test2, y_train2, y_test2=split_data(vectores_w2v, list(balanced['clase']))
	print('Entrenando Modelos...')
	SVM(X_train, X_test, y_train, y_test,'tf-idf')
	RL(X_train, X_test, y_train, y_test,'tf-idf')
	SVM(X_train2, X_test2, y_train2, y_test2,'w2v')
	RL(X_train2, X_test2, y_train2, y_test2,'w2v')

if __name__== "__main__":
	training_models()
	#chat()
