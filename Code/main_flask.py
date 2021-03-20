from flask import Flask, jsonify, render_template, request
from gensim.models import KeyedVectors

from intent_classification import get_intent, get_intent_prob, training_models, delete_sw
import pickle

#path_w2v="/root/w2v/SBW-vectors-300-min5.bin" 
path_w2v="/home/kid/Documentos/tesis unsa suicidio/cod_suicidio/codigo/modelo/w2d_latin.bin"
model_w2v = KeyedVectors.load_word2vec_format(path_w2v, binary=True)
et_gestion=['Sin tendencia suicida','Con tendencia suicida'] #Gestiones que son usadas en el bot


app=Flask(__name__)


@app.route("/")
def hello():
	return render_template('index.html')


@app.route('/get_word')
#@app.cache.cached(timeout=2)
def get_prediction():
	frase = request.args.get('intencions') 
	return get_intent_proba_dos(frase)


@app.route("/entrenar")
def training_modelss():
	training_models(model_w2v)
	return 'Modelo Entrenado'

@app.route("/identificados/<string:frase>")
def get_intent_proba_dos(frase):
	# pre-processing text
	frase_sw=delete_sw(frase)
	data={}
	# Word2vec
	intenciones_w2v, words_nw2v=get_intent_prob(frase_sw,tipo_model='w2v',modelo_w2v=model_w2v)
	respuesta_w2v='no-palabra' if not intenciones_w2v else [(et_gestion[i],str(intenciones_w2v[i])) for i in range(len(intenciones_w2v))]

	# TF-IDF
	intenciones_tf, words_ntf=get_intent_prob(frase_sw,tipo_model='tf-idf',modelo_w2v=model_w2v)
	respuesta_tf='no-palabra' if not intenciones_tf else [(et_gestion[i],str(intenciones_tf[i])) for i in range(len(intenciones_tf))]

	return jsonify({'Frase': frase, 'Intencion':respuesta_tf, 'Intencion2':respuesta_w2v, 'No w2v':', '.join(words_nw2v)})


if __name__ == '__main__':
	app.run(debug=True)
