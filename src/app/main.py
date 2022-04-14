from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os

modelo = pickle.load(open('../../models/modelo.sav', 'rb'))
colunas = ['tamanho', 'ano', 'garagem']


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME']= os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD']=os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira APP 1"

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade=tb_en.sentiment.polarity

    return 'Polaridade: {}'.format(round(polaridade,2))

@app.route('/cotacao/', methods = ['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=round(preco[0],2))



app.run(debug=True, host='0.0.0.0')
