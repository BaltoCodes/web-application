from datetime import datetime
from django.shortcuts import render, redirect
from binance import Client
import requests
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.http import HttpResponse
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
import ta
import matplotlib.pyplot as plt
import io
import urllib, base64
import json
import string
import random
import nltk
from nltk.stem import WordNetLemmatizer # It has the ability to lemmatize.
import tensorflow as tf
from tensorflow import keras # A multidimensional array of elements is represented by this symbol.
from keras import Sequential, optimizers # Sequential groups a linear stack of layers into a tf.keras.Model
from keras.layers import Dense, Dropout
from yahoo_fin.stock_info import get_data


##########  Variables   ##########
client_id = '7cb7124d4b2c48ec8dc755744a6451ce'
client_secret = 'b1508082388f4e71bc8ed94bada90280'
url_redirect = 'http://127.0.0.1:8000/callback/'
spotify_token_uri='https://accounts.spotify.com/api/token'
spotify_auth_uri='https://accounts.spotify.com/authorize'


#######################







#### John Hull -Options, futures and other derivatives
def hull(request):
    option_price = None
    stock_price_paths = None


    if request.method == 'POST':
        # Récupérer les données du formulaire
        current_stock_price = int(request.POST['current_stock_price'])
        strike_price = int(request.POST['strike_price'])
        interest_rate = float(request.POST['interest_rate'])/100
        volatility = float(request.POST['volatility'])/100
        time_to_expiration = float(request.POST['time_to_expiration'])
        option_type = request.POST['option_type']
        nb_echantillons=int(request.POST['nb_echantillons'])



        option_price, stock_price_paths = monte_carlo_option_pricing_with_plot(current_stock_price, strike_price, interest_rate, volatility, time_to_expiration, option_type, nb_echantillons)
        stock_price_paths = json.dumps(stock_price_paths.tolist())

    return render(request, 'BBS/hull.html', {'option_price': option_price, 'stock_price_paths': stock_price_paths})

    


    



def world(request):
    return render(request, "BBS/world-V1.html")








#Cette page est la page d'accueil sous sa dernière version
def index(request):
    return render(request, "BBS/index.html", context={"date": datetime.today()})




def compute_rendement(indice):
    rdt=[0]*len(indice)
    for i in range(len(indice)):
        if indice['close'][i] == indice['open'][i] :
            rdt[i]=0
        else:
            rdt[i]=round(((indice['close'][i]-indice['open'][i])/indice['open'][i])*100,2)
    indice['rendement']=rdt
    return indice



# Retourne 
def compute_best_rendement(start_date, end_date) :
    liste= get_data('SPY', start_date=start_date, end_date=end_date, index_as_date = True, interval='1mo')
    dates=liste.index 
    rdt=compute_rendement(liste)
    return rdt, dates





#### Retournes les rendements des compagnies qui ont surperformé le SP500 sur une période donnée
def compute_company_rendement(start_date, end_date, interval, indice) :
    renta_max={}
    
    for comp in indice : 
        try :
            liste=get_data(comp, start_date=start_date, end_date=end_date, index_as_date = True, interval=interval)
            liste=compute_rendement(liste)
            renta_max[comp]=liste
        except :
            pass
    return renta_max






#Retourne les entreprises qui ont surperformées l'indice
def get_best_companies(nb_months, renta_max, sp500_price):
    #renta_max=renta_max['rendement']
    best={}
    for comp in renta_max : 
        renta=renta_max[comp]['rendement']
        neg, pos=0,0
        for i in range(len(renta_max[comp][-nb_months:])):


            if renta[-nb_months:][i] < sp500_price['rendement'][-nb_months:][i] :
                neg+=1
            else :
                pos +=1
        
        if neg > pos :
            print(f"{comp}  : Pas efficace et ne surperforme pas. Moins performant : {neg} , Plus performant : {pos}")
        elif neg < pos : 
            print(f"{comp}  : Efficace et surperforme en général. Moins performant : {neg} , Plus performant : {pos}")
            best[comp]=pos




from collections import defaultdict



######Code python qui permet de calculer les actions qui surperforment le SP500
def rendement_wealth_management(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    if start_date and end_date:
        rendement_liste, dates = compute_best_rendement(start_date, end_date)
        rendement = rendement_liste['rendement']
        #dates=dates.strftime('%B %Y')
        length=len(rendement)

        matrice = defaultdict(dict)
        year, month =[], []
        for i in range(len(rendement)):

            year.append(dates[i].year)
            month.append(dates[i].month)
            



        
    else:
        rendement = None
        dates = None
        length=None

    return render(request, 'BBS/wealth-management.html', {'rendement': rendement, 'dates': dates, "length":length,  "months": month, "year":year})









##################    Spotify   ##################
def spotify_callback(request):
        code = request.GET.get('code')

        token_data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': url_redirect,
            'client_id': client_id,
            'client_secret': client_secret
        }

        token_response = requests.post(spotify_token_uri, data=token_data)
        token_json = token_response.json()

        access_token = token_json['access_token']

        user_data_response = requests.get('https://api.spotify.com/v1/me', headers={'Authorization': f'Bearer {access_token}'})
        user_data = user_data_response.json()

        api_url = 'https://api.spotify.com/v1/me/top/artists'
        headers = {'Authorization': f'Bearer {access_token}'}
        response_artist = requests.get(api_url, headers=headers)
            #artist_data = response_artist.json()
        print(response_artist)


        api_url_artist='https://api.spotify.com/v1/me/top/tracks'

        response_song = requests.get(api_url_artist, headers=headers)
        #song_data = response_song.json()
        #print(artist_data)
        print(response_song)


        name=user_data['display_name']
        followers=user_data['followers']['total']
        image=user_data['images'][0]['url']
       
        return render(request, "BBS/callback.html", context={"name": name, "followers":followers, "image":image})



#Le code qui permet de render la page d'accueil de Spotify 
def spotify(request):
    return render(request, "BBS/spotify.html")


#Le code qui permet de rediriger vers la page de login de Spotify
def obtenir_login(request):
    sp_oauth = SpotifyOAuth(client_id, client_secret, url_redirect)
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


###########################################################









############   Le code qui permet de générer le graphique interactif 
def interactive_graph(request):
   data=generation_data('BTCUSDT', '1h')
   return render(request, "BBS/interactive_graph.html", context={"data": data})
###########################################################





############   Le code qui permet de générer la version première du monde
def world_is_yours(request): 
    return render(request, 'BBS/world-V1.html')









####### Le code qui permet de générer la data financière associée à n'importe quel actif cryptomonnaie
def generation_data(coin,interval):
    from datetime import timezone
    df={}
    #df1=pd.DataFrame(columns=pairList)
    #for pair in pairList:
        # print(pair)
    binance = Client('y4pd3mx4kPQ5drHGA7xtv7xuUCobXcBJSJJ54zV5oZmAz4RXgEXwAJ9uEmzwarD2','qMr0iTqa1byVDs3xHvq0BKGO29msysaFuKYp9zKkGDa4SThE97XTbufKXyEo9J24')

    dfList = pd.DataFrame(columns= ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time'])
    candle = binance.get_klines(symbol = coin, interval =interval)

    opentime, lopen, lhigh, llow, lclose, lvol, closetime = [], [], [], [], [], [], []
    for i in candle : 
        opentime.append(i[0])
        lopen.append(i[1])
        lhigh.append(i[2])
        llow.append(i[3])
        lclose.append(i[4])
        lvol.append(i[5])
        closetime.append(i[6])
    dfList['Open_time'] = opentime
    dfList['Close_time'] = closetime
    normal_time=[]
    normal_close_time=[]
    for i in range(len(opentime)):
        #normal_time.append(opentime[i]/1000)
        normal_time.append(datetime.fromtimestamp(opentime[i]/1000, tz=timezone.utc))
        normal_close_time.append(datetime.fromtimestamp(closetime[i]/1000, tz=timezone.utc))
        
    dfList['Open time international']=normal_time
    dfList['Close time international']=normal_close_time
    dfList['Open'] = np.array(lopen).astype(float)
    dfList['High'] = np.array(lhigh).astype(float)
    dfList['Low'] = np.array(llow).astype(float)
    dfList['Close'] = np.array(lclose).astype(float)
    dfList['Volume'] = np.array(lvol).astype(float)
    dfList['Mean']=(np.array(lclose).astype(float) + np.array(lopen).astype(float))/2
    df= dfList   

    return(df)
#####################







##############  Graph View 
def graph_view(request):
    btc=generation_data('BTCUSDT', '1h')
    plt.figure(figsize=(10, 4))

    #axis.set_facecolor((0.100, 0.93, 0.96))
    plt.plot(btc['Open time international'], btc['Open'], color='red', linewidth=2, mouseover=True, fillstyle='full')

    plt.xlabel("Date")
    plt.ylabel("Prix")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encodage de l'image en base64
    graphic = urllib.parse.quote(base64.b64encode(image_png))

    btc['rsi']=ta.momentum.rsi(btc['Close'])
    #rsi=[i for i in rsi]
    liste=ta.trend.MACD(btc['Close'])
    btc['MACD']=liste.macd()
    btc['MACD_SIGNAL']=liste.macd_signal()
    btc['area'] = btc['MACD'] - btc['MACD_SIGNAL']

    btc['crossover'] = ((btc['MACD'] > btc['MACD_SIGNAL']) & (btc['MACD'].shift(1) <= btc['MACD_SIGNAL'].shift(1))) | \
                    ((btc['MACD'] < btc['MACD_SIGNAL']) & (btc['MACD'].shift(1) >= btc['MACD_SIGNAL'].shift(1)))

    # Tracer le graphique
    plt.figure(figsize=(12, 6))
    plt.plot(btc['Close time international'][500-len(btc['MACD']):], btc['MACD'], label='MACD', color='blue')
    plt.plot(btc['Close time international'][500-len(btc['MACD']):], btc['MACD_SIGNAL'], label='MACD_SIGNAL', color='orange')
    plt.fill_between(btc['Close time international'][500-len(btc['MACD']):], btc['MACD'], btc['MACD_SIGNAL'], where=(btc['MACD'] >= btc['MACD_SIGNAL']), facecolor='green', alpha=0.3)
    plt.fill_between(btc['Close time international'][500-len(btc['MACD']):], btc['MACD'], btc['MACD_SIGNAL'], where=(btc['MACD'] < btc['MACD_SIGNAL']), facecolor='red', alpha=0.3)
    plt.scatter(btc['Open time international'][btc['crossover']], btc.loc[btc['crossover'], 'MACD'], color='black', marker='o', label='Crossover')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('MACD and MACD_SIGNAL')
    plt.title('MACD and MACD_SIGNAL with Crossovers')
    plt.grid(True)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png_second = buffer.getvalue()
    buffer.close()
    macd = urllib.parse.quote(base64.b64encode(image_png_second))

    
    plt.figure(figsize=(12, 6))
    plt.plot(btc['Open time international'], btc['rsi'])
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.title('Signal RSI')
    plt.fill_between(btc['Open time international'], btc['rsi'], 70, where=(btc['rsi'] >= 70), facecolor='red', alpha=0.3)
    plt.fill_between(btc['Open time international'], btc['rsi'], 30, where=(btc['rsi'] <= 30), facecolor='green', alpha=0.3)
    plt.fill_between(btc['Open time international'], 30, 70, where=((btc['rsi'] > 30) & (btc['rsi'] < 70)), facecolor='yellow', alpha=0.3)

    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png_third = buffer.getvalue()
    buffer.close()
    rsi = urllib.parse.quote(base64.b64encode(image_png_third))

    if request.method == 'POST':
        # Code de la fonction Python à exécuter lorsque le bouton est cliqué
        if btc['rsi'][499] < 40 and btc['MACD'][499]-btc['MACD_SIGNAL'][499] < 10 :
           resutat=f"Bonne opporunité d'achat avec RSI : {btc['rsi'][499]} et MACD-SIGNAL: {btc['MACD'][499]-btc['MACD_SIGNAL'][499]}"
        else : 
           resultat=f"Voici les chiffres : RSI : {round(btc['rsi'][499], 2) } et MACD : {round(btc['MACD'][499]-btc['MACD_SIGNAL'][499], 1)} Pas d'opportunités d'achat pour l'instant "
        
        return render(request, 'DocBlog/graph.html', {'graphic': graphic, 'MACD': macd, 'rsi': rsi , 'resultat': resultat})

    # Encodage de l'image en base64
    return render(request, 'BBS/graph.html', {'graphic': graphic, 'macd': macd, 'rsi': rsi })
#####################







##################### Chatbot
## Celui-ci n'est probablement plus utilisé
def calculator_view(request):
    if request.method == 'POST':
        expression = request.POST.get('expression', '')
        try:
            result = eval(expression)
        except Exception as e:
            result = 'Erreur: {}'.format(str(e))
    else:
        result = ''
    return render(request, 'BBS/calculator.html', {'result': result})
#####################








nltk.download("punkt")
nltk.download("wordnet")
lm = WordNetLemmatizer()

data = {"intents": [

             {"tag": "age",
              "patterns": ["how old are you?", "What's your age"],
              "responses": ["I am 2 years old and my birthday was yesterday", "I'm 18 years old wbu "]
             },
              {"tag": "greeting",
              "patterns": [ "Hi", "Hello", "Hey", "What's up", "wassup", "su^p"],
              "responses": ["Hi there", "Hello", "Hi :)", "Hey there", "Hiii"],
             },
              {"tag": "goodbye",
              "patterns": [ "bye", "later", "see you", "i'm out"],
              "responses": ["Bye", "take care", "ok see you"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "who are you?"],
              "responses": ["I'm Lucie what is yours" "Lucie and you ?", "Lucie and you ?"]
             },
             {"tag": "etat",
              "patterns": ["How are you?", "How are u?", "How u doin", "What's up"],
              "responses": ["Good wbu ?", "Doing good how are you ?", "Doing good what about you"]
             },
             {"tag": "location",
              "patterns": ["where do you live?", "where are you from ?"],
              "responses": ["I live in Zurich right now and u ?" "I live in zurich at the moment ", "From Zurich wbu ?"]
             }
]}




##################### Creer les mots 
def create_word(data):
  #for getting words
    # lists
    ourClasses = []
    newWords = []
    documentX = []
    documentY = []
    # Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            ournewTkns = nltk.word_tokenize(pattern)# tokenize the patterns
            newWords.extend(ournewTkns)# extends the tokens
            documentX.append(pattern)
            documentY.append(intent["tag"])


        if intent["tag"] not in ourClasses:# add unexisting tags to their respective classes
            ourClasses.append(intent["tag"])

    newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
    newWords = sorted(set(newWords))# sorting words
    ourClasses = sorted(set(ourClasses))# sorting classes

    return(ourClasses, newWords, documentX, documentY )
##########################################




##########################################
def create_model(ourClasses, documentX,documentY, newWords):
    trainingData = [] # training list array
    outEmpty = [0] * len(ourClasses)
    # bow model
    for idx, doc in enumerate(documentX):
        bagOfwords = []
        text = lm.lemmatize(doc.lower())
        for word in newWords:
            bagOfwords.append(1) if word in text else bagOfwords.append(0)

        outputRow = list(outEmpty)
        outputRow[ourClasses.index(documentY[idx])] = 1
        trainingData.append([bagOfwords, outputRow])

    random.shuffle(trainingData)
    trainingData = np.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

    x = np.array(list(trainingData[:, 0]))# first trainig phase
    y = np.array(list(trainingData[:, 1]))
    iShape = (len(x[0]),)
    oShape = len(y[0])
    # parameter definition
    ourNewModel = Sequential()

    ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
    ourNewModel.add(Dropout(0.5))
    ourNewModel.add(Dense(64, activation="relu"))
    ourNewModel.add(Dropout(0.3))
    ourNewModel.add(Dense(oShape, activation = "softmax"))
    md = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
    ourNewModel.compile(loss='categorical_crossentropy',
                optimizer=md,
                metrics=["accuracy"])

    ourNewModel.fit(x, y, epochs=200, verbose=0)

    return ourNewModel
##########################################





def ourText(text):
  newtkns = nltk.word_tokenize(text)
  newtkns = [lm.lemmatize(word) for word in newtkns]
  return newtkns

def wordBag(text, vocab):
  newtkns = ourText(text)
  bagOwords = [0] * len(vocab)
  for w in newtkns:
    for idx, word in enumerate(vocab):
      if word == w:
        bagOwords[idx] = 1
  return np.array(bagOwords)

def Pclass(text, vocab, labels, ourNewModel):
  
  bagOwords = wordBag(text, vocab)
  ourResult = ourNewModel.predict(np.array([bagOwords]))[0]
  newThresh = 0.2
  yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

  yp.sort(key=lambda x: x[1], reverse=True)
  newList = []
  for r in yp:
    newList.append(labels[r[0]])
  return newList

def getRes(firstlist, fJson):
  tag = firstlist[0]
  listOfIntents = fJson["intents"]
  for i in listOfIntents:
    if i["tag"] == tag:
      ourResult = random.choice(i["responses"])
      break
  return ourResult
##########################################





##########################################
#ourClasses, newWords, X, Y = create_word(data)
#ourNewModel = create_model(ourClasses, X,Y, newWords)



messages=[]
def get_message(request) :   

    if request.method == 'POST':
        expression = request.POST.get('expression', '')

        try:
            
            intents = Pclass(expression, newWords, ourClasses, ourNewModel)
            result = getRes(intents, data)
            #result = eval(expression)


            
            messages.append({'user': expression, 'bot': result})
            
            if len(messages)>1:
                messages.remove[0]
            # Stockez l'historique des messages dans localStorage
           # request.session['chat_messages'] = messages
        except Exception as e:
            result = 'Erreur: {}'.format(str(e))
    else:
        result = ''
    return render(request, 'BBS/chatbot.html', {'result': result, 'messages': messages})








def monte_carlo_option_pricing_with_plot(S, K, r, sigma, T, option_type, num_simulations):

    np.random.seed(42)  # For reproducibility
    dt = T / 252  # Assuming 252 trading days in a year
    num_days = int(T * 252)

    stock_price_paths = np.zeros((num_simulations, num_days + 1))
    stock_price_paths[:, 0] = S

    for i in range(1, num_days + 1):
        Z = np.random.normal(0, 1, num_simulations)
        stock_price_paths[:, i] = stock_price_paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Plot stock price paths
    #plt.figure(figsize=(10, 6))
    #plt.plot(np.arange(num_days + 1), stock_price_paths.T, color='blue', alpha=0.1)
    #plt.title('Monte Carlo Simulation - Stock Price Paths')
    #plt.xlabel('Days')
    #plt.ylabel('Stock Price')
    #plt.show()
 
    # Calculate option payoff
    if option_type == 'call':
        option_payoff = np.maximum(stock_price_paths[:, -1] - K, 0)
    elif option_type == 'put':
        option_payoff = np.maximum(K - stock_price_paths[:, -1], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Discounted expected payoff
    option_price = np.exp(-r * T) * np.mean(option_payoff)

    return option_price, stock_price_paths

