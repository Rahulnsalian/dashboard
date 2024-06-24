from email import message
from tkinter import Label
from tokenize import Name
import numpy as np
from flask import Flask,flash, redirect, request, jsonify, render_template, url_for
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
varlis=[]

#from forms import ContactForm

data = pd.read_csv('dataset.csv', encoding='latin-1')
temp=[]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit',methods=['POST','GET'])
def welcome():
    # if func=='predict':
    #     return redirect(url_for('predict',pred=n))
    # else:
    #     return redirect(url_for('recom'))
    # form = ContactForm()
    if request.form.get('pred')=="Recommend":
        return redirect(url_for('predict'))
    elif request.form.get('reco') == "Brands":
        return redirect(url_for('recom'))
    elif request.form.get('skin') == "Skin type":
        return redirect(url_for('skin_type'))
    else:
        return redirect(url_for('home'))

@app.route('/skin_type')
def skin_type():
    return render_template('skin1.html')

@app.route('/skin_t',methods=['POST','GET'])
def skin_t():
    if request.method=='POST':
            texture= request.form.get('texture')
            pores= request.form.get('pores')
            appearance= request.form.get('appearance')
            feel=request.form.get('feel')
            tightness=request.form.get('tightness')
            pimples=request.form.get('pimples')
            wrinkles=request.form.get('wrinkles')
            sun_exposure=request.form.get('sun_exposure')
        
            scores = {
            "Normal": 0,
            "Dry": 0,
            "Oily": 0,
            "Combination": 0
            }
        
            if texture == '1':
                    scores["Normal"] += 1
            elif texture == '2':
                scores["Dry"] += 1
            elif texture == '3' or texture == '4':
                scores["Combination"] += 1
                scores["Oily"] += 1

            if pores == '1' or pores == '2':
                scores["Normal"] += 1
            elif pores == '3':
                scores["Combination"] += 1
            elif pores == '4':
                scores["Oily"] += 1
                
            if appearance == '1':
                scores["Normal"] += 1
            elif appearance == '2':
                scores["Dry"] += 1
            elif appearance == '3':
                scores["Combination"] += 1
            elif appearance == '4':
                scores["Oily"] += 1

            if feel == '1':
                scores["Normal"] += 1
            elif feel == '2':
                scores["Dry"] += 1
            elif feel == '3':
                scores["Combination"] += 1
            elif feel == '4':
                scores["Oily"] += 1

            if tightness == '1' or tightness == '2':
                scores["Dry"] += 1
            elif tightness == '3':
                scores["Combination"] += 1

            if pimples == '3':
                scores["Combination"] += 1
            elif pimples == '4':
                scores["Oily"] += 1

            if wrinkles == '2':
                scores["Dry"] += 1
            elif wrinkles == '3':
                scores["Combination"] += 1

            if sun_exposure == '1':
                scores["Normal"] += 1
            elif sun_exposure == '2':
                scores["Dry"] += 1
            elif sun_exposure == '3':
                scores["Oily"] += 1
            elif sun_exposure == '4':
                scores["Combination"] += 1
            
            skin_type = max(scores, key=scores.get)
                    
            return redirect(url_for("detect_skin_type", skin_type=skin_type))
    
    return "Please submit the form."

@app.route('/detect_skin_type/<string:skin_type>', methods=['GET'])
def detect_skin_type(skin_type):
    """Display the detected skin type."""
    return render_template('skin2.html', skin_type=skin_type)

@app.route('/predict')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    return render_template('index.html')

@app.route('/predict1',methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    Label = request.form.get('Label')
    Brand = request.form.get('Brand')
    Name = request.form.get('Name')
    Price= request.form.get('Price')
    selection = request.form.get('Combination')
    if selection == 'Combination':
        final_features = np.array([Label,Brand,Name,Price,1,1,1,1])
    elif selection == 'Dry':
        final_features = np.array([Label,Brand,Name,Price,1,1,0,0])
    elif selection == 'Normal':
        final_features = np.array([Label,Brand,Name,Price,1,0,1,0])
    elif selection == 'Oily':
        final_features = np.array([Label,Brand,Name,Price,1,0,0,1])
    # Dry = request.form.get('Dry')
    # Normal = request.form.get('Normal')
    # Oily = request.form.get('Oily')
     

    
    #final_features = np.array([Label,Brand,Name,Price,Combination])
    prediction = model.predict(final_features)

    if prediction==0:
        predictiontext = "recommended"
        #return redirect(url_for("predans"))
    elif prediction==1:
        predictiontext="not recommended"
        #return redirect(url_for("predans"))
    varlis.append(predictiontext)
    return redirect(url_for("predans"))

@app.route('/predans')
def predans():
    pred = varlis.pop()
    return render_template('predans.html',prediction_text = pred)

@app.route('/recommendation',methods=["POST", "GET"])
def recom():
    temp =[]
    if request.method=="POST":
        user= request.form['nm']
        return redirect(url_for("user",usr=user))
    else:
        return render_template("login.html")


@app.route("/<usr>")
def user (usr):
    #print(usr)
    temp.append(usr)
    kesar = temp.pop()
    df = data[data.Label==kesar].iloc[:,1:4]
    print(data.shape)
    #return f"<h1>{usr}</h1>"
    return render_template('simple.html',tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == "__main__":
    app.run(debug=True)