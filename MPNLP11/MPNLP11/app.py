import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle


 

app = Flask(__name__) #Initialize the flask App


model = pickle.load( open('random.pickle', 'rb') )
 
vecs = pickle.load( open('vectorizers.pickle', 'rb') )
classifiers = pickle.load( open('classifiers.pickle', 'rb') )
 


@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')

@app.route('/performance')
def performance():
	return render_template('performance.html')
   

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)
        
@app.route('/fake_prediction')
def fake_prediction():
    return render_template('fake_prediction.html')

  


@app.route('/predict',methods=['POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    y_pred = model.predict(final_features)
    if y_pred[0] == 1:
       label="Fake Job Post"
    elif y_pred[0] == 0:
       label="Legit Job Post"
    return render_template('fake_prediction.html', prediction_texts=label)
@app.route('/text_prediction')
def text_prediction():
 	return render_template("text_prediction.html")

 

@app.route('/job')
def job():	
	abc = request.args.get('news')	
	input_data = [abc.rstrip()]
	# transforming input
	tfidf_test = vecs.transform(input_data)
	# predicting the input
	y_preds = classifiers.predict(tfidf_test)
	if y_preds[0] == 1:
		labels="Fake Job Post"
	elif y_preds[0] == 0:
		labels="Legit Job Post"
	return render_template('text_prediction.html', prediction_text=labels) 
    
    
if __name__ == "__main__":
    app.run(debug=True)
