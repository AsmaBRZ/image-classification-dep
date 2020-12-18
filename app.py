from flask import Flask,request, render_template
from model.prediction import *
from flask import jsonify 

# create the flask object

app=Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     data = request.form.get('data')
#     if data == None:
#         return 'Got None'
#     else:
#         # model.predict.predict returns a dictionary
#         prediction = model.predict.predict(data)     
#         return json.dumps(str(prediction))

@app.route('/photoRecognize', methods=['POST'])
def photoRecognize():
    if request.method == 'POST': 
        data = request.files['image_data']
        if data == None:
            return 'no image received'
        else:
            # model.predict.predict returns a dictionary
            prediction = predict(data)
    else:
      return render_template('index.html')

    return jsonify(status='OK', results=prediction)


if __name__ == "__main__":
    app.run(debug=True)