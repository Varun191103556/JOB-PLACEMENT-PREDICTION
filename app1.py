import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

def conv(a):
    df = pd.DataFrame(a , columns = ['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board','hsc_subject', 'degree_percentage', 'undergrad_degree',
       'work_experience', 'emp_test_percentage', 'specialisation',
       'mba_percent'])
    df.gender.replace({"M":1,"F":0},inplace=True)
    df.ssc_board.replace({"Central":1,"Others":0} , inplace=True)
    df.hsc_board.replace({"Central":1,"Others":0}  , inplace=True) 
    df.hsc_subject.replace({"Commerce":1,"Science":2,"Arts":0} ,inplace=True) 
    df.undergrad_degree.replace({"Comm&Mgmt":1,"Sci&Tech":2,"Others":0} , inplace =True)  
    df.work_experience.replace({"No" :0 , "Yes":1} , inplace=True) 
    df.specialisation.replace({"Mkt&Fin":1,"Mkt&HR":2} ,inplace=True)
    df.drop(columns ='gender' , inplace=True)
    df = np.array(df).reshape(1,-1)
    return df

app =Flask(__name__)
## Load the model
regmodel = pickle.load(open('regmodel2.pkl','rb'))
#scalar = pickle.load(open('Manipulation.pkl','rb'))
@app.route('/')
def job():
    return render_template('job.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = conv(np.array(list(data.values())))
    output = regmodel.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = conv(np.array(data))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    if(output == 0):
        output = "Not Placed"
    else :
        output = "Placed"
    return render_template("job.html",prediction_text="The person is likely to be  {}".format(output))



if __name__ == "__main__":
    app.run(debug=True)

