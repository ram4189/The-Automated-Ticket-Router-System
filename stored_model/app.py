import pandas as pd, numpy as np
from flask import Flask, jsonify, request
import flask
import pickle
import joblib
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression 

#models_dir = 'C:\\Users\\rap\\Documents\\ML\\AAIC\\Masters\\consumer_complaints\\new_start\\webapp_part\\stored_models2\\'
#models_dir2 = 'C:\\Users\\rap\\Documents\\ML\\AAIC\\Masters\\consumer_complaints\\new_start\\webapp_part\\stored_models2\\is_sub_is_dict\\'
#models_dir3 = 'C:\\Users\\rap\\Documents\\ML\\AAIC\\Masters\\consumer_complaints\\new_start\\stored_weights\\'


app = Flask(__name__)

@app.route('/')
def Greet():
    msg = """
            Dear customer,
            We are sorry that you had to face harrasment with your service or product.
            We will surely work on behalf of you to provide you necessary justice.
            Please raise the ticket for the same. !"""
    return msg


@app.route('/raise_ticket')
def index():
    return flask.render_template('consumer_complaints_raise.html')


@app.route('/predict', methods=['POST'])
def predict():
    department_dict = {0:'Credit Card', 1:'Mortgage', 2:'Bank account or service',3:'Consumer Loan', 4:'Student loan' ,5:'Credit reporting', 6:'Money transfers',7:'Debt Collection', 8:'Payday Loan',9:'Prepaid Card',10:'Other financial Service'}
    
    ohe_issue_dict = joblib.load('issue_vr.joblib')
    ohe_sub_issue_dict = joblib.load('sub_issue_vr.joblib')
    model = joblib.load('svm_model.joblib')
    
    userInput = request.form.to_dict()

    print('\n')

    print(str(userInput))

    iss = np.array(userInput['Issues'].strip()).reshape(-1,1)
    sis = np.array(userInput['sub_issues'].strip()).reshape(-1,1)

    if sis[0][0]=='':
        sis=[['LEFT_BLANK']]


    issue = iss[0][0]
    sub_issue = sis[0][0]
    
    issue_vec = ohe_issue_dict[issue]
    sub_issue_vec = ohe_sub_issue_dict[sub_issue]
    all_features = list([issue, sub_issue])
    
    all_features_vec = hstack([issue_vec, sub_issue_vec])

    #print(issue, issue_vec)
    #print(sub_issue, sub_issue_vec)
    print(all_features, type(all_features_vec))
    y_pred = model.predict_proba(all_features_vec)
    depart = department_dict[np.argmax(y_pred)]
    print('prediction', depart)
    #print('model_weight', model.coef_.shape)
    



    
    #iss_vec = ohe_issue_model.transform(iss).toarray()
    #si_vec = ohe_sub_issue_model.transform(sis).toarray()

    #print(iss_vec.shape)
    #print(si_vec.shape)

    print('\n')
    
    #{'Issues': ' Account opening, closing, or management ', 'sub_issues': ' Talked to a third party about my debt ', 'submitted_via': ' Referral ', 'tags': ' Servicemember ', 'state': 'KA', 'zipcode': '450034', 'consumer_narrative': 'going to production'}
    return 'Thanks for your patience, Your ticket has been routed to %s department. We will get back to you as soon as possible.' % (depart)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
