#Regression AI model choosing and training project

This is a simple python code which trains regression AI models with the given input and figures out which model will be best suited for this use case based on its R Square value. Successfully trained model with highest R value will be saved in local directory and it can be used to predict the dependent values based on independent values provided.<br><br>
Following are the list of regression models used in this project

1. Decision Tree Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Random Forest Regression
5. Support Vector Regression

This is a simple interactive project whereupon execution, it will ask for path of training data and will train the models with that input. Once a model is trained, user can use it for prediction by selecting use a trained model option.<br><br>
###Requirements:
Python version >= 3.1.0 <br>
Valid test data in csv format in which last column is dependent variable and remaining columns are independent variables. Also please use only numerical values in both dependent and independent variables. Please use file sample-input.csv for reference.<br>

###How to use
pip install -r requirements.txt
python Regression-ai-model.py

###Coming soon
As a part of data preprocessing, taking care of missing data and encoding categorical data(non numerical data)<br>
    1.Adding different validations and enhancements in terms of performance <br>
    2.Creating a docker image out if it <br>
    3.Exposing the functionality as a service


