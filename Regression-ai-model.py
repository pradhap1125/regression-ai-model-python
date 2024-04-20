# Libraries
import pandas as pd
from sklearn.metrics import r2_score
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def model_finder(path):
    delete_old_models()
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # dataset split to training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Decision Tree Regression model training part
    from sklearn.tree import DecisionTreeRegressor
    regressor_dec_tree = DecisionTreeRegressor(random_state = 0)
    regressor_dec_tree.fit(X_train, y_train)
    # Test set results
    y_pred = regressor_dec_tree.predict(X_test)
    # R2 Evaluation
    dec_tree_perf=r2_score(y_test, y_pred)

    # Multiple Linear Regression model training part
    from sklearn.linear_model import LinearRegression
    regressor_lin = LinearRegression()
    regressor_lin.fit(X_train, y_train)

    # Test set results
    y_pred = regressor_lin.predict(X_test)
    mul_lin_perf=r2_score(y_test, y_pred)

    # Polynomial Regression model Training part

    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X_train)
    regressor_pol = LinearRegression()
    regressor_pol.fit(X_poly, y_train)

    # Test set results
    y_pred = regressor_pol.predict(poly_reg.transform(X_test))
    pol_reg_perf=r2_score(y_test, y_pred)

    # Random Forest Regression model Training part
    from sklearn.ensemble import RandomForestRegressor
    regressor_ran_for = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor_ran_for.fit(X_train, y_train)

    # Test set results
    y_for = y.reshape(len(y),1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_for, test_size = 0.2, random_state = 0)
    y_pred = regressor_ran_for.predict(X_test)
    ran_for_perf=r2_score(y_test, y_pred)

    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)

    # SVR model Training part
    from sklearn.svm import SVR
    regressor_svr = SVR(kernel = 'rbf')
    regressor_svr.fit(X_train, y_train)
    # Test set results
    y_pred = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_test)).reshape(-1,1))
    svr_perf=r2_score(y_test, y_pred)

    maximum_number = max([dec_tree_perf,mul_lin_perf,pol_reg_perf,ran_for_perf,svr_perf])

    if(maximum_number==dec_tree_perf):
        print("decision tree regression model is best fit for this input")
        joblib.dump(regressor_dec_tree, "regressor_dec_tree.pkl")

    elif(maximum_number==mul_lin_perf):
        print("multiple linear regression model is best fit for this input")
        joblib.dump(regressor_lin, "regressor_lin.pkl")

    elif (maximum_number == pol_reg_perf):
        print("polynomial regression model is best fit for this input")
        joblib.dump(regressor_pol, "regressor_pol.pkl")
        joblib.dump(poly_reg, "pol_dep.pkl")


    elif (maximum_number == ran_for_perf):
        print("randon forest regression model is best fit for this input")
        joblib.dump(regressor_ran_for, "regressor_ran_for.pkl")

    else:
        print("support vector regression model is best fit for this input")
        joblib.dump(regressor_svr, "regressor_svr.pkl")

    print ("Model has been trained with input and stored and it is ready for prediction")


def model_predict(path):

    dataset = pd.read_csv(path)
    X_test = dataset.iloc[:, :].values
    y_test = None


    my_file_dec_tree = Path("regressor_dec_tree.pkl")
    my_file_mul_lin = Path("regressor_lin.pkl")
    my_file_pol = Path("regressor_pol.pkl")
    my_file_ran_for = Path("regressor_ran_for.pkl")
    my_file_svr = Path("regressor_svr.pkl")

    if (my_file_dec_tree.is_file()):
        print("using trained regression tree model to predict the output")
        reg=joblib.load("regressor_dec_tree.pkl")
        y_test = reg.predict(X_test)

    elif (my_file_mul_lin.is_file()):
        print("using trained multi linear tree model to predict the output")
        reg = joblib.load("regressor_lin.pkl")
        y_test = reg.predict(X_test)

    elif (my_file_pol.is_file()):
        print("using trained polynomial regression model to predict the output")
        reg = joblib.load("regressor_pol.pkl")
        poly_reg = joblib.load("pol_dep.pkl")
        y_test = reg.predict(poly_reg.transform(X_test))

    elif (my_file_ran_for.is_file()):
        print("using random forest regression model to predict the output")
        reg = joblib.load("regressor_ran_for.pkl")
        y_test = reg.predict(X_test)

    elif (my_file_svr.is_file()):
        print("using SVR regression model to predict the output")
        reg = joblib.load("regressor_svr.pkl")
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        y_test = sc_y.inverse_transform(reg.predict(sc_X.transform(X_test)).reshape(-1, 1))
    else:
        print("Unable to find any trained model, Please train the model first")
        raise Exception("unable to find any trained model")
    #dataset["predicted value"] = "24"
    dataset['predicted value'] = y_test
    dataset.to_csv(path, index=False)
    print(y_test)

def delete_old_models():
    #deleting old models
    try:
        Path("regressor_dec_tree.pkl").unlink()
    except FileNotFoundError:
        pass
    try:
        Path("regressor_lin.pkl").unlink()
    except FileNotFoundError:
        pass
    try:
        Path("regressor_pol.pkl").unlink()
    except FileNotFoundError:
        pass
    try:
        Path("regressor_ran_for.pkl").unlink()
    except FileNotFoundError:
        pass
    try:
        Path("regressor_svr.pkl").unlink()
    except FileNotFoundError:
        pass

if __name__ == '__main__':
    option=input('please enter 1 to start fresh,2 to use an trained model \n')
    option=int(option)
    match option:
      case 1:
          print("This is a simple python code to decide the best regression mode for input dataset and use it")
          print("Dataset should be in csv format and should contain independent columns and dependent column at last")
          path = input("Please enter the path of the file \n")
          model_finder(path)
      case 2:
          pre_path=input("please pass the file path of independent columns for prediction in the same format as trained dataset without dependent column\n")
          model_predict(pre_path)
      case _:
         print("invalid option, please try again")