# from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]

train_data = pd.read_csv("adult.data", names=headers)
test_data = pd.read_csv("adult.test", names=headers)

# Remove rows which contains "?" (remove unknown values)
train_data = train_data[~train_data.astype(str).apply(lambda x: x.str.contains('\?')).any(axis=1)]
test_data = test_data[~test_data.astype(str).apply(lambda x: x.str.contains('\?')).any(axis=1)]

# Encoding pipelines
continuous_transformer = KBinsDiscretizer(n_bins=25, strategy="uniform", encode="ordinal")
continuous_headers = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
categorical_headers = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

# Split data and label
train_X = train_data.iloc[:, 0:-1]
train_y = categorical_transformer.fit_transform(np.array(train_data.iloc[:, -1]).reshape(-1, 1))

test_X = test_data.iloc[1:, 0:-1]
test_y = categorical_transformer.fit_transform(np.array(test_data.iloc[1:, -1]).reshape(-1, 1))


def DTC(train_X, train_y, test_X, test_y):
    preprocessor = ColumnTransformer(transformers=[
        ("continuous", continuous_transformer, continuous_headers),
        ("categorical", categorical_transformer, categorical_headers)
    ])

    DTClassifier = DecisionTreeClassifier()
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", DTClassifier)])
    pipeline.fit(train_X, train_y)

    y_pred = pipeline.predict(test_X)
    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(test_y, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    # print('DTC Accuracy:', acc)
    return acc


def NN(train_X, train_y, test_X, test_y, n_batch_size, n_hidden_dim):
    # Transform data
    X_con = continuous_transformer.fit_transform(train_X[continuous_headers])
    X_cat = categorical_transformer.fit_transform(train_X[categorical_headers])
    transformed_X = pd.concat([pd.DataFrame(X_cat), pd.DataFrame(X_con)], axis=1)

    test_X_con = continuous_transformer.fit_transform(test_X[continuous_headers])
    test_X_cat = categorical_transformer.transform(test_X[categorical_headers])
    transformed_test_X = pd.concat([pd.DataFrame(test_X_cat), pd.DataFrame(test_X_con)], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(transformed_X, train_y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(Dense(n_hidden_dim, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=1, batch_size=n_batch_size, validation_data=(X_val, y_val), verbose=0)
    test_loss, test_acc = model.evaluate(transformed_test_X, test_y)
    # print('NN accuracy:', test_acc)
    return test_acc


def nn_tuning_run():
    args_list = [
        {"batch_size": 16, "hidden_dim": 32},
        {"batch_size": 16, "hidden_dim": 64},
        {"batch_size": 16, "hidden_dim": 128},

        {"batch_size": 32, "hidden_dim": 32},
        {"batch_size": 32, "hidden_dim": 64},
        {"batch_size": 32, "hidden_dim": 128},

        {"batch_size": 64, "hidden_dim": 32},
        {"batch_size": 64, "hidden_dim": 64},
        {"batch_size": 64, "hidden_dim": 128}
    ]

    result_list = []
    best_acc = 0
    best_result = {}

    for args in args_list:
        result_list.append(
            {
                "args": args,
                "result": nn_tuning_run_helper(args)
            }
        )

    with open("nn_result.txt", "w") as f:
        for result in result_list:
            acc = result['result']
            if acc > best_acc:
                best_acc = acc
                best_result = {"batch_size": result['args']['batch_size'], "hidden_dim": result['args']['hidden_dim']}
            f.write(f"batch_size: {result['args']['batch_size']}, hidden_dim: {result['args']['hidden_dim']}\n")
            f.write(f"nn_acc: {acc}\n")
            f.write("================================================\n")
        f.write(f"best result:\n")
        f.write(f"batch_size: {best_result['batch_size']}, hidden_dim: {best_result['hidden_dim']}\n")
        f.write(f"nn_acc: {best_acc}\n")

    print("nn tuning done")


def nn_tuning_run_helper(args):
    nn_acc = NN(train_X, train_y, test_X, test_y, args["batch_size"], args["hidden_dim"])
    # dtc_acc = DTC(train_X, train_y, test_X, test_y, continuous_transformer)
    return nn_acc



nn_tuning_run()
