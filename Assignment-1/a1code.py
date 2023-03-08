import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV


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


def DTC(train_X, train_y, test_X, test_y, params):
    preprocessor = ColumnTransformer(transformers=[
        ("continuous", continuous_transformer, continuous_headers),
        ("categorical", categorical_transformer, categorical_headers)
    ])

    DTClassifier = DecisionTreeClassifier(max_depth=params['classifier__max_depth'], min_samples_split=params['classifier__min_samples_split'], min_samples_leaf=params['classifier__min_samples_leaf'])
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", DTClassifier)])
    pipeline.fit(train_X, train_y)

    y_pred = pipeline.predict(test_X)
    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(test_y, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Fine-tuned DTC Accuracy:', acc)
    return acc


def DTC_search_tuning(train_X, train_y, test_X, test_y):
    param_grid = {
        'classifier__max_depth': [5, 10, 15],
        'classifier__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    preprocessor = ColumnTransformer(transformers=[
        ("continuous", continuous_transformer, continuous_headers),
        ("categorical", categorical_transformer, categorical_headers)
    ])
    DTClassifier = DecisionTreeClassifier()
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", DTClassifier)])
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(train_X, train_y)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    return grid_search.best_score_, grid_search.best_params_


def DTC_run():
    score, params = DTC_search_tuning(train_X, train_y, test_X, test_y)
    acc = DTC(train_X, train_y, test_X, test_y, params)
    with open("DTC_result.txt", "w") as f:
        f.write(f"tuning score: {score}\n")
        f.write(f"max depth: {params['classifier__max_depth']}\n")
        f.write(f"min sample leaf: {params['classifier__min_samples_leaf']}\n")
        f.write(f"min sample split: {params['classifier__min_samples_split']}\n")
        f.write(f"DTC acc: {acc}")


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

    time_1 = time.time()
    history = model.fit(X_train, y_train, epochs=20, batch_size=n_batch_size, validation_data=(X_val, y_val), verbose=1)
    time_2 = time.time()
    test_loss, test_acc = model.evaluate(transformed_test_X, test_y)
    time_3 = time.time()

    return test_acc, test_loss, time_2 - time_1, time_3 - time_2


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

    best_acc = 0
    best_result = {}

    with open("nn_result.txt", "w") as f:
        for args in args_list:
            acc, loss, train_time, test_time = NN(train_X, train_y, test_X, test_y, args["batch_size"], args["hidden_dim"])
            if acc > best_acc:
                best_acc = acc
                best_result = {"batch_size": args['batch_size'], "hidden_dim": args['hidden_dim'], "loss": loss, "train_time": train_time, "test_time": test_time}
            f.write(f"batch_size: {args['batch_size']}, hidden_dim: {args['hidden_dim']}\n")
            f.write(f"nn_acc: {acc}, nn_loss: {loss}, train_time: {train_time}, test_time: {test_time}\n")
            f.write("================================================\n")
        f.write(f"best result:\n")
        f.write(f"batch_size: {best_result['batch_size']}, hidden_dim: {best_result['hidden_dim']}\n")
        f.write(f"nn_acc: {best_acc}, nn_loss: {best_result['loss']}, train_time: {best_result['train_time']}, test_time: {best_result['test_time']}\n")

    print("nn tuning done")


nn_tuning_run()
DTC_run()
