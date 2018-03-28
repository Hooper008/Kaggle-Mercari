# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:00:22 2018

@author: Jerry
"""

import numpy as np
import pandas as pd
import os
#import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVR

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, LSTM
from keras.optimizers import Adam
from keras.models import Model
#from keras import backend as K

#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


os.chdir(r"E:\Videos\com")
# Define RMSL Error Function
def rmsle(Y, Y_pred):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))







# Load train and test data


#train_df = pd.read_table('../input/train.tsv')
train_df = pd.read_table('.\\input\\train.tsv')
#test_df = pd.read_table('../input/test.tsv')
test_df = pd.read_table('.\\input\\test.tsv')
print(train_df.head(3), test_df.head(3))

# Prepare data for processing by RNN and Ridge
# Handle missing data.
def fill_missing_values(df):
    df.category_name.fillna(value="Other", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="None", inplace=True)
    return df

train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)


# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=347, train_size=0.9)

Y_train = train_df.target.values.reshape(-1, 1)
Y_dev = dev_df.target.values.reshape(-1, 1)

# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
print("Training on", n_trains, "examples")
print("Validating on", n_devs, "examples")
print("Testing on", n_tests, "examples")


"""
This section will use RNN Model to solve the competition with following steps:

1. Preprocessing data
1. Define RNN model
1. Fitting RNN model on training examples
1. Evaluating RNN model on dev examples
1. Make prediction for test data using RNN model
"""

# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])

# stemmer and drop out stopwords
print("Processing stopwords and stem...")
stop = stopwords.words('english')
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split() if not word in stop])

print("   Processing description...")
full_df["item_description"] = full_df["item_description"].map(lambda x:str_stemmer(x))
print("   Processing name...")
full_df["name"] = full_df["name"].map(lambda x:str_stemmer(x))



# Process categorical data


print("Processing categorical data...")
le = LabelEncoder()

le.fit(full_df.category_name)
full_df.category_name = le.transform(full_df.category_name)

le.fit(full_df.brand_name)
full_df.brand_name = le.transform(full_df.brand_name)

del le

# Process text data


print("Transforming text data to sequences...")
raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower()])

print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print("   Transforming text to sequences...")
full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())

del tok_raw


# Define constants to use when define RNN model
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = np.max([
    np.max(full_df.seq_name.max()),
    np.max(full_df.seq_item_description.max()),
]) + 4
MAX_CATEGORY = np.max(full_df.category_name.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1


# Get data for RNN model


def get_keras_data(df):
    X = {
        'name': pad_sequences(df.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(df.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(df.brand_name),
        'category_name': np.array(df.category_name),
        'item_condition': np.array(df.item_condition_id),
        'num_vars': np.array(df[["shipping"]]),
    }
    return X

train = full_df[:n_trains]
dev = full_df[n_trains:n_trains+n_devs]
test = full_df[n_trains+n_devs:]

X_train = get_keras_data(train)
X_dev = get_keras_data(dev)
X_test = get_keras_data(test)

# Define RNN model
def new_rnn_model(lr=0.001, decay=0.0):    
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    # rnn layers
    rnn_layer1 = LSTM(18) (emb_item_desc)
    rnn_layer2 = GRU(6) (emb_name)

    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_category_name)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])

    main_l = Dense(256)(main_l)
    main_l = Activation('elu')(main_l)
    main_1 = Dropout(rate=0.15,seed=1234)(main_l)

    main_l = Dense(128)(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dense(64)(main_l)
    main_l = Activation('elu')(main_l)

    # the output layer.
    output = Dense(1, activation="linear") (main_l)

    model = Model([name, item_desc, brand_name , category_name, item_condition, num_vars], output)

    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="mse", optimizer=optimizer)

    return model

model = new_rnn_model()
model.summary()
del model


# Fit RNN model to train data


# Set hyper parameters for the model.
BATCH_SIZE = 784
epochs = 3

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(n_trains / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.007, 0.0005
lr_decay = exp_decay(lr_init, lr_fin, steps)

rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)

print("Fitting RNN model to training examples...")
rnn_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev), verbose=1,
)


# Evaluate RNN model on dev data


print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
print(" RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))


# Make prediction for test data
rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
rnn_preds = np.expm1(rnn_preds)

#make df for step2
df_step2 = pd.DataFrame()
df_step2["rnn_test_pred"] = list(Y_dev_preds_rnn)
df_step_pre = pd.DataFrame()
df_step_pre["rnn_test_pred"] = list(np.log1p(rnn_preds))

del rnn_model
"""
Ridge Model

This section will solve the competition using Ridge model with following steps:

1. Preprocessing data
1. Fitting Ridge model on training examples
1. Evaluating Ridge model on dev examples
1. Make prediction for test data using Ridge model
"""

# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])
#full_df.drop(['seq_item_description', 'seq_name'], axis = 1, inplace = True)
# Convert data type to string


# Convert data type to string
full_df['shipping'] = full_df['shipping'].astype(str)
full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)

# Extract features from data


print("Vectorizing data...")
default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(full_df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('category_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('category_name'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])

X = vectorizer.fit_transform(full_df.values)

X_train = X[:n_trains]
X_dev = X[n_trains:n_trains+n_devs]
X_test = X[n_trains+n_devs:]

print(X.shape, X_train.shape, X_dev.shape, X_test.shape)


# Fitting Ridge model on training data


print("Fitting Ridge model on training examples...")
ridge_model = Ridge(
    solver='auto', fit_intercept=True, alpha=0.5,
    max_iter=100, normalize=False, tol=0.05,
)
ridge_model.fit(X_train, Y_train)


# Evaluating Ridge model on dev data
Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
df_step2["ridge_test_pred"] = list(Y_dev_preds_ridge)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))

# Make prediction for test data


ridge_preds = ridge_model.predict(X_test)
ridge_preds = np.expm1(ridge_preds)
df_step_pre["ridge_test_pred"] = list(np.log1p(ridge_preds))
# Evaluating for associated model on dev data
#
def para_search(X, Y, param_dict, aim_estimator_object):

    gsearch = GridSearchCV(estimator = aim_estimator_object, 
                            param_grid = param_dict,
                            n_jobs=-1,
                            iid=False, 
                            cv=2,
                            scoring = "neg_mean_squared_error")
    gsearch.fit(X, Y)
    print("Best para:%s" % str(gsearch.best_params_))
    print("Best score:%s" % str(gsearch.best_score_))
    return gsearch

#stacking method by svr model
clf = SVR(kernel= "linear")

gsearch = para_search(df_step2, Y_dev, {"C":[2]}, clf)
#==============================================================================
# def aggregate_predicts(Y1, Y2):
#     assert Y1.shape == Y2.shape
#     ratio = 0.63
#     return Y1 * ratio + Y2 * (1.0 - ratio)
#==============================================================================

preds = np.expm1(gsearch.predict(df_step_pre))
#print("RMSL error for RNN + Ridge on dev set:", rmsle(Y_dev, Y_dev_preds))

# Creating Submission
#preds = aggregate_predicts(rnn_preds, ridge_preds)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv(".\\input\\rnn_ridge_submission.csv", index=False)
