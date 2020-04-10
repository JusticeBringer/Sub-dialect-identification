import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GRU, SpatialDropout1D
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression


# function to write the submission txt file
def write_txt(filename, predictions, test_nr_before):
    txt_file = open(filename, 'w')
    txt_file.write("id,label\n")

    predictions = np.array(predictions)
    test_nr_before = np.array(test_nr_before)
    predictions = predictions.ravel()
    test_nr_before = test_nr_before.ravel()

    for i in range(len(test_nr_before)):
        str_to_put = ""
        str_to_put += str(test_nr_before[i])
        str_to_put += ','
        str_to_put += str(int(predictions[i]))
        str_to_put += "\n"
        txt_file.write(str_to_put)

    txt_file.close()

    print(len(test_nr_before))
    print(len(predictions))


# function to force int on predictions
def make_ints(predictions):
    new_pred = []
    for x in range(len(predictions)):
        if predictions[x] <= 0.5:
            # predictions[x] = int(0)
            new_pred.append(int(0))
        else:
            # predictions[x] = int(1)
            new_pred.append(int(1))

    return np.array(new_pred)


# function to load the content of a txt file of only integer values
def get_txt_int_content(file_location):
    return np.loadtxt(file_location, dtype=np.int64)


# function to tranform the content of a txt file into an indexed array variable
def get_txt_array(file_location):
    txt_file = open(file_location, 'r', encoding='utf8')
    # opening the file containing the samples data
    aux_txt_file = txt_file.readlines()
    # putting the file lines into indexed variable
    return np.array(aux_txt_file)
    # transforming variable into an np.array


# function to extract only the samples from given txt file
def extract_samples(txt_samples):
    ret_scaled = []

    for index, u2tuple in enumerate(txt_samples):
        # iterating through the original train samples np.array

        items = u2tuple.split("\t")
        # items[0] contains the number before the text
        # items[1] contains the text

        ret_scaled.append(items[1])

    return np.array(ret_scaled)


# function to extract only the labels from given txt file
def extract_labels(txt_labels):
    ret_labels = []

    for index, u2tuple in enumerate(txt_labels):
        # iterating through the original train labels np.array
        # u2tuple[0] contains the number before the 0/1
        # u2tuple[1] contains 0/1

        ret_labels.append(u2tuple[1])

    return np.array(ret_labels)


# function to extract only the numbers from given txt file
def extract_numbers(txt_numbers):
    ret_numbers = []

    for index, u2tuple in enumerate(txt_numbers):
        # iterating through the original train labels np.array
        items = u2tuple.split("\t")
        # items[0] contains the number before the 0/1
        # items[1] contains 0/1

        ret_numbers.append(items[0])

    return np.array(ret_numbers)


# function to put all words together (from training, validation and testing data)
def put_all_words_three(s_one, s_two, s_three=None):
    if s_three is None:
        s_three = []
    all_words = []

    for x in s_one:
        all_words.append(x)
    for x in s_two:
        all_words.append(x)
    for x in s_three:
        all_words.append(x)

    return np.array(all_words)


# function to put all labels for words (from training, validation and testing data)
def put_all_labels(l_one, l_two, l_three=None):
    if l_three is None:
        l_three = []
    all_labels = []

    for x in l_one:
        all_labels.append(x)
    for x in l_two:
        all_labels.append(x)
    for x in l_three:
        all_labels.append(x)

    return np.array(all_labels)


# function to steem the corpus data
def steem_this(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


# function to count the frequency of words from an array
def count_frequencies_words(arr_samples, words_freq):
    for x in arr_samples:
        w = x.split(" ")
        for y in w:
            if y in words_freq:
                words_freq[y] += 1
            else:
                words_freq[y] = 1


# function to sort in descending order an array
def desc_sort(dict_freq):
    return {k: v for k, v in sorted(dict_freq.items(), key=lambda item: item[1], reverse=True)}


# function to assign frequency indexes
def assign_index(dict_freq, freq_ind):
    ind = 1
    for x in dict_freq:
        if dict_freq[x] > 1:
            freq_ind[x] = ind
            ind += 1
        # else:
        #     freq_ind[x] = 0
        # words that appear once are not considered
        elif dict_freq[x] == 2:
            freq_ind[x] = 12000
        else:
            freq_ind[x] = 150000
        # words that appear once or twice have same index


# function to assign in text indexes
def text_assigning(arr_samples, freq_ind):
    asn_samples = []

    for x in range(len(arr_samples)):
        aux = []
        w = arr_samples[x]
        w = w.split(" ")
        for y in range(len(w)):
            aux.append(freq_ind[w[y]])
        asn_samples.append(aux)
    return np.array(asn_samples)


# function to predict with logistic regression
def logistic_regression_predict(tr_smp, tr_lb, test_smp, all_wrd):
    Train_X = tr_smp
    Train_Y = tr_lb
    Test_X = test_smp

    Tfidf_vect = TfidfVectorizer(max_features=5000, strip_accents='unicode',
                                 ngram_range=(1, 3), max_df=0.9, min_df=5, sublinear_tf=True)
    Tfidf_vect.fit(all_wrd)

    Train_X = Tfidf_vect.transform(Train_X)
    Test_X = Tfidf_vect.transform(Test_X)

    model = LogisticRegression(C=30, dual=False)
    model.fit(Train_X, Train_Y)

    # predict the labels on validation dataset
    predictions = model.predict(Test_X)
    predictions = make_ints(predictions)

    return predictions


# function to predict with svm
def svm_predict(tr_smp, tr_lb, test_smp, all_wrd):
    Train_X = tr_smp
    Train_Y = tr_lb
    Test_X = test_smp

    # Metoda TF-IDF
    Tfidf_vect = TfidfVectorizer(max_features=5000, strip_accents='unicode',
                                 ngram_range=(1, 3), max_df=0.9, min_df=5, sublinear_tf=True)
    Tfidf_vect.fit(all_wrd)

    Train_X = Tfidf_vect.transform(Train_X)
    Test_X = Tfidf_vect.transform(Test_X)

    # Normalizare
    Train_X = normalize(Train_X, axis=1, norm='l1')
    Test_X = normalize(Test_X, axis=1, norm='l1')

    # Standardizare
    # scaler = preprocessing.Normalizer()
    scaler = preprocessing.RobustScaler(quantile_range=(0.1, 0.9), with_centering=False)
    Train_X = scaler.fit_transform(Train_X)
    Test_X = scaler.fit_transform(Test_X)

    model = svm.SVC(C=10, kernel='linear', degree=3, gamma='auto')
    model.fit(Train_X, Train_Y)

    # predict the labels on validation dataset
    predictions = model.predict(Test_X)
    predictions = make_ints(predictions)

    return predictions


# function to predict with bayes
def bayes_predict(tr_smp, tr_lb, test_smp, all_wrd):
    Train_X = tr_smp
    Train_Y = tr_lb
    Test_X = test_smp

    # Metoda TF-IDF
    Tfidf_vect = TfidfVectorizer(max_features=5000, strip_accents='unicode',
                                 ngram_range=(1, 3), max_df=0.9, min_df=5, sublinear_tf=True)
    Tfidf_vect.fit(all_wrd)

    Train_X = Tfidf_vect.transform(Train_X)
    Test_X = Tfidf_vect.transform(Test_X)

    # Normalizare
    Train_X = normalize(Train_X, axis=1, norm='l1')
    Test_X = normalize(Test_X, axis=1, norm='l1')

    # Standardizare
    #scaler = preprocessing.Normalizer()
    scaler = preprocessing.RobustScaler(quantile_range=(0.1, 0.9), with_centering=False)
    Train_X = scaler.fit_transform(Train_X)
    Test_X = scaler.fit_transform(Test_X)

    model = naive_bayes.MultinomialNB(alpha=0.0001)
    model.fit(Train_X, Train_Y)

    # predict the labels on validation dataset
    predictions = model.predict(Test_X)
    predictions = make_ints(predictions)

    return predictions


# function to predict with linear regression
def linear_regression(tr_smp, tr_lb, test_smp, all_wrd):
    Train_X = tr_smp
    Train_Y = tr_lb
    Test_X = test_smp

    # Metoda TF-IDF
    Tfidf_vect = TfidfVectorizer(max_features=5000, strip_accents='unicode',
                                 ngram_range=(1, 3), max_df=0.9, min_df=5, sublinear_tf=True)
    Tfidf_vect.fit(all_wrd)

    Train_X = Tfidf_vect.transform(Train_X)
    Test_X = Tfidf_vect.transform(Test_X)

    # Normalizare
    Train_X = normalize(Train_X, axis=1, norm='l1')
    Test_X = normalize(Test_X, axis=1, norm='l1')

    # Standardizare
    #scaler = preprocessing.Normalizer()
    scaler = preprocessing.RobustScaler(quantile_range=(0.1, 0.9), with_centering=False)
    Train_X = scaler.fit_transform(Train_X)
    Test_X = scaler.fit_transform(Test_X)

    model = LinearRegression()
    model.fit(Train_X, Train_Y)

    # predict the labels on validation dataset
    predictions = model.predict(Test_X)
    predictions = make_ints(predictions)

    return predictions


# function to predict with LSTM
def lstm_predict(assigned_words_train, assigned_words_test, assigned_words_validation
                 , scaled_train_labels, scaled_validation_labels):

    max_review_length = 500

    X_train = assigned_words_train
    X_test = assigned_words_test
    X_validate = assigned_words_validation
    top_words = 10000

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    X_validate = sequence.pad_sequences(X_validate, maxlen=max_review_length)
    y_train = scaled_train_labels
    y_validate = scaled_validation_labels

    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_validate, y_validate), verbose=0)
    pred = model.predict(X_test)
    pred = make_ints(pred)

    return pred


# function to predict with LSTM + CNN
def lstm_cnn_predict(assigned_words_train, assigned_words_test, assigned_words_validation
                     , scaled_train_labels, scaled_validation_labels):

    max_review_length = 500
    top_words = 10000
    X_train = assigned_words_train
    X_test = assigned_words_test
    X_validate = assigned_words_validation

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    X_validate = sequence.pad_sequences(X_validate, maxlen=max_review_length)
    y_train = scaled_train_labels
    y_validate = scaled_validation_labels
    # create model

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu', strides=1))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides=None))
    model.add(LSTM(100))
    model.add(Dense(500, input_dim=2, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = optimizers.Adam(lr=0.01, amsgrad=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_validate, y_validate), verbose=0)

    pred = model.predict(X_test)
    pred = make_ints(pred)

    return pred


# function to predict with gru cells
def gru_predict(assigned_words_train, assigned_words_test, assigned_words_validation
                     , scaled_train_labels, scaled_validation_labels):

    max_review_length = 500
    top_words = 10000
    X_train = assigned_words_train
    X_test = assigned_words_test
    X_validate = assigned_words_validation

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    X_validate = sequence.pad_sequences(X_validate, maxlen=max_review_length)
    y_train = scaled_train_labels
    y_validate = scaled_validation_labels
    # create model

    model = Sequential()
    model.add(Embedding(top_words,
                        150,
                        input_length=500,
                        trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(GRU(150, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(GRU(150, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Fit the model with early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    model.fit(X_train, y_train, batch_size=64, epochs=10,
              verbose=0, validation_data=(X_validate, y_validate), callbacks=[earlystop])

    pred = model.predict(X_test)
    pred = make_ints(pred)

    return pred


if __name__ == '__main__':
    # opening the txt files
    orig_train_samples = get_txt_array('data/train_samples.txt')
    orig_train_labels = get_txt_int_content('data/train_labels.txt')
    orig_validation_samples = get_txt_array('data/validation_samples.txt')
    orig_validation_labels = get_txt_int_content('data/validation_labels.txt')
    orig_test_samples = get_txt_array('data/test_samples.txt')

    # extracting samples and labels
    scaled_train_samples = extract_samples(orig_train_samples)
    scaled_train_labels = extract_labels(orig_train_labels)
    scaled_validation_samples = extract_samples(orig_validation_samples)
    scaled_validation_labels = extract_labels(orig_validation_labels)
    scaled_test_samples = extract_samples(orig_test_samples)
    scaled_test_samples_numbers = extract_numbers(orig_test_samples)

    ################# Preprocessing the data ###############

    all_words_data = put_all_words_three(scaled_train_samples, scaled_validation_samples)
    all_words_labels = put_all_labels(scaled_train_labels, scaled_validation_labels)

    train_words_freq = {}
    # counting the frequencies in train_samples
    count_frequencies_words(scaled_train_samples, train_words_freq)
    # counting the frequencies in validation_samples
    count_frequencies_words(scaled_validation_samples, train_words_freq)
    # counting the frequencies in test_samples
    count_frequencies_words(scaled_test_samples, train_words_freq)

    train_words_freq = desc_sort(train_words_freq)

    freq_ind = {}
    assign_index(train_words_freq, freq_ind)

    # assigning the frequency to each word in the sentence
    assigned_words_train = text_assigning(scaled_train_samples, freq_ind)
    assigned_words_test = text_assigning(scaled_test_samples, freq_ind)
    assigned_words_validation = text_assigning(scaled_validation_samples, freq_ind)

    ################## Predicting Part #####################

    pred_bayes = bayes_predict(scaled_train_samples, scaled_train_labels, scaled_validation_samples, all_words_data)
    print("Bayes prediction on normal data ->", accuracy_score(scaled_validation_labels, pred_bayes) * 100)
    print(confusion_matrix(scaled_validation_labels, pred_bayes))
    print(classification_report(scaled_validation_labels, pred_bayes))
    print('Bayes F1 score: {}'.format(f1_score(scaled_validation_labels, pred_bayes, average='weighted')))

    pred_logistic = logistic_regression_predict(scaled_train_samples, scaled_train_labels, scaled_validation_samples, all_words_data)
    print("Logistic prediction ->", accuracy_score(scaled_validation_labels, pred_logistic))
    print(confusion_matrix(scaled_validation_labels, pred_logistic))
    print(classification_report(scaled_validation_labels, pred_logistic))
    print('Logistic Regression F1 score: {}'.format(f1_score(scaled_validation_labels, pred_logistic, average='weighted')))

    pred_svm = svm_predict(scaled_train_samples, scaled_train_labels, scaled_validation_samples, all_words_data)
    print("SVM prediction ->", accuracy_score(scaled_validation_labels, pred_svm))
    print(confusion_matrix(scaled_validation_labels, pred_svm))
    print(classification_report(scaled_validation_labels, pred_svm))
    print('SVM F1 score: {}'.format(f1_score(scaled_validation_labels, pred_svm, average='weighted')))

    pred_linear_regression = linear_regression(scaled_train_samples, scaled_train_labels, scaled_validation_samples, all_words_data)
    print("Linear Regression prediction ->", accuracy_score(scaled_validation_labels, pred_linear_regression))
    print(confusion_matrix(scaled_validation_labels, pred_linear_regression))
    print(classification_report(scaled_validation_labels, pred_linear_regression))
    print('Linear Regression F1 score: {}'.format(f1_score(scaled_validation_labels, pred_linear_regression, average='weighted')))

    pred_lstm = lstm_predict(assigned_words_train, assigned_words_validation, assigned_words_validation,
                             scaled_train_labels, scaled_validation_labels)
    print("LSTM prediction ->", accuracy_score(scaled_validation_labels, pred_lstm))
    print(confusion_matrix(scaled_validation_labels, pred_lstm))
    print(classification_report(scaled_validation_labels, pred_lstm))
    print('LSTM F1 score: {}'.format(f1_score(scaled_validation_labels, pred_lstm, average='weighted')))

    pred_LSTM_CNN = lstm_cnn_predict(assigned_words_train, assigned_words_validation, assigned_words_validation,
                                     scaled_train_labels, scaled_validation_labels)
    print("LSTM with CNN prediction ->", accuracy_score(scaled_validation_labels, pred_LSTM_CNN))
    print(confusion_matrix(scaled_validation_labels, pred_LSTM_CNN))
    print(classification_report(scaled_validation_labels, pred_LSTM_CNN))
    print('LSTM with CNN F1 score: {}'.format(f1_score(scaled_validation_labels, pred_LSTM_CNN, average='weighted')))

    pred_gru = gru_predict(assigned_words_train, assigned_words_validation, assigned_words_validation,
                                     scaled_train_labels, scaled_validation_labels)
    print("GRU prediction ->", accuracy_score(scaled_validation_labels, pred_gru))
    print(confusion_matrix(scaled_validation_labels, pred_gru))
    print(classification_report(scaled_validation_labels, pred_gru))
    print('GRU F1 score: {}'.format(f1_score(scaled_validation_labels, pred_gru, average='weighted')))
