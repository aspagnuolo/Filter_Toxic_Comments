import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, f1_score
import numpy as np
from sklearn.metrics import classification_report, hamming_loss, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate

def plot_comment_statistics(data_eda, distribution):
    """
    Plots statistics about comments including their distribution across categories
    and the distribution of their lengths.

    :param data_eda: A pandas DataFrame containing the comments data. 
                     Must include columns 'comment_text' and 'comment_length'.
    :param distribution: A pandas Series or similar containing the distribution of comments per category.
    """
    # Calculate the length of each comment
    data_eda['comment_length'] = data_eda['comment_text'].apply(len)

    # Initialize the plot
    plt.figure(figsize=(14, 6))

    # Plot the distribution of comments per category
    plt.subplot(1, 2, 1)
    sns.barplot(x=distribution.values, y=distribution.index)
    plt.title('Distribution of Comments by Category')
    plt.xlabel('Number of Comments')
    plt.ylabel('Category')

    # Plot the distribution of comment lengths
    plt.subplot(1, 2, 2)
    sns.histplot(data_eda['comment_length'], bins=50, color='skyblue')
    plt.title('Distribution of Comment Length')
    plt.xlabel('Comment Length')
    plt.ylabel('Frequency')
    plt.xlim(0, 2000)

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_co_occurrence_heatmap(data_eda, categories):
    """
    Plots a heatmap representing the co-occurrence matrix of harmful categories.

    :param data_eda: A pandas DataFrame containing the comments data.
                     Must include columns for each category in 'categories'.
    :param categories: A list of strings representing the names of the harmful categories.
    """
    # Calculate the co-occurrence matrix
    co_occurrence_matrix = data_eda[categories].T.dot(data_eda[categories])

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Plot the heatmap
    sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title('Co-occurrence Heatmap between Harmful Categories')
    plt.xlabel('Categories')
    plt.ylabel('Categories')

    # Display the plot
    plt.show()

def plot_damaging_comment_stats(data_eda, categories):
    """
    Plots statistics about damaging and non-damaging comments, including their distribution and the lengths.
    Also calculates and returns the correlation between comment length and whether it is damaging.

    :param data_eda: A pandas DataFrame containing the comments data.
                     Must include a 'comment_length' column and columns for each category in 'categories'.
    :param categories: A list of strings representing the names of the categories to be considered harmful.
    :return: A pandas DataFrame containing the correlation between comment length and its harmfulness.
    """
    # Calculate the total number of damaging vs non-damaging comments
    data_eda['is_damaging'] = data_eda[categories].sum(axis=1) > 0
    damaging_distribution = data_eda['is_damaging'].value_counts()

    # Calculate the length of damaging vs non-damaging comments
    damaging_comment_length = data_eda[data_eda['is_damaging']]['comment_length']
    non_damaging_comment_length = data_eda[~data_eda['is_damaging']]['comment_length']

    # Initialize the plot
    plt.figure(figsize=(18, 5))

    # Plot the distribution of damaging vs non-damaging comments
    plt.subplot(1, 3, 1)
    sns.barplot(x=damaging_distribution.index, y=damaging_distribution.values, palette='viridis')
    plt.title('Distribution of Damaging vs Non-Damaging Comments')
    plt.xlabel('Damaging')
    plt.ylabel('Number of Comments')
    plt.xticks([0, 1], ['Non-Damaging', 'Damaging'])

    # Plot the distribution of lengths for damaging comments
    plt.subplot(1, 3, 2)
    sns.histplot(damaging_comment_length, bins=50, color='red', label='Damaging')
    plt.title('Length of Damaging Comments')
    plt.xlabel('Comment Length')
    plt.ylabel('Frequency')
    plt.xlim(0, 2000)

    # Plot the distribution of lengths for non-damaging comments
    plt.subplot(1, 3, 3)
    sns.histplot(non_damaging_comment_length, bins=50, color='green', label='Non-Damaging')
    plt.title('Length of Non-Damaging Comments')
    plt.xlabel('Comment Length')
    plt.ylabel('Frequency')
    plt.xlim(0, 2000)

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

    # Calculate and return the correlation between comment length and its harmfulness
    correlation = data_eda[['comment_length', 'is_damaging']].corr()
    return correlation

def calculate_correlation_with_categories(data_eda, categories):
    """
    Calculates the correlation between the length of comments and each specified category of harmfulness.

    :param data_eda: A pandas DataFrame containing the comments data.
                     Must include a 'comment_length' column and columns for each category in 'categories'.
    :param categories: A list of strings representing the names of the harmful categories.
    :return: A pandas DataFrame containing the correlations between comment length and each harmful category.
    """
    # Calculate the correlation between comment length and each category of harmfulness
    correlation_with_categories = data_eda[['comment_length'] + categories].corr().iloc[1:, :1]

    # Return the correlation DataFrame
    return correlation_with_categories


def plot_comment_length_outliers(data_eda):
    """
    Creates a boxplot to visualize outliers in the lengths of damaging and non-damaging comments.

    :param data_eda: A pandas DataFrame containing the comments data.
                     Must include 'is_damaging' and 'comment_length' columns.
    """
    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='is_damaging', y='comment_length', data=data_eda, palette='coolwarm')
    plt.title('Boxplot of Comment Lengths by Harmfulness')
    plt.xlabel('Harmful')
    plt.ylabel('Comment Length')
    plt.xticks([0, 1], ['Non-Damaging', 'Damaging'])
    plt.ylim(0, 2000)

    # Display the plot
    plt.show()

def calculate_comment_length_statistics(data_eda):
    """
    Calculates descriptive statistics, interquartile ranges (IQR), and outlier limits for the lengths of damaging and non-damaging comments. It also counts the number of outliers in each category.

    :param data_eda: A pandas DataFrame containing the comments data.
                     Must include 'is_damaging' and 'comment_length' columns.
    :return: A dictionary containing the descriptive statistics, IQRs, outlier limits, and number of outliers for damaging and non-damaging comments.
    """
    # Separate the lengths of damaging and non-damaging comments
    damaging_comment_length = data_eda[data_eda['is_damaging']]['comment_length']
    non_damaging_comment_length = data_eda[~data_eda['is_damaging']]['comment_length']

    # Calculate descriptive statistics
    desc_stats_damaging = damaging_comment_length.describe()
    desc_stats_non_damaging = non_damaging_comment_length.describe()

    # Calculate IQR
    iqr_damaging = desc_stats_damaging['75%'] - desc_stats_damaging['25%']
    iqr_non_damaging = desc_stats_non_damaging['75%'] - desc_stats_non_damaging['25%']

    # Calculate outlier limits
    outlier_limit_damaging_high = desc_stats_damaging['75%'] + 1.5 * iqr_damaging
    outlier_limit_non_damaging_high = desc_stats_non_damaging['75%'] + 1.5 * iqr_non_damaging

    # Count outliers
    num_outliers_damaging = damaging_comment_length[damaging_comment_length > outlier_limit_damaging_high].count()
    num_outliers_non_damaging = non_damaging_comment_length[non_damaging_comment_length > outlier_limit_non_damaging_high].count()

    # Return all calculated statistics
    # Organize data into a DataFrame for clear presentation
    summary_table = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'IQR', 'Outlier Limit', 'Num Outliers'],
        'Damaging': list(desc_stats_damaging) + [iqr_damaging, outlier_limit_damaging_high, num_outliers_damaging],
        'Non-Damaging': list(desc_stats_non_damaging) + [iqr_non_damaging, outlier_limit_non_damaging_high, num_outliers_non_damaging]
    })
    # Return the summary table
    return summary_table   

def calculate_outliers_ratio(data_eda):
    """
    Calculates the ratio of outliers to the total number of comments for both damaging and non-damaging categories.

    :param data_eda: A pandas DataFrame containing the comments data.
                     Must include 'is_damaging' and 'comment_length' columns.
    :return: A dictionary containing the ratios of outliers for damaging and non-damaging comments.
    """
    # Separate the lengths of damaging and non-damaging comments
    damaging_comment_length = data_eda[data_eda['is_damaging']]['comment_length']
    non_damaging_comment_length = data_eda[~data_eda['is_damaging']]['comment_length']

    # Calculate the number of outliers based on previously established outlier limits
    desc_stats_damaging = damaging_comment_length.describe()
    desc_stats_non_damaging = non_damaging_comment_length.describe()
    iqr_damaging = desc_stats_damaging['75%'] - desc_stats_damaging['25%']
    iqr_non_damaging = desc_stats_non_damaging['75%'] - desc_stats_non_damaging['25%']
    outlier_limit_damaging_high = desc_stats_damaging['75%'] + 1.5 * iqr_damaging
    outlier_limit_non_damaging_high = desc_stats_non_damaging['75%'] + 1.5 * iqr_non_damaging
    num_outliers_damaging = damaging_comment_length[damaging_comment_length > outlier_limit_damaging_high].count()
    num_outliers_non_damaging = non_damaging_comment_length[non_damaging_comment_length > outlier_limit_non_damaging_high].count()

    # Calculate the ratios of outliers to total comments for each category
    rapporto_outliers_dannosi = num_outliers_damaging / len(damaging_comment_length)
    rapporto_outliers_non_dannosi = num_outliers_non_damaging / len(non_damaging_comment_length)

    # Return the calculated ratios
    return {
        'Ratio of Outliers (Damaging)': rapporto_outliers_dannosi,
        'Ratio of Outliers (Non-Damaging)': rapporto_outliers_non_dannosi
    }

def preprocess_text(text):
    """
    Preprocesses the given text by removing URLs, HTML tags, non-alphabetic characters,
    tokenizing, removing stopwords, and lemmatizing the words.

    :param text: A string containing the text to be preprocessed.
    :return: A string of the preprocessed text.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]

    # Reconstruct the preprocessed text
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

def remove_empty_or_space_only_rows(data):
    """
    Removes rows from the DataFrame where the 'cleaned_comment_text' column contains only spaces or is empty.
    
    :param data: A pandas DataFrame expected to contain a column named 'cleaned_comment_text'.
    :return: A tuple containing two DataFrames: 
             1. The rows that were removed (having only spaces or being empty in 'cleaned_comment_text').
             2. The cleaned DataFrame with these rows removed.
    """
    # Identify rows with 'cleaned_comment_text' that are either empty or contain only spaces
    rows_with_only_spaces = data[data['cleaned_comment_text'].apply(lambda x: x.isspace() or not x)]
    
    # Remove these rows from the dataset
    data = data[~data['cleaned_comment_text'].apply(lambda x: x.isspace() or not x)]
    
    return data

def calculate_class_weights(y_train):
    """
    Calculates inverse class weights based on the frequencies of classes in the training set. This is often used to 
    handle imbalanced datasets by giving more weight to underrepresented classes during model training.
    
    :param y_train: A pandas DataFrame or numpy array containing the training labels. 
                    Each column represents a class, and each row represents a sample with binary indicators for class membership.
    :return: A dictionary with class indices as keys and calculated inverse weights as values.
    """
    # Calculate the frequency of each class
    class_frequencies = y_train.sum(axis=0)
    
    # Calculate inverse weights for each class
    weights = [len(y_train) / (len(class_frequencies) * frequency) if frequency > 0 else 0 for frequency in class_frequencies]
    
    # Create a dictionary mapping class index to its weight
    class_weights = {i: weight for i, weight in enumerate(weights)}
    
    # Optionally, you can print the calculated weights
    #print("Calculated weights for classes:", class_weights)
    
    return class_weights

def predict_and_save_f1_scores(model, X_test, y_test, model_name, f1_scores_collection):
    """
    Predicts using the model, calculates F1 scores for each label, includes the weighted average F1 score,
    saves them with a given model name, and returns the predictions. This function works silently without
    printing out information.

    :param model: The trained model to use for predictions.
    :param X_test: The test set features.
    :param y_test: The true labels for the test set.
    :param model_name: A custom name for the model to use when saving the scores.
    :param f1_scores_collection: A dictionary to store the F1 scores for different models.
    :return: A tuple containing the model's predictions and the updated F1 scores collection.
    """
    # Define the labels internally
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Make predictions
    y_test_pred = model.predict(X_test)
    
    # Calculate F1 scores for each label
    f1_scores = f1_score(y_test, y_test_pred, average=None)
    
    # Create a dictionary mapping each label to its F1 score
    f1_scores_dict = dict(zip(labels, f1_scores))
    
    # Get the classification report in dictionary form
    report_dict = classification_report(y_test, y_test_pred, target_names=labels, output_dict=True)
    
    # Extract the weighted average F1 score
    weighted_avg_f1_score = report_dict['weighted avg']['f1-score']
    
    # Update the F1 scores dictionary to include the weighted average F1 score
    f1_scores_dict['weighted_avg'] = weighted_avg_f1_score
    
    # Update the F1 scores collection with the new scores
    f1_scores_collection[model_name] = f1_scores_dict
    
    # Return the predictions and the (optionally updated) F1 scores collection
    return y_test_pred, f1_scores_collection

def evaluate_model_performance(y_test, y_test_pred):
    """
    Evaluates the performance of a multi-label classification model by displaying the classification report,
    computing the Hamming loss, and plotting heatmaps of confusion matrices for each label.
    
    :param y_test: True labels for the test set.
    :param y_test_pred: Predicted labels for the test set.
    """
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Print classification report
    print("Classification Report on the test set:")
    print(classification_report(y_test, y_test_pred, zero_division=0, target_names=labels))
    
    # Print Hamming loss
    print("Hamming Loss:", hamming_loss(y_test, y_test_pred))
    
    # Plot confusion matrices
    plt.figure(figsize=(15, 10))
    for i, label in enumerate(labels):
        plt.subplot(2, 3, i+1)
        sns.heatmap(confusion_matrix(y_test[:, i], y_test_pred[:, i]), annot=True, fmt='d', cmap='Blues')
        plt.title(label)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def prepare_data(data, test_size=0.05, val_size=0.05):
    input_texts = data['cleaned_comment_text'].values
    labels = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.astype(np.float32)
    X_train_val, X_test, y_train_val, y_test = train_test_split(input_texts, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(vocabulary_size=10000, max_comment_length=100):
    comment_vectorizer = TextVectorization(max_tokens=vocabulary_size, output_mode='int', output_sequence_length=max_comment_length)
    input_layer = Input(shape=(1,), dtype=tf.string)
    processed_text = comment_vectorizer(input_layer)
    embedding = Embedding(input_dim=vocabulary_size + 1, output_dim=128)(processed_text)
    bi_lstm = Bidirectional(LSTM(24, return_sequences=False))(embedding)
    intermediate_dense = Dense(20, activation='relu')(bi_lstm)
    output_dense = Dense(6, activation='sigmoid')(intermediate_dense)
    model = Model(input_layer, output_dense)
    model.compile(loss='BinaryCrossentropy', optimizer=Adam(), metrics=[Precision(), Recall(), F1Score(num_classes=6, average='micro', threshold=0.5), AUC(multi_label=True)])
    return model, comment_vectorizer

def plot_custom_metrics(history):
    metrics = ['precision', 'recall', 'f1_score', 'auc']
    plt.figure(figsize=(10, 8))

    for n, metric in enumerate(metrics):
        name = metric.capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.epoch, history.history[val_metric], color='orange', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        
        if metric in ['precision', 'recall', 'f1_score']:
            plt.ylim([0, 1])
        plt.legend()
    
    plt.tight_layout()

def predict_with_threshold(model, dataset, threshold=0.4):
    """
    Predicts labels for the given dataset using the specified model and applies a threshold to determine the final binary predictions.

    :param model: The trained TensorFlow/Keras model to use for prediction.
    :param dataset: The dataset to predict on, typically a tf.data.Dataset object.
    :param threshold: The threshold to use for converting probabilities to binary predictions. Defaults to 0.4.
    :return: An array of binary predictions.
    """
    # Obtain probability predictions
    y_pred_probs = model.predict(dataset)
    
    # Apply the threshold to get binary predictions
    y_pred = (y_pred_probs > threshold).astype(int)
    
    return y_pred

def update_f1_scores(y_true, y_pred, model_name, f1_scores_collection, labels=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    """
    Calculate F1 scores for each label and the weighted average F1 score, then update a given dictionary.

    :param y_true: The true labels.
    :param y_pred: The predicted labels (binary).
    :param model_name: The name of the model (used as a key in the dictionary).
    :param f1_scores_collection: The dictionary to be updated with F1 scores.
    :param labels: A list of label names corresponding to the columns in y_true and y_pred.
    """
    # Calculate F1 score for each label
    f1_scores_each_label = f1_score(y_true, y_pred, average=None)
    
    # Calculate weighted average F1 score
    f1_score_weighted_avg = f1_score(y_true, y_pred, average='weighted')
    
    # Create a dictionary of label F1 scores and the weighted average
    f1_scores_dict = {label: f1 for label, f1 in zip(labels, f1_scores_each_label)}
    f1_scores_dict['weighted_avg'] = f1_score_weighted_avg
    
    # Update the collection with the new F1 scores
    f1_scores_collection[model_name] = f1_scores_dict

def create_second_model(comment_vectorizer, vocabulary_size=10000, embedding_dim=128, lstm_units=24, dense_units=20, dropout_rate=0.5):
    """
    Creates and compiles a second RNN model with dropout and batch normalization layers.

    :param comment_vectorizer: Pre-configured TextVectorization layer.
    :param vocabulary_size: Size of the text vocabulary.
    :param embedding_dim: Dimensionality of the embedding layer.
    :param lstm_units: Number of units in the LSTM layer.
    :param dense_units: Number of units in the intermediate dense layer.
    :param dropout_rate: Dropout rate for regularization.
    :return: Compiled Keras model.
    """
    input_layer = Input(shape=(1,), dtype=tf.string, name='input_text')
    processed_text = comment_vectorizer(input_layer)
    embedding = Embedding(input_dim=vocabulary_size+1, output_dim=embedding_dim, name='text_embedding_layer')(processed_text)
    bi_lstm = Bidirectional(LSTM(lstm_units, return_sequences=False), name='bi_lstm_layer')(embedding)
    dropout_1 = Dropout(dropout_rate, name='dropout_1')(bi_lstm)
    batch_norm_1 = BatchNormalization(name='batch_norm_1')(dropout_1)
    intermediate_dense = Dense(dense_units, activation='relu', name='intermediate_dense_layer')(batch_norm_1)
    dropout_2 = Dropout(dropout_rate, name='dropout_2')(intermediate_dense)
    batch_norm_2 = BatchNormalization(name='batch_norm_2')(dropout_2)
    output_dense = Dense(6, activation='sigmoid', name='output_dense_layer')(batch_norm_2)

    model = Model(input_layer, output_dense, name='model2')

    # Assuming METRICS is defined outside this function as per your initial model setup
    METRICS = [Precision(name='precision'), Recall(name='recall'), F1Score(num_classes=6, average='micro', threshold=0.5, name='f1_score'), AUC(name='auc', multi_label=True)]

    model.compile(loss='BinaryCrossentropy', optimizer=Adam(), metrics=METRICS)

    return model

def prepare_data_numerical_input(data, test_size=0.05, val_size=0.05):
    input_texts = data['cleaned_comment_text'].values
    sum_injurious = data['sum_injurious'].values.astype(np.float32).reshape(-1, 1)  # Reshape per renderlo compatibile con l'input del modello
    labels = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.astype(np.float32)
    # Split con entrambi gli input
    X_train_val, X_test, y_train_val, y_test, sum_injurious_train_val, sum_injurious_test = train_test_split(input_texts, labels, sum_injurious, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val, sum_injurious_train, sum_injurious_val = train_test_split(X_train_val, y_train_val, sum_injurious_train_val, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, sum_injurious_train, sum_injurious_val, sum_injurious_test

def create_model_with_numerical_input(comment_vectorizer, vocabulary_size=10000, embedding_dim=128, lstm_units=24, intermediate_dense_units=64):
    input_text = Input(shape=(1,), dtype=tf.string, name='input_text')
    input_numerical = Input(shape=(1,), dtype=tf.float32, name='input_sum_injurious')
    
    processed_text = comment_vectorizer(input_text)
    embedding = Embedding(input_dim=vocabulary_size + 1, output_dim=embedding_dim)(processed_text)
    bi_lstm = Bidirectional(LSTM(lstm_units, return_sequences=False))(embedding)
    
    concatenated = Concatenate()([bi_lstm, input_numerical])
    
    # Intermediate dense layer added before the final output layer
    intermediate_dense = Dense(intermediate_dense_units, activation='relu')(concatenated)
    
    # Output layer
    output_dense = Dense(6, activation='sigmoid')(intermediate_dense)

    METRICS = [Precision(name='precision'), Recall(name='recall'), F1Score(num_classes=6, average='micro', threshold=0.5, name='f1_score'), AUC(name='auc', multi_label=True)]
    
    model = Model(inputs=[input_text, input_numerical], outputs=output_dense)
    model.compile(loss='BinaryCrossentropy', optimizer=Adam(), metrics=METRICS)
    
    return model

def get_custom_loss(class_weights):
    # Convert class_weights to a Keras tensor
    class_weights_tensor = K.constant([class_weights[i] for i in range(len(class_weights))])

    def custom_weighted_binary_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * class_weights_tensor + (1. - y_true)
        weighted_bce = weight_vector * bce

        return K.mean(weighted_bce)

    return custom_weighted_binary_crossentropy

def create_model_with_numerical_input_custom_loss(comment_vectorizer, class_weights, vocabulary_size=10000, embedding_dim=128, lstm_units=24, intermediate_dense_units=64):
    # Define the input layers
    input_text = Input(shape=(1,), dtype=tf.string, name='input_text')
    input_numerical = Input(shape=(1,), dtype=tf.float32, name='input_sum_injurious')
    
    # Text processing pipeline
    processed_text = comment_vectorizer(input_text)
    embedding = Embedding(input_dim=vocabulary_size + 1, output_dim=embedding_dim)(processed_text)
    bi_lstm = Bidirectional(LSTM(lstm_units, return_sequences=False))(embedding)
    
    # Combine text and numerical features
    concatenated = Concatenate()([bi_lstm, input_numerical])
    intermediate_dense = Dense(intermediate_dense_units, activation='relu')(concatenated)
    output_dense = Dense(6, activation='sigmoid')(intermediate_dense)

    # Compile the model with the custom loss
    custom_loss = get_custom_loss(class_weights)
    METRICS = [Precision(name='precision'), Recall(name='recall'), F1Score(num_classes=6, average='micro', threshold=0.5, name='f1_score'), AUC(name='auc', multi_label=True)]
    
    model = Model(inputs=[input_text, input_numerical], outputs=output_dense)
    model.compile(loss=custom_loss, optimizer=Adam(), metrics=METRICS)
    
    return model

def predict_single_example(model, text, sum_injurious):
    # Creiamo un dataset per un singolo esempio
    single_example_dataset = tf.data.Dataset.from_tensor_slices(((text, sum_injurious),))
    single_example_dataset = single_example_dataset.batch(1)  # Il batch deve essere 1 se stai facendo una previsione su un singolo esempio

    # Facciamo la previsione
    prediction = model.predict(single_example_dataset)

    # Continua come prima
    threshold = 0.4
    predicted_labels = (prediction > threshold).astype(int)
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predicted_label_names = [label_names[i] for i, val in enumerate(predicted_labels[0]) if val == 1]
    
    return predicted_label_names

