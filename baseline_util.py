#!/usr/bin/env python3
"""
This module defines functions that are used by more than one baseline script.
Functions defined:
    -combine_feature_matrices
    -display_classification_results
    -get_features_and_labels
    -ivectors_dict_to_feature_matrix
    -load_feature_vectors
    -pretty_print_cm
    -write_feature_files
    -write_predictions_file
"""
from zipfile import ZipFile, ZIP_DEFLATED
from io import StringIO
import csv
from os import path
import numpy as np
from time import strftime
from scipy.sparse import hstack as sparse_hstack
from scipy import hstack as dense_hstack
from sklearn import metrics
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer

import json

SCRIPT_DIR = path.dirname(path.realpath(__file__))

# valid labels
CLASS_LABELS = [
#    'ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]



def combine_feature_matrices(matrix1, matrix2, sparse=False):
    """
    Combine two sparse feature matrices by concatenating each corresponding row.
    Note that the rows in `sparse_matrix1` should correspond with those in
    `sparse_matrix2` (and `sparse_matrix1.shape[1]` should be equal to
    `sparse_matrix2.shape[1]`).

    Parameters
    ----------
    matrix1: sparse csr_matrix
        Feature matrix for a dataset.

    matrix2: sparse csr_matrix
        Feature matrix of another dataset with the same number of rows, and
        whose rows should be in the same order as those in `sparse_matrix1`.

    sparse: boolean
        Indicates if the matrices are sparse or not.

    Returns
    -------
    csr_sparse_matrix
        A single sparse matrix representing the concatenation of the
        corresponding rows of the input matrices.

    """
    hstack = sparse_hstack if sparse else dense_hstack
    assert (matrix1.shape[0] == matrix2.shape[0])
    return hstack((matrix1, matrix2))


def display_classification_results(encoded_test_labels, predictions):
    """
    Print confusion matrix and classification report.

    Parameters
    ----------
    encoded_test_labels: list of ints
        True class values as ints corresponding to the index of the label in
        `CLASS_LABELS`.

    predictions: list of list of ints
        Predictions as ints corresponding to the index of the label in
        `CLASS_LABELS`. Should be in the same order as `encoded_test_labels`.

    Returns
    -------
    None. Prints results only.

    """
    if -1 not in encoded_test_labels:
        print("\nConfusion Matrix:\n")
        cm = metrics.confusion_matrix(encoded_test_labels, predictions).tolist()
        pretty_print_cm(cm, CLASS_LABELS)
        print("\nClassification Results:\n")
        print(metrics.classification_report(encoded_test_labels,
                                            predictions,
                                            target_names=CLASS_LABELS))
    else:
        print("The test set labels aren't known, cannot print accuracy report.")


def get_features_and_labels(train_partition, test_partition,
                            training_feature_file, test_feature_file,
                            baseline='essays', preprocessor='tokenized',
                            vectorizer=None, transformer=None):
    """
    If no feature files are provided, generates feature matrices for training
    and test data. By default, it assumes the use of the text in the
    'tokenized' directory (vs 'original'). This is also the directory pointed
    to in the labels files. To use your own processing pipeline, write the
    processed essays to a new directory under "../data/essays/<train_dir>/" and
    "../data/essays/<test_dir>/", and modify the corresponding labels files
    with the correct "essay_path" column.

    If precomputed feature files are provided, this function reads features and
    labels for the training and test sets instead of creating new ones.

    NOTE: `training_feature_file` and `test_feature_file` must be provided
    together. If only one is provided, all features will be recomputed to avoid
    dimension mismatch.

    Parameters
    -----------
    train_partition: str
        String indicating the name of the training directory (e.g. 'train').
        This directory should exist in "../data/essays/" and "../data/labels/".
        It will also be created in "../data/features/" to output the training
        features

    test_partition: str
        String indicating the name of the testing directory (e.g. 'dev' or
        'test'). This directory should exist in "../data/essays/" and
        "../data/labels/". It will also be created in "../data/features/" to
        output the test features.

    training_feature_file: str
        Path to saved training feature file (must be svm_light format).

    test_feature_file: str
        Path to saved test feature file (must be svm_light format).

    baseline: str
        String indicating which baseline directory is to be used. Should be
        either "essays" or "speech_transcriptions".

    preprocessor: str, 'tokenized' by default
        Name of directory under '../data/essays/<partition_name>/ where the
        processed essay text is stored.
        Options:
            'original': raw text
            'tokenized': segmented on sentence boundaries (one sentence per
                         line) and word-tokenized (tokens surrounded by white
                         space).
        HINT: You can use a custom preprocessing pipeline by saving the
        processed data in
        "../data/essays/<partition>/<custom_preprocessor_name>/"
        for train and test partitions.

    vectorizer: Vectorizer object or NoneType, None by default
        Object to convert a collection of text documents to a matrix. Must
        implement fit, transform, and fit_transform methods. If no vectorizer
        is provided, this function will use sklearn's CountVectorizer as
        default.

    transformer: sklearn transformer object or NoneType, None by default.
        Object to normalize feature matrices. Should implement fit_transform
        and transform methods.

    Returns
    -------
    tuple (length 2)
        -list of
        [training matrix, training labels as ints, training labels as strings]
        -list of
        [test matrix, test labels as ints, test labels as strings]

    """
    train_labels_path = ("{script_dir}/../data/labels/{train}"
                         "/labels.{train}.csv".format(train=train_partition,
                                                      script_dir=SCRIPT_DIR))

    train_data_path = ("{script_dir}/../data/essays/{train}/tokenized/"
                       .format(train=train_partition, script_dir=SCRIPT_DIR))

#    test_labels_path = ("{script_dir}/../data/labels/{test}/labels.{test}.csv"
 #                       .format(test=test_partition, script_dir=SCRIPT_DIR))

    test_labels_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
	                         .format(test=test_partition, task="essay" if baseline == "essays" else "speech", 
	                                 script_dir=SCRIPT_DIR))
    
    test_data_path = ("{script_dir}/../data/essays/{test}/tokenized"
                      .format(test=test_partition, script_dir=SCRIPT_DIR))

    path_and_descriptor_list = [(train_labels_path, "training labels file"),
                                (train_data_path, "training data directory"),
                                (test_labels_path, "testing labels file"),
                                (test_data_path, "testing data directory")]

    for path_, path_descriptor in path_and_descriptor_list:
        if not path.exists(path_):
            raise Exception("Could not find {desc}: {pth}"
                            .format(desc=path_descriptor, pth=path_))

    #
    #  Read labels files. If feature files provided, `training_files` and
    # `test_files` below will be ignored.
    #
    with open(train_labels_path) as train_labels_f, \
            open(test_labels_path) as test_labels_f:
        essay_path_train = ('{script_dir}/../data/{bl}/{train}/{preproc}'
                            .format(script_dir=SCRIPT_DIR,
                                    bl=baseline,
                                    train=train_partition,
                                    preproc=preprocessor))

        essay_path_test = ('{script_dir}/../data/{bl}/{test}/{preproc}'
                           .format(script_dir=SCRIPT_DIR,
                                   bl=baseline,
                                   test=test_partition,
                                   preproc=preprocessor))

        training_files, training_labels = \
            zip(*[(path.join(essay_path_train, row['test_taker_id'] + '.txt'),
                   row['L1']) for row in csv.DictReader(train_labels_f)])

        test_files, test_labels = \
            zip(*[(path.join(essay_path_test, row['test_taker_id'] + '.txt'),
                   row['L1']) for row in csv.DictReader(test_labels_f)])

    #
    #  Verify that either both or neither of training/test feature files are
    # provided
    #
    if bool(training_feature_file) != bool(test_feature_file):
        print("Feature files were not provided for both test and train "
              "partitions. Generating default unigram features now.")

    #
    #  If feature files provided, get features and labels from them
    #
    elif training_feature_file and test_feature_file:

        train_matrix, \
        encoded_train_labels = load_svmlight_file(training_feature_file)

        original_training_labels = tuple([CLASS_LABELS[int(i)]
                                          for i in encoded_train_labels])
        if original_training_labels != training_labels:
            raise Exception("Training labels in feature file do not match those"
                            " in the labels file.")

        n_dims = train_matrix.shape[1]

        test_matrix, \
        encoded_test_labels = load_svmlight_file(test_feature_file,
                                                 n_features=n_dims,
                                                 zero_based=True)

        original_test_labels = tuple([CLASS_LABELS[int(i)]
                                      for i in encoded_test_labels])
        if original_test_labels != test_labels:
            raise Exception("Test labels in feature file do not match those in"
                            " the labels file.")

        return [(train_matrix, encoded_train_labels, original_training_labels),
                (test_matrix, encoded_test_labels, original_test_labels)]

    #
    #  If no feature files provided, create feature matrix from the data files
    #
    print("Found {} text files in {} and {} in {}"
          .format(len(training_files), train_partition,
                  len(test_files), test_partition))
    print("Loading training and testing data from {} & {}"
          .format(train_partition, test_partition))

    train_matrix, \
    encoded_train_labels, \
    vectorizer = load_feature_vectors(training_files,
                                      training_labels,
                                      vectorizer)
    test_matrix, \
    encoded_test_labels, _ = load_feature_vectors(test_files,
                                                  test_labels,
                                                  vectorizer)

    if transformer is not None:
        train_matrix = transformer.fit_transform(train_matrix)
        test_matrix = transformer.transform(test_matrix)

    return [(train_matrix, encoded_train_labels, training_labels),
            (test_matrix, encoded_test_labels, test_labels)]


def ivectors_dict_to_features(ivectors_dict,
                              partition_name,
                              mat_format=np.matrix):
    """
    Given a map from test-taker id to ivector, arrange the ivectors in the
    order corresponding with that in the labels file for the partition in
    question. Supply the return type if desired, otherwise the feature matrix
    will be a numpy matrix.

    Parameters
    ----------
    ivectors_dict: dict of str -> list
        Maps test-taker ids to ivector lists.

    partition_name: str
        Name of partition directory (e.g. 'train', 'dev', 'test', etc.)

    mat_format: type
        To return the feature matrix as a specific type, specify it here.
        Defaults to numpy.matrix.

    Returns
    -------
    Feature matrix in the specified return type.

    """
    #labels_file_path = ("{script_dir}/../data/labels/{part}/labels.{part}.csv"
     #                   .format(script_dir=SCRIPT_DIR, part=partition_name))
    
    #if partition_name == 'test':
    #        labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv".format(test=partition_name, task="speech", script_dir=SCRIPT_DIR))
    #else:
	  #      labels_file_path = ("{script_dir}/../data/labels/{part}/labels.{part}.csv"
	   #                     .format(script_dir=SCRIPT_DIR, part=partition_name))

    if partition_name == 'test':
            labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv".format(test=partition_name, task="fusion", script_dir=SCRIPT_DIR))
    else:
	        labels_file_path = ("{script_dir}/../data/labels/{part}/labels.{part}.csv"
	                        .format(script_dir=SCRIPT_DIR, part=partition_name))

    spkr_ids = [row['test_taker_id']
                for row in csv.DictReader(open(labels_file_path))]

    matrix = [ivectors_dict[spkr_id] for spkr_id in spkr_ids]
    return mat_format(matrix)


def load_feature_vectors(file_list, labels, vectorizer=None):
    """
    This function creates a document-term matrix, given a list of file names
    and a list of labels.

    If a feature vectorizer has been created, it can be passed in as
    `vectorizer` to be used. Otherwise one will be instantiated.

    Parameters
    ----------
    file_list: list of str
        File names to be used for creating matrix.

    labels: list of str
        Correct class labels corresponding with the essays in `file_list`.
        These will be encoded as integers for saving in svm_light format.

    vectorizer: Vectorizer object or NoneType, None by default.
        Object to convert a collection of text documents to a matrix. Must have
        fit, transform, and fit_transform methods implemented. If no vectorizer
        is provided, this function will use sklearn's CountVectorizer. The
        vectorizer that is fit on the training data should be re-used for the
        testing data.

    Returns
    -------
    tuple (length 4)
        -doc-term matrix (numpy array),
        -list of correct labels encoded as ints,
        -list of labels as strings,
        -vectorizer instance

    """
    # convert label strings to integers
    labels_encoded = [CLASS_LABELS.index(label) for label in labels]
    if vectorizer is None:
        vectorizer = CountVectorizer(input="filename")  # create a new one
        doc_term_matrix = vectorizer.fit_transform(file_list)
    elif not hasattr(vectorizer, 'vocabulary_'):
        doc_term_matrix = vectorizer.fit_transform(file_list)
    else:
        doc_term_matrix = vectorizer.transform(file_list)
    print("Created a document-term matrix with %d rows and %d columns."
          % (doc_term_matrix.shape[0], doc_term_matrix.shape[1]))

    return doc_term_matrix.astype(float), labels_encoded, vectorizer


def pretty_print_cm(cm, class_labels):
    """
    Print a Confusion Matrix in an easy-to-read format.

    Parameters
    ----------
    cm: list of lists
        Confusion matrix

    class_labels: iterable of strings
        Labels corresponding to the indeces of the confusion matrix.

    Returns
    -------
    None. Prints confusion matrix only.

    """
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))


def write_feature_files(partition_name,
                        feature_outfile_name,
                        baseline,
                        matrix,
                        encoded_labels):
    """
    Save a feature matrix to a file in svmlight format under
    "../data/features/<baseline>/<partition_name>/".

    Parameters
    ----------
    partition_name: str
        Name of partition to which the feature matrix corresponds.

    feature_outfile_name: str or None
        Name of the feature file. If None, the name will be the timestamp.

    baseline: str
        Name of baseline.

    matrix: sparse csr_matrix
        Feature matrix to be written.

    encoded_labels: list of ints
        True class values, encoded as ints.

    Returns
    -------
    None. Writes file only.

    """
    outfile_name = (strftime("{}-%Y-%m-%d-%H.%M.%S.features"
                             .format(partition_name))
                    if feature_outfile_name is None
                    else "{}-{}".format(partition_name, feature_outfile_name))

    outfile = strftime("{script_dir}/../data/features/{bl}/{train}"
                       "/{outfile_name}".format(script_dir=SCRIPT_DIR,
                                                bl=baseline,
                                                train=partition_name,
                                                outfile_name=outfile_name))

    dump_svmlight_file(matrix, encoded_labels, outfile)
    print("Wrote {par} features to".format(par=partition_name),
          outfile.replace(SCRIPT_DIR, '')[1:])  # path relative to script


def write_predictions_file(predictions,
                           test_partition_name,
                           predictions_outfile_name,
                           baseline):
    """
    Write a csv that is a copy of the test-set labels file, with an added
    'prediction' field containing the model's predictions.

    Parameters
    ----------
    predictions: list or tuple
        Model's class predictions. Their order should correspond with the order
        in the labels file.

    test_partition_name: str
        Name of test partition directory (e.g. "dev" or "test")

    predictions_outfile_name: str
        Name of predictions output file

    baseline: str
        Name of baseline. This will be used to write the predictions file to
        the appropriate directory.

    Returns
    -------
    None. Writes file only and prints completed status only.

    """
#    labels_file_path = ('{script_dir}/../data/labels/{test}/labels.{test}.csv'
#                        .format(script_dir=SCRIPT_DIR,
#                                test=test_partition_name))

#    labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
#                     .format(test=test_partition_name, task="essay" if baseline == "essays" else "speech", 
#                             script_dir=SCRIPT_DIR))

    labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
                     .format(test=test_partition_name, task="fusion", 
                             script_dir=SCRIPT_DIR))

    if predictions_outfile_name is None:
        predictions_file_name = strftime("predictions-%Y-%m-%d-%H.%M.%S.csv")
    else:
        predictions_file_name = predictions_outfile_name

    outfile = ('{script_dir}/../predictions/{bl}/{pred_file}'
               .format(script_dir=SCRIPT_DIR,
                       bl=baseline,
                       pred_file=predictions_file_name))

    with open(outfile, 'w+', newline='', encoding='utf8') as output_file:
        file_writer = csv.writer(output_file)
        with open(labels_file_path, encoding='utf-8') as labels_file:
            label_rows = [row for row in csv.reader(labels_file)]
            label_rows[0].append('prediction')
            for i, row in enumerate(label_rows[1:]):
                encoded_prediction = int(predictions[i])
                prediction = CLASS_LABELS[encoded_prediction]
                row.append(prediction)
        file_writer.writerows(label_rows)

    print("Predictions written to", outfile.replace(SCRIPT_DIR, '')[1:],
          "({:d} lines)".format(len(predictions)))

### PI modifications

def get_labels(train_partition, test_partition, baseline):
    """
    Parameters
    -----------
    train_partition: str
        String indicating the name of the training directory (e.g. 'train').
        This directory should exist in "../data/essays/" and "../data/labels/".
        
    test_partition: str
        String indicating the name of the testing directory (e.g. 'dev' or
        'test'). This directory should exist in "../data/essays/" and
        "../data/labels/". It will also be created in "../data/features/" to
        output the test features.

        
    Returns
    -------
    tuple (length 2)
        -list of
        [training labels as ints, training labels as strings]
        -list of
        [test labels as ints, test labels as strings]

    """
    train_labels_path = ("{script_dir}/../data/labels/{train}"
                         "/labels.{train}.csv".format(train=train_partition,
                                                      script_dir=SCRIPT_DIR))

   # test_labels_path = ("{script_dir}/../data/labels/{test}/labels.{test}.csv"
   #                     .format(test=test_partition, script_dir=SCRIPT_DIR))
   
   # test_labels_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
	 #                        .format(test=test_partition, task="essay" if baseline == "essays" else "speech", 
	 #                                script_dir=SCRIPT_DIR))
   
    test_labels_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
	                         .format(test=test_partition, task="fusion", 
	                                 script_dir=SCRIPT_DIR))

    
    path_and_descriptor_list = [(train_labels_path, "training labels file"),
                                (test_labels_path, "testing labels file")]
                                

    for path_, path_descriptor in path_and_descriptor_list:
        if not path.exists(path_):
            raise Exception("Could not find {desc}: {pth}"
                            .format(desc=path_descriptor, pth=path_))

    #
    #  Read labels files. 
    
    with open(train_labels_path) as train_labels_f, \
            open(test_labels_path) as test_labels_f:
                        
                train_labels = [row['L1'] for row in csv.DictReader(train_labels_f)]
                test_labels = [row['L1'] for row in csv.DictReader(test_labels_f)]
                
    
                encoded_train_labels = [CLASS_LABELS.index(label) for label in train_labels]
                encoded_test_labels = [CLASS_LABELS.index(label) for label in test_labels]
                
                return [(encoded_train_labels, train_labels),
                        (encoded_test_labels, test_labels)]

def partition_ivectors(train_partition, test_partition):
    
    ivectors_path = ('../data/features/ivectors/train/'
                         'ivectors.json')
    
    print ("Loading the full ivectors ...")
  
    full_ivectors_dict = json.load(open(ivectors_path))            ###   loading the full ivector json          
    
    train_ivectors_dict = {}
    test_ivectors_dict = {}
 
    train_labels_path = ("{script_dir}/../data/labels/{train}"                #### Loading the  t_train and t_dev label files
                         "/labels.{train}.csv".format(train=train_partition,
                                                      script_dir=SCRIPT_DIR))
    
    test_labels_path = ("{script_dir}/../data/labels/{test}/labels.{test}.csv"
                        .format(test=test_partition, script_dir=SCRIPT_DIR))
    
    with open(train_labels_path) as train_labels_f, \
        open(test_labels_path) as test_labels_f:         
                
                train_speaker_ids = [row['test_taker_id'] for row in csv.DictReader(train_labels_f)]
                test_speaker_ids = [row['test_taker_id'] for row in csv.DictReader(test_labels_f)]     
    
        
    # spkr_ids_train = [row['test_taker_id']
    #            for row in csv.DictReader(open(train_labels_path))]

    for key in full_ivectors_dict.keys():
        if key in train_speaker_ids:
            train_ivectors_dict[key]=full_ivectors_dict[key]
        elif key in test_speaker_ids:
            test_ivectors_dict[key]=full_ivectors_dict[key]
        else:
            print("Unknown ID ", key)
    
    return[train_ivectors_dict,test_ivectors_dict]   
    #return mat_format(matrix)
    
def get_features_from_text(train_partition, test_partition,
                           baseline, preprocessor='tokenized',
                            vectorizer=None, transformer=None):

    train_labels_path = ("{script_dir}/../data/labels/{train}"
                         "/labels.{train}.csv".format(train=train_partition,
                                                      script_dir=SCRIPT_DIR))

    train_data_path = ("{script_dir}/../data/{text_data}/{train}/tokenized/"
                       .format(text_data=baseline, train=train_partition, script_dir=SCRIPT_DIR))

    #test_labels_path = ("{script_dir}/../data/labels/{test}/labels.{test}.csv"
    #                    .format(test=test_partition, script_dir=SCRIPT_DIR))
    
    #test_labels_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
	  #                       .format(test=test_partition, task="essay" if baseline == "essays" else "speech", 
	  #                               script_dir=SCRIPT_DIR))
    
    test_labels_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
	                         .format(test=test_partition, task="fusion", 
	                                 script_dir=SCRIPT_DIR))

    test_data_path = ("{script_dir}/../data/{text_data}/{test}/tokenized"
                      .format(text_data=baseline, test=test_partition, script_dir=SCRIPT_DIR))

    path_and_descriptor_list = [(train_labels_path, "training labels file"),
                                (train_data_path, "training data directory"),
                                (test_labels_path, "testing labels file"),
                                (test_data_path, "testing data directory")]

    for path_, path_descriptor in path_and_descriptor_list:
        if not path.exists(path_):
            raise Exception("Could not find {desc}: {pth}"
                            .format(desc=path_descriptor, pth=path_))

    #
    #  Read labels files. If feature files provided, `training_files` and
    # `test_files` below will be ignored.
    #
    with open(train_labels_path) as train_labels_f, \
            open(test_labels_path) as test_labels_f:
        essay_path_train = ('{script_dir}/../data/{text_data}/{train}/{preproc}'
                            .format(script_dir=SCRIPT_DIR,
                                    text_data=baseline,
                                    train=train_partition,
                                    preproc=preprocessor))

        essay_path_test = ('{script_dir}/../data/{text_data}/{test}/{preproc}'
                           .format(script_dir=SCRIPT_DIR,
                                   text_data=baseline,
                                   test=test_partition,
                                   preproc=preprocessor))

        training_files, training_labels = \
            zip(*[(path.join(essay_path_train, row['test_taker_id'] + '.txt'),
                   row['L1']) for row in csv.DictReader(train_labels_f)])

        test_files, test_labels = \
            zip(*[(path.join(essay_path_test, row['test_taker_id'] + '.txt'),
                   row['L1']) for row in csv.DictReader(test_labels_f)])
   
    print("Found {} text files in {} and {} in {}"
          .format(len(training_files), train_partition,
                  len(test_files), test_partition))
    
    print("Loading training and testing data from {} & {}"
          .format(train_partition, test_partition))

    train_matrix, \
    encoded_train_labels, \
    vectorizer = create_feature_vectors(training_files,
                                      training_labels,
                                      vectorizer)
    test_matrix, \
    encoded_test_labels, _ = create_feature_vectors(test_files,
                                                  test_labels,
                                                  vectorizer)

    if transformer is not None:
        train_matrix = transformer.fit_transform(train_matrix)
        test_matrix = transformer.transform(test_matrix)

    return [train_matrix, test_matrix]
    
def create_feature_vectors(file_list, labels, vectorizer=None):
    """
    This function creates a document-term matrix, given a list of file names
    and a list of labels.

    If a feature vectorizer has been created, it can be passed in as
    `vectorizer` to be used. Otherwise one will be instantiated.

    Parameters
    ----------
    file_list: list of str
        File names to be used for creating matrix.

    labels: list of str
        Correct class labels corresponding with the essays in `file_list`.
        These will be encoded as integers for saving in svm_light format.

    vectorizer: Vectorizer object or NoneType, None by default.
        Object to convert a collection of text documents to a matrix. Must have
        fit, transform, and fit_transform methods implemented. If no vectorizer
        is provided, this function will use sklearn's CountVectorizer. The
        vectorizer that is fit on the training data should be re-used for the
        testing data.

    Returns
    -------
    tuple (length 4)
        -doc-term matrix (numpy array),
        -list of correct labels encoded as ints,
        -list of labels as strings,
        -vectorizer instance

    """
    # convert label strings to integers
    labels_encoded = [CLASS_LABELS.index(label) for label in labels]
    
    if vectorizer is None:
        vectorizer = CountVectorizer(input="filename")  # create a new one
        doc_term_matrix = vectorizer.fit_transform(file_list)
    elif not hasattr(vectorizer, 'vocabulary_'):
        doc_term_matrix = vectorizer.fit_transform(file_list)
    else:
        doc_term_matrix = vectorizer.transform(file_list)
    
    
    print("Created a document-term matrix with %d rows and %d columns."
          % (doc_term_matrix.shape[0], doc_term_matrix.shape[1]))

    return doc_term_matrix.astype(float), labels_encoded, vectorizer

def write_predictions_file_with_probs(predictions,prediction_probs,
                           test_partition_name,
                           predictions_outfile_name,
                           baseline):
    """
    Write a csv that is a copy of the test-set labels file, with an added
    'prediction' field containing the model's predictions.

    Parameters
    ----------
    predictions: list or tuple
        Model's class predictions. Their order should correspond with the order
        in the labels file.

    test_partition_name: str
        Name of test partition directory (e.g. "dev" or "test")

    predictions_outfile_name: str
        Name of predictions output file

    baseline: str
        Name of baseline. This will be used to write the predictions file to
        the appropriate directory.

    Returns
    -------
    None. Writes file only and prints completed status only.

    """
#    labels_file_path = ('{script_dir}/../data/labels/{test}/labels.{test}.csv'
 #                       .format(script_dir=SCRIPT_DIR,
  #                              test=test_partition_name))

    #labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
     #                .format(test=test_partition_name, task="essay" if baseline == "essays" else "speech", 
      #                       script_dir=SCRIPT_DIR))
    
    labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
                     .format(test=test_partition_name, task="fusion", 
                             script_dir=SCRIPT_DIR))
    
    if predictions_outfile_name is None:
        predictions_file_name = strftime("predictions-%Y-%m-%d-%H.%M.%S.csv")
    else:
        predictions_file_name = predictions_outfile_name

    outfile = ('{script_dir}/../predictions/{bl}/{pred_file}'
               .format(script_dir=SCRIPT_DIR,
                       bl=baseline,
                       pred_file=predictions_file_name))

    header = ['prediction','ppst1','ppst2','ppst3','ppst4','ppst5','ppst6','ppst7','ppst8','ppst9','ppst10','ppst11']
    
    with open(outfile,'w',newline='') as csvfile:
        results_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
           
        with open(labels_file_path, encoding='utf-8') as labels_file:
            label_rows = [row for row in csv.reader(labels_file)]
            for j in range(len(header)):
                label_rows[0].append(header[j])
            for i, row in enumerate(label_rows[1:]):
                encoded_prediction = int(predictions[i])
                prediction = CLASS_LABELS[encoded_prediction]
                row.append(prediction)
                for j in range(len(prediction_probs[i])):
                    row.append(prediction_probs[i][j])
            
            results_writer.writerows(label_rows)

    print("Predictions written to", outfile.replace(SCRIPT_DIR, '')[1:],
          "({:d} lines)".format(len(predictions)))
    
def generate_submission_csv(baseline,test_partition_name,predicted,predictions_outfile_name=None):
    
    #labels_file_path = ('{script_dir}/../data/labels/{test}/labels.{test}.csv'
     #                   .format(script_dir=SCRIPT_DIR,
      #                          test=test_partition_name))
    
  #  labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
   #                  .format(test=test_partition_name, task="essay" if baseline == "essays" else "speech", 
    #                         script_dir=SCRIPT_DIR))

    labels_file_path = ("{script_dir}/../data/labels/{test}/{task}.labels.{test}.csv"
                     .format(test=test_partition_name, task="fusion", 
                             script_dir=SCRIPT_DIR))
    
    if predictions_outfile_name is None:
        predictions_file_name = strftime("submission-%Y-%m-%d-%H.%M.%S")
    else:
        predictions_file_name = predictions_outfile_name

    csv_name = predictions_file_name + '.csv'
    zip_name = predictions_file_name + '.zip'
 
    outfile_zip = ('{script_dir}/../submissions/{bl}/{pred_file}'
               .format(script_dir=SCRIPT_DIR,
                       bl=baseline,
                       pred_file=zip_name))
    
    with open(labels_file_path, encoding='utf-8') as labels_file:
            test_takers = [row['test_taker_id'] for row in csv.DictReader(labels_file)]

       
    #with open(outfile, 'w',newline="\n", encoding="utf-8") as csvfile:
     #   fieldnames = ['test_taker_id', 'prediction']
      #  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #writer.writeheader()
        #for i in range(len(test_takers)):
         #   writer.writerow({'test_taker_id':test_takers[i],'prediction': CLASS_LABELS[predicted[i]]})
            
    
    
    with ZipFile(outfile_zip, 'w', ZIP_DEFLATED) as zip_file:
        string_buffer = StringIO()
        fieldnames = ['test_taker_id', 'prediction']
        writer = csv.DictWriter(string_buffer, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(test_takers)):
            writer.writerow({'test_taker_id':test_takers[i],'prediction': CLASS_LABELS[predicted[i]]})
        
        zip_file.writestr(csv_name, string_buffer.getvalue())