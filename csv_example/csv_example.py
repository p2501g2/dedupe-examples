#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use dedupe with a comma separated values
(CSV) file. All operations are performed in memory, so will run very
quickly on datasets up to ~10,000 rows.

We start with a CSV file containing our messy data. In this example,
it is listings of early childhood education centers in Chicago
compiled from several different sources.

The output will be a CSV with our clustered results.

For larger datasets, see our [mysql_example](mysql_example.html)
"""
from future.builtins import next

import os
import csv
import re
import logging
import optparse
import time
import dedupe
from unidecode import unidecode

# ## Logging

# Dedupe uses Python logging to show or suppress verbose output. This
# code block lets you change the level of loggin on the command
# line. You don't need it if you don't want that. To enable verbose
# logging, run `python examples/csv_example/csv_example.py -v`
optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING
if opts.verbose:
    if opts.verbose == 1:
        log_level = logging.INFO
    elif opts.verbose >= 2:
        log_level = logging.DEBUG
logging.getLogger().setLevel(log_level)

# ## Setup

input_file = 'csv_example_messy_input.csv'
output_file = 'csv_example_output.csv'
settings_file = 'csv_example_learned_settings'
training_file = 'csv_example_training.json'

def preProcess(column):
    """
    Do a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines can be ignored.
    """
    try : # python 2/3 string differences
        column = column.decode('utf8')
    except AttributeError:
        pass
    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    # If data is missing, indicate that by setting the value to `None`
    if not column:
        column = None
    return column

def readData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records,
    where the key is a unique record ID and each value is dict
    """

    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = int(row['Id'])
            data_d[row_id] = dict(clean_row)

    return data_d

start_time = time.time()

print('importing data ...')
data_d = readData(input_file)

data_iot = time.time() - start_time
print 'creating data dictionary -- {0} seconds'.format(data_iot)

# If a settings file already exists, we'll just load that and skip training
if os.path.exists(settings_file):
    print('reading from', settings_file)

    with open(settings_file, 'rb') as f:
        settings_time_start = time.time()
        deduper = dedupe.StaticDedupe(f)
        settings_time = time.time() - settings_time_start
        print 'creating deduper from settings file -- {0} seconds'.format(settings_time)
else:
    # ## Training

    # Define the fields dedupe will pay attention to
    fields = [
        {'field' : 'Site name', 'type': 'String'},
        {'field' : 'Address', 'type': 'String'},
        {'field' : 'Zip', 'type': 'Exact', 'has missing' : True},
        {'field' : 'Phone', 'type': 'String', 'has missing' : True},
        ]
    # Timing variables
    training_setup_start = time.time()

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields)

    # To train dedupe, we feed it a sample of records.
    deduper.sample(data_d, 15000)

    # Timing variables
    training_setup_time = time.time() - training_setup_start
    print 'training setup (dedupe instantiate and sample) -- {0} seconds'.format(training_setup_time)
    settings_time = training_setup_time


    # If we have training data saved from a previous run of dedupe,
    # look for it and load it in.
    # __Note:__ if you want to train from scratch, delete the training_file
    if os.path.exists(training_file):
        print('reading labeled examples from ', training_file)

        # Timing variables
        read_training_time_start = time.time()

        with open(training_file, 'rb') as f:
            deduper.readTraining(f)

            # Timing variables
            settings_time += time.time() - read_training_time_start
            print 'reading from training file -- {0} seconds'.format(time.time() - read_training_time_start)


    # Timing variables
    active_learning_start = time.time()
    # ## Active learning
    # Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.
    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    print('starting active labeling...')
    dedupe.consoleLabel(deduper)

    # Timing variables
    labeling_time = time.time() - active_learning_start
    train_start_time = time.time()

    # Using the examples we just labeled, train the deduper and learn
    # blocking predicates
    deduper.train()

    # When finished, save our training to disk
    with open(training_file, 'w') as tf:
        deduper.writeTraining(tf)

    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    with open(settings_file, 'wb') as sf:
        deduper.writeSettings(sf)

    # Timing variables
    training_time = time.time() - train_start_time
    print 'training time -- {0} seconds'.format(training_time)
    settings_time += training_time

# Timing variables
clustering_time_start = time.time()

# Find the threshold that will maximize a weighted average of our
# precision and recall.  When we set the recall weight to 2, we are
# saying we care twice as much about recall as we do precision.

# If we had more data, we would not pass in all the blocked data into
# this function but a representative sample.

threshold = deduper.threshold(data_d, recall_weight=1)

# ## Clustering

# `match` will return sets of record IDs that dedupe
# believes are all referring to the same entity.

print('clustering...')
clustered_dupes = deduper.match(data_d, threshold)

# Timing variables
clustering_time = time.time() - clustering_time_start
print 'clustering/thresholding -- {0} seconds'.format(clustering_time)
print('# duplicate sets', len(clustered_dupes))

# ## Writing Results

# Write our original data back out to a CSV with a new column called
# 'Cluster ID' which indicates which records refer to each other.
writing_results_start_time = time.time()

cluster_membership = {}
cluster_id = 0
for (cluster_id, cluster) in enumerate(clustered_dupes):
    id_set, scores = cluster
    cluster_d = [data_d[c] for c in id_set]
    canonical_rep = dedupe.canonicalize(cluster_d)
    for record_id, score in zip(id_set, scores):
        cluster_membership[record_id] = {
            "cluster id" : cluster_id,
            "canonical representation" : canonical_rep,
            "confidence": score
        }

singleton_id = cluster_id + 1

with open(output_file, 'w') as f_output, open(input_file) as f_input:
    writer = csv.writer(f_output)
    reader = csv.reader(f_input)

    heading_row = next(reader)
    heading_row.insert(0, 'confidence_score')
    heading_row.insert(0, 'Cluster ID')
    canonical_keys = canonical_rep.keys()
    for key in canonical_keys:
        heading_row.append('canonical_' + key)

    writer.writerow(heading_row)

    for row in reader:
        row_id = int(row[0])
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]["cluster id"]
            canonical_rep = cluster_membership[row_id]["canonical representation"]
            row.insert(0, cluster_membership[row_id]['confidence'])
            row.insert(0, cluster_id)
            for key in canonical_keys:
                row.append(canonical_rep[key].encode('utf8'))
        else:
            row.insert(0, None)
            row.insert(0, singleton_id)
            singleton_id += 1
            for key in canonical_keys:
                row.append(None)
        writer.writerow(row)

# Timing variables
writing_results_time = time.time() - writing_results_start_time

total_time = time.time() - start_time
print 'Total process time w/ learning -- {0} seconds'.format(total_time)

algorithmic_total = writing_results_time + clustering_time + settings_time + data_iot

print 'algorithm time (exclusing active labeling) -- {0} seconds'.format(algorithmic_total)

print 'sanity check timings, {0} seconds calculated for learning'.format(total_time - algorithmic_total)
print 'and {0} seconds recorded for learning'.format(labeling_time)
