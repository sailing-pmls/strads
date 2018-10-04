Input data file format:

Each line represents 1 document, and consists of the following in order:

     0 or more integer labels [0 to num_label) in a one-hot encoding.

     num_xtra real values as constant extra predictors in the classifier matrix.

     num_target real values as regression targets (each regression target is a separate "task" to predict a scalar).

     any number of word:count pairs where word is an arbitrary string containing no spaces or ':'s and count is a positive integer.






