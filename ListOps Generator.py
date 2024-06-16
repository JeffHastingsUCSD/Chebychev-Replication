import os
import csv
import random
import numpy as np
import tensorflow.compat.v1 as tf

"""
In this part of the project, I generate a synthetic dataset for the ListOps task,
based on the LRA repo, by constructing tree-like equations using a
recursive function, where each node can be an operator
(e.g., MIN, MAX, MED, SUM_MOD) or a value from a predefined set.
The tree's depth and the number of arguments for each operator are
controlled by specified parameters such as the max-depth
(length of each individual sequence) and max-legnth
(the length of each total sequence).
The dataset is divided into training, validation,
and test sets based on the desired number of samples for each set.
The generated equations and their corresponding values are then saved
to TSV files for use in training machine learning models. 
"""

# Define constants and flags
MIN = '[MIN'
MAX = '[MAX'
MED = '[MED'
FIRST = '[FIRST'
LAST = '[LAST'
SUM_MOD = '[SM'
END = ']'

OPERATORS = [MIN, MAX, MED, SUM_MOD]
VALUES = range(10)
VALUE_P = 0.25

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('task', 'basic', 'Name of task to create.')
tf.app.flags.DEFINE_integer('num_train_samples', 56000, 'Number of train samples.')
tf.app.flags.DEFINE_integer('num_valid_samples', 2000, 'Number of validation samples.')
tf.app.flags.DEFINE_integer('num_test_samples', 2000, 'Number of test samples.')
tf.app.flags.DEFINE_integer('max_depth', 7, 'Maximum tree depth of training sequences.')
tf.app.flags.DEFINE_integer('max_args', 7, 'Maximum number of arguments per operator in training sequences.')
tf.app.flags.DEFINE_integer('max_length', 1000, 'Maximum length per sequence in training sequences.')
tf.app.flags.DEFINE_integer('min_length', 250, 'Minimum length per sequence in training sequences.')
tf.app.flags.DEFINE_string('output_dir', 'output_dir', 'Directory to output files.')

def generate_tree(depth, max_depth, max_args):
    """Generate tree-like equations."""
    if depth < max_depth:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value, 1
    else:
        length = 2
        num_values = random.randint(2, max_args)
        values = []
        for _ in range(num_values):
            sub_t, sub_l = generate_tree(depth + 1, max_depth, max_args)
            values.append(sub_t)
            length += sub_l

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t, length

def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'

def to_value(t):
    """Compute the output of equation t."""
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return np.sum(l[1]) % 10
    elif isinstance(l, tuple):
        # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])

def write_to_file(data, fp):
    """Write to file output."""
    print(f'Writing {len(data)} samples to {fp}.tsv')
    os.makedirs(os.path.dirname(fp), exist_ok=True)  # Ensure directory exists
    with open(fp + '.tsv', 'w+', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Source', 'Target'])
        writer.writerows(data)

def main(_):
    print('Start dataset construction')

    data = set()
    num_samples = FLAGS.num_train_samples + FLAGS.num_test_samples + FLAGS.num_valid_samples
    while len(data) < num_samples:
        tree, length = generate_tree(1, FLAGS.max_depth, FLAGS.max_args)
        if length > FLAGS.min_length and length < FLAGS.max_length:
            data.add(tree)
            if len(data) % 1000 == 0:
                print(f'Processed {len(data)}')

    train = []
    for example in data:
        train.append([to_string(example), to_value(example)])

    print('Finished running dataset construction')

    val = train[FLAGS.num_train_samples:]
    test = val[FLAGS.num_valid_samples:]
    val = val[:FLAGS.num_valid_samples]
    train = train[:FLAGS.num_train_samples]

    print(f'Dataset size: {len(train)}/{len(val)}/{len(test)}')

    write_to_file(train, f'{FLAGS.output_dir}/{FLAGS.task}_train')
    write_to_file(val, f'{FLAGS.output_dir}/{FLAGS.task}_val')
    write_to_file(test, f'{FLAGS.output_dir}/{FLAGS.task}_test')
    print('Finished writing all to file')

if __name__ == '__main__':
    tf.app.run(main)
