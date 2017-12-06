import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularier):
    weigths = tf.get_variable("weigths", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularier != None:
        tf.add_to_collection("losses", regularier(weigths))
    return weigths

def inference (input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weight= get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weight) + biases)
    with tf.variable_scope("layer2"):
        weigths = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weigths)+biases
    return layer2


