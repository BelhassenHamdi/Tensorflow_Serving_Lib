import tensorflow as tf
import numpy as np
import sys, os
from sklearn.metrics import accuracy_score


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS


epochs = 10000


Labels = tf.placeholder(dtype=tf.float32, shape=(None, 10))
# Input = tf.placeholder(dtype=tf.float32,shape=(None, 784), name="input")

Input = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
tf_example = tf.parse_example(Input, feature_configs)
x = tf.identity(tf_example['x'], name='x')


reshape = tf.reshape(x, (-1, 28, 28))
expand = tf.expand_dims(reshape, axis=3)

conv1 = tf.layers.conv2d(expand, 32, 3, padding='same')
maxpool1 = tf.layers.max_pooling2d(conv1, 3, 2)
conv2 = tf.layers.conv2d(maxpool1, 128, 3)

flat = tf.layers.Flatten()(conv2)
logits = tf.layers.dense(flat,10)

output = tf.nn.softmax(logits, name="output")


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Labels))
optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)


values, indices = tf.nn.top_k(output, 3)
table = tf.contrib.lookup.index_to_string_table_from_tensor(
    tf.constant([str(i) for i in range(10)]))
prediction_classes = table.lookup(tf.to_int64(indices))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        X, Y = mnist.train.next_batch(64)
        sess.run(optimizer, feed_dict={x: X, Labels :Y})
        if i%1000==0 :
            X_val, Y_val = mnist.validation.next_batch(64)
            Loss, Out = sess.run([loss, output], feed_dict={x: X_val, Labels :Y_val})
            print("iteration num ", i," Accuracy : ",accuracy_score(y_pred=np.argmax(Out, axis=1),y_true=np.argmax(Y_val, axis=1))," loss : ", Loss)

    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(
        Input)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
        prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classification_inputs
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classification_outputs_classes,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    classification_outputs_scores
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(output)

    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'images': tensor_info_x},
    #         outputs={'scores': tensor_info_y},
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')




