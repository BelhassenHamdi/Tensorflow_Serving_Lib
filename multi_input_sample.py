import tensorflow as tf
import numpy as np
import sys, os

tf.app.flags.DEFINE_integer('model_version', 2, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

a = tf.placeholder(tf.float32, shape=(None, 10), name="a")
b = tf.layers.dense(a, 2)
c = tf.nn.sigmoid(b, name="c")

aa = tf.placeholder(tf.float32, shape=(10, 5), name="aa")
d = tf.layers.dense(aa, 3, activation=tf.nn.softmax)

x = tf.nn.softmax(tf.matmul(a,aa), name='x')
e = tf.reduce_mean(d, axis=1, name="e")

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

e_f = np.random.uniform(0,1,(10,5))
a_f = np.random.uniform(0,1,(4,10))
# print("e_f", e_f)
# print("a_f", a_f)

print("a_f", e_f.shape, "a_f" , a_f.shape)

res_a, res_aa, res_x = sess.run([e,c, x], feed_dict={ a : a_f ,aa : e_f })

print("a_f", res_a, "\naa_f" , res_aa, "\nres_x", res_x)




export_path_base = sys.argv[-1]
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(FLAGS.model_version)))
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# Build the signature_def_map.
# classification_inputs = tf.saved_model.utils.build_tensor_info(
#     Input)
# classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
#     prediction_classes)
# classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

# classification_signature = (
#     tf.saved_model.signature_def_utils.build_signature_def(
#         inputs={
#             tf.saved_model.signature_constants.CLASSIFY_INPUTS:
#                 classification_inputs
#         },
#         outputs={
#             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
#                 classification_outputs_classes,
#             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
#                 classification_outputs_scores
#         },
#         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

tensor_info_a = tf.saved_model.utils.build_tensor_info(a)
tensor_info_aa = tf.saved_model.utils.build_tensor_info(aa)
tensor_info_e = tf.saved_model.utils.build_tensor_info(e)
tensor_info_c = tf.saved_model.utils.build_tensor_info(c)
tensor_info_x = tf.saved_model.utils.build_tensor_info(x)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_1': tensor_info_a, 'input_2': tensor_info_aa},

        outputs={'output_1': tensor_info_e, 'output_2': tensor_info_c, 'output_3': tensor_info_x},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'prediction':
            prediction_signature,
    },
    legacy_init_op=legacy_init_op)

builder.save()

print('Done exporting!')





