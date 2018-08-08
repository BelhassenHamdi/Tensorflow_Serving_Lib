import tensorflow as tf
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        content = f.read()
        print(content)
        print("this is : ", graph_def)
        graph_def.ParseFromString(content)

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph



if __name__ == '__main__':
    e_f = np.array([[0.86911871, 0.44577367, 0.34135966, 0.50887036, 0.99913369],
                    [0.77955704, 0.96841831, 0.05152591, 0.02192218, 0.22724724],
                    [0.40600356, 0.61162894, 0.37346274, 0.9562048 , 0.03910843],
                    [0.1274892 , 0.1300438 , 0.68572659, 0.22520398, 0.18749317],
                    [0.04235638, 0.71345292, 0.01422105, 0.28112432, 0.27120046],
                    [0.21647193, 0.80661877, 0.07522653, 0.39278219, 0.30707708],
                    [0.94541457, 0.0865018 , 0.38298925, 0.49254321, 0.18799376],
                    [0.8216806 , 0.90996165, 0.6829972 , 0.24805386, 0.78476169],
                    [0.96388576, 0.76076104, 0.80601641, 0.91795448, 0.39047248],
                    [0.46699564, 0.87474153, 0.73782404, 0.13783029, 0.91972298]])
    a_f = np.array([[0.33577171, 0.76346878, 0.22338613, 0.88184362, 0.71001269, 0.30147327,
            0.81346278, 0.15725204, 0.28549459, 0.05368817],
            [0.12544905, 0.09295932, 0.50912075, 0.27892857, 0.07151685, 0.6004968,
            0.91960399, 0.54878713, 0.41400665, 0.32209193],
            [0.11792933, 0.10189013, 0.16459027, 0.66575211, 0.43335042, 0.70464706,
            0.27999379, 0.80455494, 0.94025668, 0.25369552],
            [0.54301618, 0.75674652, 0.42827687, 0.93053037, 0.3696865,  0.77331241,
            0.15946556, 0.75578234, 0.47238596, 0.67096338]])
    # We use our "load_graph" function
    # graph = load_graph("./freezed_model/1/saved_model.pb")
    export_dir = "./freezed_model/2"
    # We can verify that we can access the list of operations in the graph


    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        graph = tf.get_default_graph()
        # graph.get_operations()
        # for op in graph.get_operations():
        #     print(op.name)
        #     # prefix/Placeholder/inputs_placeholder
        #     # ...
        # # prefix/Accuracy/predictions
        # for n in tf.get_default_graph().as_graph_def().node :
        #     print(n.name) 
        # print(tensors)
        a = graph.get_tensor_by_name("a:0")
        aa = graph.get_tensor_by_name("aa:0")
        e = graph.get_tensor_by_name("e:0")
        c = graph.get_tensor_by_name("c:0")
        x = graph.get_tensor_by_name("x:0")
        res_a, res_aa, res_x = sess.run([e,c, x], feed_dict={ a : a_f ,aa : e_f })
        print("a_f", res_a, "\na_f" , res_aa, "\nres_x", res_x)
        # print(res_a)


