import tensorflow as tf 

def unpooling2d(x,strides=[2,2],name='unpooling2d'):
    with name_scope(name):
        x_shape=x.get_shape().as_list()
        x=tf.expand_dims(x,3)
        x=tf.concat([x]*strides[1],3)
        x=tf.concat([x]*strides[0],2)
        x=tf.reshape(x,[-1,x_shape[1]*strides[0],x_shape[2]*strides[1],x_shape[3]])
    return x