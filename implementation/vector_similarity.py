import math
import tensorflow as tf

def Cosine(vec1, vec2) :
    # result = InnerProduct(vec1,vec2) / (VectorSize(vec1) * VectorSize(vec2))
    return tf.matmul(vec1, tf.transpose(vec2)) / (VectorSize(vec1) * VectorSize(vec2))

def VectorSize(vec) :
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(vec)), tf.constant(1e-08)))
    # return math.sqrt(sum(math.pow(v,2) for v in vec))

# def InnerProduct(vec1, vec2) :
#     return sum(v1*v2 for v1,v2 in zip(vec1,vec2))

def Euclidean(vec1, vec2) :
    #return math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in zip(vec1, vec2)))
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(vec1 - vec2), axis=1, keep_dims=True), tf.constant(1e-08)))

def Theta(vec1, vec2) :
    return tf.acos(Cosine(vec1, vec2)) + tf.constant(10.0)

def Triangle(vec1, vec2) :
    #theta = math.radians(Theta(vec1, vec2))
    theta = tf.constant(2*math.pi/360) * Theta(vec1, vec2)
    return VectorSize(vec1) * VectorSize(vec2) * tf.sin(theta) / tf.constant(2.0)

def Magnitude_Difference(vec1, vec2) :
    return tf.abs(VectorSize(vec1) - VectorSize(vec2))

def Sector(vec1, vec2) :
    ED = Euclidean(vec1, vec2)
    MD = Magnitude_Difference(vec1, vec2)
    theta = Theta(vec1, vec2)
    #return math.pi * math.pow((ED+MD),2) * theta/360
    return tf.constant(math.pi) * tf.square(ED + MD) * theta / tf.constant(360.0)

def TS_SS(vects) :
    vec1, vec2 = vects
    return Triangle(vec1, vec2) * Sector(vec1, vec2)
#
# vec1 = [1,2]
# vec2 = [2,4]
#
# print(Euclidean(vec1,vec2))
# print(Cosine(vec1,vec2))
# print(TS_SS((vec1,vec2)))