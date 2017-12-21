# Copyright (c) 2017, JAMES MASON
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the author nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL JAMES MASON BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import scipy.io
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import tensorflow as tf

def loadVggModel(path, image):
    vgg = scipy.io.loadmat(path)
    def weights(layer, expectedLayerName):
        assert vgg['layers'][0][layer][0][0][0][0] == expectedLayerName
        return vgg['layers'][0][layer][0][0][2][0][0], vgg['layers'][0][layer][0][0][2][0][1]
    def conv2D(prevLayer, layer, layerName):
        w, b = weights(layer, layerName)
        return tf.nn.conv2d(prevLayer, filter=tf.constant(w), strides=[1, 1, 1, 1], padding='SAME') + tf.constant(np.reshape(b, (b.size)))
    graph = {}
    graph['input'] = tf.Variable(np.zeros((1, image.shape[1], image.shape[2], image.shape[3])), dtype = 'float32')
    graph['conv1_1'] = tf.nn.relu(conv2D(graph['input'], 0, 'conv1_1'))
    graph['conv1_2'] = tf.nn.relu(conv2D(graph['conv1_1'], 2, 'conv1_2'))
    graph['maxpool1'] = tf.nn.max_pool(graph['conv1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv2_1'] = tf.nn.relu(conv2D(graph['maxpool1'], 5, 'conv2_1'))
    graph['conv2_2'] = tf.nn.relu(conv2D(graph['conv2_1'], 7, 'conv2_2'))
    graph['maxpool2'] = tf.nn.max_pool(graph['conv2_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv3_1'] = tf.nn.relu(conv2D(graph['maxpool2'], 10, 'conv3_1'))
    graph['conv3_2'] = tf.nn.relu(conv2D(graph['conv3_1'], 12, 'conv3_2'))
    graph['conv3_3'] = tf.nn.relu(conv2D(graph['conv3_2'], 14, 'conv3_3'))
    graph['conv3_4'] = tf.nn.relu(conv2D(graph['conv3_3'], 16, 'conv3_4'))
    graph['maxpool3'] = tf.nn.max_pool(graph['conv3_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv4_1'] = tf.nn.relu(conv2D(graph['maxpool3'], 19, 'conv4_1'))
    graph['conv4_2'] = tf.nn.relu(conv2D(graph['conv4_1'], 21, 'conv4_2'))
    graph['conv4_3'] = tf.nn.relu(conv2D(graph['conv4_2'], 23, 'conv4_3'))
    graph['conv4_4'] = tf.nn.relu(conv2D(graph['conv4_3'], 25, 'conv4_4'))
    graph['maxpool4'] = tf.nn.max_pool(graph['conv4_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv5_1'] = tf.nn.relu(conv2D(graph['maxpool4'], 28, 'conv5_1'))
    graph['conv5_2'] = tf.nn.relu(conv2D(graph['conv5_1'], 30, 'conv5_2'))
    graph['conv5_3'] = tf.nn.relu(conv2D(graph['conv5_2'], 32, 'conv5_3'))
    graph['conv5_4'] = tf.nn.relu(conv2D(graph['conv5_3'], 34, 'conv5_4'))
    graph['maxpool5'] = tf.nn.max_pool(graph['conv5_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return graph

def generateNoiseImage(contentImage, noiseRatio = 0.6): return np.random.uniform(-20, 20, (1, contentImage.shape[1], contentImage.shape[2], contentImage.shape[3])).astype('float32') * noiseRatio + contentImage * (1 - noiseRatio)

def computeContentCost(model, contentLayers, loss = 0):
    for layer, weight in contentLayers: loss += weight * computeContentLayerCost(sess.run(model[layer]), model[layer])
    return loss

def computeContentLayerCost(aC, aG):
    _, nH, nW, nC = aG.get_shape().as_list()
    return (1 / (4 * nH * nW * nC)) * tf.reduce_sum(tf.square(tf.subtract(aC, aG)))

def gramMatrix(l, k, shift = -1.): return tf.matmul(tf.add(l, tf.constant(shift)), tf.transpose(tf.add(k, tf.constant(shift)), perm=[1, 0]))

def computeStyleLayerCost(l, model, layers, loss = 0, blur = True, l2kWeight = 0.25, l2lWeight = 0.75):
    gL = tf.stack(model[layers[l][0]])
    sL = tf.stack(sess.run(model[layers[l][0]]))
    _, lH, lW, lC = gL.get_shape().as_list()
    if l == 0:
        GS = tf.multiply(gramMatrix(tf.transpose(tf.reshape(sL, [lH * lW, lC])), tf.transpose(tf.reshape(sL, [lH * lW, lC]))), l2lWeight + l2kWeight)
        GG = tf.multiply(gramMatrix(tf.transpose(tf.reshape(gL, [lH * lW, lC])), tf.transpose(tf.reshape(gL, [lH * lW, lC]))), l2lWeight + l2kWeight)
    else:
        gK = tf.stack(model[layers[l - 1][0]])
        sK = tf.stack(sess.run(model[layers[l - 1][0]]))
        sK = tf.transpose(tf.image.resize_images(tf.transpose(sK, perm = [0, 3, 2, 1]), [lC, lW]), perm = [0, 3, 2, 1]) if tf.transpose(sL, perm = [0, 3, 2, 1]).shape != tf.transpose(sK, perm = [0, 3, 2, 1]).shape else sK
        gK = tf.transpose(tf.image.resize_images(tf.transpose(gK, perm = [0, 3, 2, 1]), [lC, lW]), perm = [0, 3, 2, 1]) if tf.transpose(gL, perm = [0, 3, 2, 1]).shape != tf.transpose(gK, perm = [0, 3, 2, 1]).shape else gK
        sK = tf.image.resize_images(sK, [lH, lW]) if sL.shape != sK.shape else sK
        gK = tf.image.resize_images(gK, [lH, lW]) if gL.shape != gK.shape else gK
        if blur == True:
            sK = tf.stack(gaussian_filter(sK.eval(), sigma = 0.5))
            gK = tf.stack(gaussian_filter(gK.eval(), sigma = 0.5))
        GS = tf.add(tf.multiply(gramMatrix(tf.transpose(tf.reshape(sL, [lH * lW, lC])), tf.transpose(tf.reshape(sK, [lH * lW, lC]))), l2kWeight), tf.multiply(gramMatrix(tf.transpose(tf.reshape(sL, [lH * lW, lC])), tf.transpose(tf.reshape(sL, [lH * lW, lC]))), l2lWeight))
        GG = tf.add(tf.multiply(gramMatrix(tf.transpose(tf.reshape(gL, [lH * lW, lC])), tf.transpose(tf.reshape(gK, [lH * lW, lC]))), l2kWeight), tf.multiply(gramMatrix(tf.transpose(tf.reshape(gL, [lH * lW, lC])), tf.transpose(tf.reshape(gL, [lH * lW, lC]))), l2lWeight))
    loss += (1 / (4 * (lC ** 2) * ((lH * lW) ** 2))) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    return loss

def computeStyleCost(model, styleLayers, loss = 0):
    for l in range(0, len(styleLayers)): loss += styleLayers[l][1] * computeStyleLayerCost(l, model, styleLayers)
    return loss

def getWeightedContentLayers(layers, output = []):
    for i in range(len(layers)): output += [(layers[i], 2**(i + 1))]
    return output

def getWeightedStyleLayers(layers, output = []):
    for i in range(len(layers)): output += [(layers[i], 2**(len(layers) - (i + 1)))]
    return output

def loadImage(image, width = None, height = None):
    image = np.reshape(image, ((1,) + image.shape))
    image = image - np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    if height != None and width != None: return tf.cast(tf.image.resize_images(image, [height, width]), tf.float32)
    elif width != None:
        height = tf.cast((tf.cast(image.shape[1], tf.int32) / tf.cast(image.shape[2], tf.int32)) * width, tf.int32).eval()
        return tf.cast(tf.image.resize_images(image, [height, width]), tf.float32)
    else: return tf.cast(tf.image.resize_images(image, [tf.cast(image.shape[1], tf.int32), tf.cast(image.shape[2], tf.int32)]), tf.float32)

def saveImage(path, image):
    image = image + np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    scipy.misc.imsave(path, np.clip(image[0], 0, 255).astype('uint8'))

#Setup
tf.reset_default_graph()
sess = tf.InteractiveSession()
content = loadImage(scipy.misc.imread(sys.argv[1]), int(sys.argv[3]))
style = loadImage(scipy.misc.imread(sys.argv[2]), tf.cast(content.shape[2], tf.int32), tf.cast(content.shape[1], tf.int32))
output = generateNoiseImage(content)

#Save base images for reference.
saveImage('output/setup/content.jpg', content.eval())
saveImage('output/setup/style.jpg', style.eval())

model = loadVggModel('imagenet-vgg-verydeep-19.mat', content)
modelLayers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
contentLayers = getWeightedContentLayers(modelLayers)
styleLayers = getWeightedStyleLayers(modelLayers)

#Content Cost
sess.run(model['input'].assign(content))
jContent = computeContentCost(model, contentLayers)

#Style Cost
sess.run(model['input'].assign(style))
jStyle = computeStyleCost(model, styleLayers)

#Total Cost and Optimizer
j = 10 * jContent + 1 * jStyle # (#) = alpha, (#) = beta
optimizer = tf.train.AdamOptimizer(2.0).minimize(j)

def neuralStyleTransfer(sess, input, iterations = int(sys.argv[4])):
    sess.run(tf.global_variables_initializer())
    output = sess.run(model['input'].assign(input))
    for i in range(iterations):
        sess.run(optimizer)
        output = sess.run(model['input'])
        t, c, s = sess.run([j, jContent, jStyle])
        print('Iteration: ' + str(i))
        print('    Style Cost: ' + str(s))
        print('    Content Cost: ' + str(c))
        print('    Total Cost: ' + str(t))
        if i % 100 == 0: saveImage('output/iterations/' + str(i) + '.jpg', output)
    saveImage('output/output.jpg', output)

neuralStyleTransfer(sess,output)
