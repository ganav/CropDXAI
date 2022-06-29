
import Network, math
import  Utils
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.layers import TimeDistributed
from tqdm import tqdm
import numpy as np
import argparse,csv, time,cv2
from tensorflow.keras.optimizers import Adam,SGD
from keras_apps_ import efficientNet, resnet_common, densenet, inception_resnet_v2,inception_v4

np.random.seed(10)

out_size = 12
image_shape = (200,200,3)
image_shape2 = (out_size,out_size,3)

def XDGan_network(discriminator, generator, optimizer):
    discriminator.trainable = False
    inp=Input(shape = image_shape)

    x,hp0,hp1,hp2,hp3,hp4,hp5 = generator(inp)

    xdgan_output = discriminator([hp0,hp1,hp2,hp3,hp4,hp5])

    xdgan = Model(inputs=inp, outputs=xdgan_output)
    xdgan.compile(loss=["categorical_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    return xdgan


def train(epochs, batch_size, output_dir, model_save_dir, ext,n_class):

    generator = Network.generator(image_shape,n_class)
    discriminator = Network.discriminator(image_shape2,n_class+1)

    optimizer = Adam(learning_rate=0.00001, decay=0.000001)ss
    discriminator.compile(loss="categorical_crossentropy", optimizer=optimizer)
    
    xdgan = XDGan_network(discriminator,generator, optimizer)
    xdgan2 = XDGan_network2(generator, optimizer)
    
    data_csv = []
    img0 = np.zeros((out_size,out_size), dtype=np.uint8)
    img0=img0[np.newaxis,...]

    x_train, y_train,labels2 = Utils.load_training_data('train', ext,n_class,out_size)
    x_test, y_test,labels = Utils.load_training_data('test', ext,n_class,out_size)
            
    batch_count = int(x_train.shape[0] / batch_size)
    batch_count2 = int(x_test.shape[0] / batch_size)

    indx3 = tf.one_hot(n_class, n_class+1)
    indx3 = indx3[np.newaxis,...]
    indx3 = np.array(indx3)


    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        loss = 0
        loss1 = 0
        loss2 = 0

        for num in tqdm(range(batch_count)):

            #one hot
            indx = tf.one_hot(labels2[num], n_class)
            indx = indx[np.newaxis,...]
            indx = np.array(indx)

            indx2 = tf.one_hot(labels2[num], n_class+1)
            indx2 = indx2[np.newaxis,...]
            indx2 = np.array(indx2)

            prob = generator.train_on_batch(x_train[num],indx)
            loss1 = loss1 + prob[0]

            prob, hp0,hp1,hp2,hp3,hp4,hp5 = generator.predict(x_train[num])
            discriminator.trainable = True
                               
            if  labels2[num] ==0:
                loss_1 = discriminator.train_on_batch([y_train[num],img0,img0,img0,
                    img0,img0], indx2)
            elif  labels2[num] ==1:
                loss_1 = discriminator.train_on_batch([img0,y_train[num],img0,img0,
                    img0,img0], indx2)
            elif  labels2[num] ==2:
                loss_1 = discriminator.train_on_batch([img0,img0,y_train[num],img0,
                    img0,img0], indx2)
            elif  labels2[num] ==3:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,y_train[num],
                    img0,img0], indx2)
            elif  labels2[num] ==4:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,
                    img0,y_train[num],img0], indx2)
            elif  labels2[num] ==5:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,
                    img0,img0,y_train[num]], indx2)
           
                        
            loss_2 = discriminator.train_on_batch([hp0,hp1,hp2,hp3,hp4,hp5], indx3)

            loss1 = loss1 + 0.5 * np.add(loss_1, loss_2)

            discriminator.trainable = False
            xdgan_loss = xdgan.train_on_batch(x_train[num],indx2)

        generator.save(model_save_dir + 'gen_model%d.h5' % e)

if __name__== "__main__":
                 
    batch_size=8
    epochs=50
    model_save_dir='./model/'
    output_dir='./output/'
    ext='.png'
    n_class = 6
    train(epochs, batch_size, output_dir, model_save_dir,ext,n_class)
