import os
import imageio
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class Attenuation_layer(layers.Layer):
    def __init__(self):
        super(Attenuation_layer, self).__init__(self)

    def build(self,input_shape):
        self.eta_r = self.add_weight(
            name='eta_r',
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(mean=0.33, stddev=0.01),
            trainable=True)
        self.eta_b = self.add_weight(
            name='eta_b',
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(mean=0.30,stddev=0.01),
            trainable=True)
        self.eta_g = self.add_weight(
            name='eta_g',
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(mean=0.30,stddev=0.01),
            trainable=True)

    @tf.function
    def call(self,inputs):
        depth=tf.expand_dims(inputs,axis=len(inputs.get_shape()))
        eta=tf.stack([self.eta_r,self.eta_g,self.eta_b],axis=2)
        # print(eta)
        eta_d=tf.exp(tf.multiply(-1.0,tf.multiply(depth,eta)))
        return eta_d,self.eta_r,self.eta_g,self.eta_b
class Squeeze_layer(layers.Layer):
    def __init__(self):
        super(Squeeze_layer, self).__init__(self)

    def build(self,input_shape):
        pass
    @tf.function
    def call(self,inputs):
        result=tf.squeeze(inputs,axis=3)
        return result

class Backscatter(layers.Layer):
    def __init__(self):
        super(Backscatter, self).__init__(self)

    def build(self,input_shape):
        self.A = self.add_weight(
            name='A',
            shape=[1,1,1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(mean=1.5, stddev=0.01),
            trainable=True)
    @tf.function
    def call(self,inputs):
        A2=self.A+np.random.uniform(low=-0.1, high=0.1,size=[1,1,1])
        output=tf.multiply(inputs,A2)
        return output
class Preprocess(layers.Layer):
    def __init__(self):
        super(Preprocess, self).__init__(self)

    def build(self,input_shape):
        self.r = self.add_weight(
            name='r',
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(mean=0.8, stddev=0.01),
            trainable=True)
        self.g = self.add_weight(
            name='g',
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(mean=1.1,stddev=0.01),
            trainable=True)
        self.b = self.add_weight(
            name='b',
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(mean=1.1,stddev=0.01),
            trainable=True)
    @tf.function
    def call(self,inputs):
        factor=tf.stack([self.r,self.g,self.b],axis=2)
        outputs=tf.multiply(inputs,factor)
        return outputs

class WGAN(object):
    def __init__(self,
                 # train parameter
                 batch_size=64,learning_rate=0.0002, beta1=0.5,
                 # image size
                 is_crop=True,input_height=640, input_width=480, input_water_height=1360,
                 input_water_width=1024, output_height=256, output_width=256,c_dim=3,max_depth=3.0,
                 #layer parameter
                 z_dim=100, gf_dim=64, df_dim=64,gfc_dim=1024, dfc_dim=1024,
                 #sample parameters
                 num_samples=4000,sample_output_width=640,sample_output_height=480,
                 #save parameter
                 save_epoch = 100, checkpoint_dir=None,
                 results_dir=None,sample_dir=None,back_dir=None,ckpt_path=None,ckpt_path2=None,
                 # dataset$pattern
                 water_dataset_name='default',air_dataset_name='default',test_dataset_name='default',
                 test_depth_name='default',depth_dataset_name='default',input_fname_pattern='*.png'):
        # train parameter
        self.batch_size = batch_size
        self.gen_size=batch_size
        self.learning_rate=learning_rate
        self.beta1=beta1
        # image size
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.input_water_height = input_water_height
        self.input_water_width = input_water_width
        self.c_dim = c_dim
        self.max_depth = max_depth
        #layer parameter
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        # dataset
        self.test_dataset_name=test_dataset_name
        self.test_depth_name=test_depth_name
        self.water_dataset_name = water_dataset_name
        self.air_dataset_name = air_dataset_name
        self.depth_datset_name = depth_dataset_name
        self.input_fname_pattern = input_fname_pattern
        #sample parameters
        self.num_samples = num_samples
        self.sample_output_height=sample_output_height
        self.sample_output_width=sample_output_width
        #save parameter
        self.save_epoch = save_epoch
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.sample_dir=sample_dir
        self.back_dir=back_dir
        self.ckpt_path=ckpt_path
        self.ckpt_path2=ckpt_path2
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.build_model()


    def build_model(self):
        # self.G=self.make_discriminator_model(depth_inputs,air_inputs,z)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=self.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=self.beta1)
        self.backscatter=Backscatter()
        self.attenuation_layer=Attenuation_layer()
        self.preprocess= Preprocess()
        self.squeeze_layer=Squeeze_layer()
        self.discriminator=self.make_discriminator_model()
        self.generator=self.make_generator_model()
        self.generator_sample=self.make_generator_sample_model()
    def make_generator_model(self):
        depth = Input(shape=(self.output_height,self.output_width), dtype='float32')
        assert (depth.get_shape()[0]==None)&\
               (depth.get_shape()[1]==self.output_height)\
               &(depth.get_shape()[2]==self.output_width)

        eta_d,eta_r,eta_g,eta_b=self.attenuation_layer(depth)
        # tf.print(eta_r)

        # print(eta_r)
        # print(tf.keras.backend.eval(eta_r))
        assert  (eta_d.get_shape()[0]==None)&(eta_d.get_shape()[1]==self.output_height)&\
        (eta_d.get_shape()[2]==self.output_width)&(eta_d.get_shape()[3]==self.c_dim)

        image= Input(shape=(self.output_height,self.output_width,3), dtype='float32')


        assert  (image.get_shape()[0]==None)&(image.get_shape()[1]==self.output_height)&\
            (image.get_shape()[2]==self.output_width)&(image.get_shape()[3]==self.c_dim)
        #preprocess
        image2=self.preprocess(image+1)-1
        image3=layers.ReLU(max_value=2)(image2+1)-1
        #attenuation
        h0=layers.multiply([image3+1,eta_d])-1
        #back
        backscatter_factor=1-eta_d
        back=self.backscatter(backscatter_factor)
        h1= layers.add([back,h0])
        assert  (h1.get_shape()[0]==None)&(h1.get_shape()[1]==self.output_height)&\
            (h1.get_shape()[2]==self.output_width)&(h1.get_shape()[3]==self.c_dim)


        h_out= h1
        h_out=tf.keras.activations.tanh(h_out)
        gen_model = Model([depth,image], [image3,h_out,h0,h1])
        return gen_model

    def make_generator_sample_model(self):
        depth = Input(shape=(self.sample_output_height, self.sample_output_width), dtype='float32')
        assert (depth.get_shape()[0] == None) & \
               (depth.get_shape()[1] == self.sample_output_height) \
               & (depth.get_shape()[2] == self.sample_output_width)
        depth_small = Input(shape=(self.output_height, self.output_width), dtype='float32')

        assert (depth_small.get_shape()[0] == None) & \
               (depth_small.get_shape()[1] == self.output_height) \
               & (depth_small.get_shape()[2] == self.output_width)

        eta_d, eta_r, eta_g, eta_b = self.attenuation_layer(depth)

        assert (eta_d.get_shape()[0] == None) & (eta_d.get_shape()[1] == self.sample_output_height) & \
               (eta_d.get_shape()[2] == self.sample_output_width) & (eta_d.get_shape()[3] == self.c_dim)

        image = Input(shape=(self.sample_output_height, self.sample_output_width, 3), dtype='float32')
        assert (image.get_shape()[0] == None) & (image.get_shape()[1] == self.sample_output_height) & \
               (image.get_shape()[2] == self.sample_output_width) & (image.get_shape()[3] == self.c_dim)
        #preprocess
        image2=self.preprocess(image+1)-1
        image3=layers.ReLU(max_value=2)(image2+1)-1
        h0=layers.multiply([image3+1,eta_d])-1
        assert (h0.get_shape()[0] == None) & (h0.get_shape()[1] == self.sample_output_height) & \
               (h0.get_shape()[2] == self.sample_output_width) & (h0.get_shape()[3] == self.c_dim)
        #back
        backscatter_factor=1-eta_d
        back=self.backscatter(backscatter_factor)
        h1= layers.add([back,h0])


        h_out=h1
        h_out=tf.keras.activations.tanh(h_out)
        gen_model = Model([depth, depth_small, image], [image3, h_out])
        return gen_model





    def make_discriminator_model(self):
        k_h = 5
        k_w = 5
        d_h = 2
        d_w = 2
        stddev = 0.02
        model = tf.keras.Sequential()
        # 1 conv2d
        model.add(layers.Conv2D(self.df_dim,
                                (k_h, k_w),
                                name='D_conv1',
                                padding='same',
                                strides=[d_h, d_w],
                                input_shape=[self.output_height,self.output_width,3],
                                kernel_initializer=tf.initializers.TruncatedNormal(stddev=stddev),
                                bias_initializer=tf.initializers.Constant(0.0)))
        model.add(layers.LeakyReLU(alpha=0.2))
        # 2 conv2d
        model.add(layers.Conv2D(self.df_dim * 2,
                                (k_h, k_w),
                                name='D_conv2',
                                padding='same',
                                strides=[ d_h, d_w],
                                input_shape=[self.input_height, self.input_width, 3],
                                kernel_initializer=tf.initializers.TruncatedNormal(stddev=stddev),
                                bias_initializer=tf.initializers.Constant(0.0)))

        model.add(layers.BatchNormalization(epsilon=1e-5,momentum=0.9))
        model.add(layers.LeakyReLU(alpha=0.2))
        # 3 conv2d
        model.add(layers.Conv2D(self.df_dim * 4,
                                (k_h, k_w),
                                name='D_conv3',
                                padding='same', strides=[d_h, d_w],
                                input_shape=[self.input_height, self.input_width, 3],
                                kernel_initializer=tf.initializers.TruncatedNormal(stddev=stddev),
                                bias_initializer=tf.initializers.Constant(0.0)))

        model.add(layers.BatchNormalization(epsilon=1e-5,momentum=0.9))
        model.add(layers.LeakyReLU(alpha=0.2))
        # 4 conv2d
        model.add(layers.Conv2D(self.df_dim * 8,
                                (k_h, k_w),
                                name='D_conv4',
                                padding='same', strides=[d_h, d_w],
                                input_shape=[self.input_height, self.input_width, 3],
                                kernel_initializer=tf.initializers.TruncatedNormal(stddev=stddev),
                                bias_initializer=tf.initializers.Constant(0.0)))
        model.add(layers.BatchNormalization(epsilon=1e-5,momentum=0.9))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(1, use_bias=True,
                               activation=tf.nn.sigmoid,
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                               bias_initializer=tf.constant_initializer(0.0),
                               ))  # (tf.reshape(h3, [self.batch_size, -1])))
        return model

    def train(self,config):
        ## train data
        water_data = sorted(glob(os.path.join(
            "../data", config.water_dataset,  "*.jpg")))
        air_data = sorted(glob(os.path.join(
            "../data", config.air_dataset, self.input_fname_pattern)))
        depth_data = sorted(glob(os.path.join(
            "../data", config.depth_dataset, "*.mat")))

        water_batch_idxs = min([len(air_data), len(water_data), config.train_size]) // config.batch_size
        air_batch_idxs = water_batch_idxs


        ##shuffle
        randombatch = np.arange(water_batch_idxs * config.batch_size)
        np.random.shuffle(randombatch)
        ##end shuffle
        ## test data
        file_board=os.path.join("logs"+datetime.now().strftime("%Y%m%d%H%M%S"))
        summary_writer = tf.summary.create_file_writer(file_board)
        # file_board2=os.path.join("logs"+datetime.now().strftime("%Y%m%d%H%M%S"))
        # summary_writer2 = tf.summary.create_file_writer(file_board2)
        # tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace，可以记录图结构和profile信息

        # 开始模型训练
        # 进行训练
        for epoch in range(config.epoch):
            start = time.time()
            # Load water images

            for idx in range(0, (water_batch_idxs * config.batch_size), config.batch_size):
                water_batch_files = []
                air_batch_files = []
                depth_batch_files = []
                ## start make patch
                for id in range(0, config.batch_size):
                    water_batch_files = np.append(water_batch_files, water_data[randombatch[idx + id]])
                    air_batch_files = np.append(air_batch_files, air_data[randombatch[idx + id]])
                    depth_batch_files = np.append(depth_batch_files, depth_data[randombatch[idx + id]])
                # print(depth_batch_files)
                if self.is_crop:
                    air_batch = [self.read_img(air_batch_file) for air_batch_file in air_batch_files]
                    water_batch = [self.read_img(water_batch_file) for water_batch_file in water_batch_files]
                    depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
                else:
                    air_batch = [imageio.imread(air_batch_file) for air_batch_file in air_batch_files]
                    water_batch = [imageio.imread(water_batch_file) for water_batch_file in water_batch_files]
                    depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
                air_batch_images = np.array(air_batch).astype(np.float32)
                water_batch_images = np.array(water_batch).astype(np.float32)
                depth_batch_images=np.array(depth_batch)
                # depth_batch_images = np.expand_dims(depth_batch, axis=3)
                ## end make patch

                eta_r,eta_g,eta_b,real_output,fake_output,gen_loss,real_loss,fake_loss,disc_loss,generated_images,atten_images=\
                    self.train_step(depth_batch_images=depth_batch_images,
                                air_batch_images=air_batch_images,
                                water_batch_images=water_batch_images)



                if((idx//config.batch_size)%5==0):
                    with summary_writer.as_default():  # 希望使用的记录器
                        tf.summary.scalar("gen_loss", gen_loss, step=idx//config.batch_size+epoch*water_batch_idxs)
                        tf.summary.scalar("disc_loss", disc_loss, step=idx//config.batch_size+epoch*water_batch_idxs)  # 还可以添加其他自定义的变
                        tf.summary.scalar("eta_r", eta_r,step=idx//config.batch_size+epoch*water_batch_idxs)
                        tf.summary.scalar("eta_g", eta_g, step=idx // config.batch_size + epoch * water_batch_idxs)
                        tf.summary.scalar("eta_b", eta_b, step=idx // config.batch_size + epoch * water_batch_idxs)
                        tf.summary.scalar("A",tf.squeeze(tf.convert_to_tensor(self.backscatter.A)),
                                          step=idx // config.batch_size + epoch * water_batch_idxs)
                        tf.summary.scalar("r",tf.squeeze(tf.convert_to_tensor(self.preprocess.r)),
                                          step=idx // config.batch_size + epoch * water_batch_idxs)
                        tf.summary.scalar("g",tf.squeeze(tf.convert_to_tensor(self.preprocess.g)),
                                          step=idx // config.batch_size + epoch * water_batch_idxs)
                        tf.summary.scalar("b",tf.squeeze(tf.convert_to_tensor(self.preprocess.b)),
                                          step=idx // config.batch_size + epoch * water_batch_idxs)
                        tf.summary.scalar("D_real_loss",real_loss,step=idx // config.batch_size + epoch * water_batch_idxs)
                        tf.summary.scalar("D_fake_loss",fake_loss,step=idx // config.batch_size + epoch * water_batch_idxs)
                        gen_images = np.reshape(generated_images[0:1], (-1, self.output_height, self.output_width, 3))
                        tf.summary.image("gen_images", gen_images, max_outputs=1, step=0)
                        att_images = np.reshape(atten_images[0:1], (-1, self.output_height, self.output_width, 3))
                        tf.summary.image("att_images", att_images, max_outputs=1, step=0)
                        tf.summary.histogram(
                            "How fake works(real)", real_output, step=idx//config.batch_size+epoch*water_batch_idxs, buckets=None, description=None
                        )
                        tf.summary.histogram(
                            "How fake works", fake_output, step=idx//config.batch_size+epoch*water_batch_idxs, buckets=None, description=None
                        )
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        attenuation = self.attenuation_layer
        backscatter = self.backscatter
        checkpoint = tf.train.Checkpoint(attenuation=attenuation, backscatter=backscatter)  # 实例化Checkpoint，设置保存对象为model
        checkpoint.save(file_prefix='./save/model.ckpt')  # 保存模型参数到文件

    def test(self, config):
        checkpoint = tf.train.Checkpoint(attenuation=self.attenuation_layer, backscatter=self.backscatter)
        checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_path2))  # 从文件恢复模型参数

        test_air_data = sorted(glob(os.path.join(
            "../data", config.test_dataset, '*.png')))
        test_depth_data = sorted(glob(os.path.join(
            "../data", config.test_depth, '*.mat')))
        test_batch_size = min([len(test_depth_data), len(test_air_data), self.num_samples])



        for i in range(test_batch_size//self.gen_size):
            test_air_batch = [self.read_sample_img(air_batch_file) for air_batch_file in
                              test_air_data[i*self.gen_size:(i+1)*self.gen_size]]
            test_depth_small_batch = [self.read_depth(depth_batch_file) for depth_batch_file in
                                      test_depth_data[i*self.gen_size:(i+1)*self.gen_size]]
            test_depth_batch = [self.read_depth_sample(depth_batch_file) for depth_batch_file in
                                test_depth_data[i*self.gen_size:(i+1)*self.gen_size]]


            test_air_batch = np.array(test_air_batch).astype(np.float32)
            test_depth_small_batch = np.array(test_depth_small_batch)
            test_depth_batch = np.array(test_depth_batch)
            self.generate_and_save_images(self.generator_sample,
                                          i,
                                          depth=test_depth_batch,
                                          depth_small=test_depth_small_batch,
                                          image=test_air_batch,
                                          )



    def generate_and_save_images(self, model, epoch, depth, depth_small, image):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        assert depth.shape == (self.gen_size, self.sample_output_height, self.sample_output_width)
        assert image.shape == (self.gen_size, self.sample_output_height, self.sample_output_width, self.c_dim)

        image3,predictions = model([depth, depth_small, image])

        predictions = tf.dtypes.cast(predictions * 127.5 + 127.5, tf.uint8)

        image3 = tf.dtypes.cast(image3 * 127.5+127.5, tf.uint8)

        for i in range(predictions.shape[0]):
            filename = os.path.join(
                self.results_dir, ('gen_image_at_epoch_{:04d}_at_image_{:04d}.png'.format(epoch, i)))
            imageio.imwrite(filename, (predictions[i, :, :, :]))
            filename = os.path.join(
                self.sample_dir, ('image_at_epoch_{:04d}_at_image_{:04d}.png'.format(epoch, i)))
            imageio.imwrite(filename, (image3[i, :, :, :]))

            # filename = os.path.join(
            #     self.back_dir, ('depth_at_epoch_{:04d}_at_image_{:04d}.mat'.format(epoch, i)))
            # dep = np.array((depth[i, :, :]))
            # sio.savemat(filename, {'name': dep})


    def train_step(self, depth_batch_images, air_batch_images, water_batch_images):
        ##model
        for update_g in range(2):
            with tf.GradientTape() as gen_tape:
                assert depth_batch_images.shape==(self.batch_size,self.output_height,self.output_width)
                assert air_batch_images.shape==(self.batch_size,self.output_height,self.output_width,self.c_dim)
                [image3, generated_images,atten_images,before_camera_images] = self.generator(
                    [depth_batch_images,air_batch_images])
                fake_output = self.discriminator(inputs=generated_images)
                gen_loss = self.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        with tf.GradientTape() as disc_tape:
            assert depth_batch_images.shape == (self.batch_size, self.output_height, self.output_width)

            assert air_batch_images.shape == (self.batch_size, self.output_height, self.output_width, self.c_dim)

            [image3, generated_images, atten_images, before_camera_images] = self.generator(
                [depth_batch_images, air_batch_images])
            eta_r = tf.convert_to_tensor(self.attenuation_layer.eta_r.numpy()[0][0])
            eta_g = tf.convert_to_tensor(self.attenuation_layer.eta_g.numpy()[0][0])
            eta_b = tf.convert_to_tensor(self.attenuation_layer.eta_b.numpy()[0][0])

            real_output = self.discriminator(inputs=water_batch_images)

            fake_output = self.discriminator(inputs=generated_images)
            gen_loss = self.generator_loss(fake_output)
            real_loss, fake_loss, disc_loss = self.discriminator_loss(real_output, fake_output)


        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return eta_r,eta_g,eta_b,real_output,fake_output,gen_loss,real_loss,fake_loss,disc_loss,generated_images,atten_images



    def discriminator_loss(self,real_output, fake_output):
        real_loss =self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return real_loss,fake_loss,total_loss

    def generator_loss(self,fake_output):
        eta_r = tf.convert_to_tensor(self.attenuation_layer.eta_r.numpy()[0][0])
        eta_g = tf.convert_to_tensor(self.attenuation_layer.eta_g.numpy()[0][0])
        eta_b= tf.convert_to_tensor(self.attenuation_layer.eta_b.numpy()[0][0])
        eta_r_loss=-tf.minimum(eta_r,0)*10000
        eta_g_loss=-tf.minimum(eta_g,0)*10000
        eta_b_loss=-tf.minimum(eta_b,0)*10000
        eta_loss = tf.maximum(0,eta_g-eta_r)*10000+tf.maximum(0,eta_b-eta_r)*10000
        r = tf.convert_to_tensor(self.preprocess.r.numpy()[0][0])
        g = tf.convert_to_tensor(self.preprocess.g.numpy()[0][0])
        b = tf.convert_to_tensor(self.preprocess.b.numpy()[0][0])
        preprocess_loss = tf.maximum(0, 0.4-r)*10000+tf.maximum(0, r-1.4)*10000+\
                          tf.maximum(0, 0.8-g)*10000+tf.maximum(0, g-2.0)*10000+\
                          tf.maximum(0, 0.8-b)*10000+tf.maximum(0, b-2.0)*10000
        return eta_r_loss+eta_g_loss+eta_b_loss+ eta_loss+preprocess_loss+\
               self.cross_entropy(tf.ones_like(fake_output), fake_output) #+C1_loss+C2_loss+A_loss\


    def read_img(self, filename):
        # print(filename)
        try:
            imgtmp = imageio.imread(filename)
            # imgtmp.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError, OSError) as e:
            print( filename)
            # os.remove(base_dir+"\\"+filename) (Maybe)
        # imgtmp = imageio.imread(filename)
        img = np.array(Image.fromarray(imgtmp).resize((self.output_width,self.output_height))).astype(np.float32)
        img=(img-127.5 ) / 127.5
        return img
    def read_sample_img(self, filename):
        imgtmp = imageio.imread(filename)
        img = np.array(Image.fromarray(imgtmp).resize((self.sample_output_width,self.sample_output_height))).astype(np.float32)
        img=(img-127.5 ) / 127.5
        return img

    def read_depth(self, filename):
        # print(filename)
        depth_mat = sio.loadmat(filename)
        depthtmp = depth_mat["depth"]
        depthtmp=np.array(depthtmp*10000).astype(np.uint32)

        depth = np.array(Image.fromarray(depthtmp).resize((self.output_width, self.output_height))).astype(np.float32)
        depth = np.multiply(self.max_depth, np.divide(depth,  np.amax(depth)))
        return depth

    def read_depth_sample(self, filename):
        depth_mat = sio.loadmat(filename)
        depthtmp = depth_mat["depth"]
        depthtmp = np.array(depthtmp*10000).astype(np.uint32)
        depth = np.array(Image.fromarray(depthtmp).resize((self.sample_output_width, self.sample_output_height))).astype(np.float32)
        depth = np.multiply(self.max_depth, np.divide(depth, np.amax(depth)))
        return depth

