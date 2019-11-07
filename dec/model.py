from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class AssignableDense(object):
    def __init__(self, input_, units, activation=tf.nn.relu, keep_prob=None,
                 w_init=tf.random_normal_initializer(stddev=0.01),
                 b_init=tf.constant_initializer(0.),
                 name=""):
        self.activation = activation
        self.keep_prob = keep_prob
        self.w = tf.get_variable(shape=(int(input_.shape[-1]), units), name='w{}'.format(name), initializer=w_init)
        self.b = tf.get_variable(shape=(units,), name='b{}'.format(name), initializer=b_init)
        
    def get_assign_ops(self, from_dense):
        return [tf.assign(self.w, from_dense.w), tf.assign(self.b, from_dense.b)]
    
    def apply(self, x):
        if self.keep_prob is not None:
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        x = tf.matmul(x, self.w) + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x


class StackedAutoEncoder(object):
    def __init__(self, encoder_dims, input_dim):
        self.layerwise_autoencoders = []
        
        layer_dims = [input_dim]+encoder_dims
        for i in range(1, len(layer_dims)):
            name = "sae{}".format(i)
            with tf.variable_scope(name):
                if i==1:
                    sub_ae = AutoEncoder([layer_dims[i]], layer_dims[i-1], decode_activation=False)
                elif i==len(layer_dims)-1:
                    sub_ae = AutoEncoder([layer_dims[i]], layer_dims[i-1], encode_activation=False)
                else:
                    sub_ae = AutoEncoder([layer_dims[i]], layer_dims[i-1])
                self.layerwise_autoencoders.append(sub_ae)


class AutoEncoder(object):
    def __init__(self, encoder_dims, input_dim, encode_activation=True, decode_activation=True):
        self.input_ = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.input_batch_size = tf.placeholder(tf.int32, shape=())
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.layers = []
        # w_init = tf.keras.initializers.glorot_normal()
        w_init = tf.random_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(0.)

        with tf.variable_scope("encoder"):
            # initializers
            layer = self.input_
            for i, dim in enumerate(encoder_dims):
                with tf.variable_scope("fully_layer_{}".format(i)):
                    # 1st hidden layer
                    if i != len(encoder_dims) - 1:
                        dense = AssignableDense(layer, units=dim, activation=tf.nn.relu, keep_prob=self.keep_prob,
                                                w_init=w_init, b_init=b_init)
                    else:
                        dense = AssignableDense(layer, units=dim, activation=None, keep_prob=self.keep_prob,
                                                w_init=w_init, b_init=b_init)
                    layer = dense.apply(layer)
                    self.layers.append(dense)
            self.encoder = layer
            # self.encoder = self._fully_layer(self.input_, encoder_dims, encode_activation)

        with tf.variable_scope("decoder"):
            decoder_dims = list(reversed(encoder_dims[:-1])) + [input_dim]
            layer = self.encoder
            for i, dim in enumerate(decoder_dims):
                with tf.variable_scope("fully_layer_{}".format(i)):
                    # 1st hidden layer
                    if i != len(encoder_dims) - 1:
                        dense = AssignableDense(layer, units=dim, activation=tf.nn.relu, keep_prob=self.keep_prob,
                                                w_init=w_init, b_init=b_init)
                    else:
                        dense = AssignableDense(layer, units=dim, activation=None, keep_prob=self.keep_prob,
                                                w_init=w_init, b_init=b_init)
                    layer = dense.apply(layer)
                    self.layers.append(dense)
            self.decoder = layer
            # self.decoder = self._fully_layer(self.encoder, decoder_dims, decode_activation)

        with tf.name_scope("sae-train"):
            self.loss = tf.losses.mean_squared_error(self.input_, self.decoder)
            learning_rate = tf.train.exponential_decay(learning_rate=0.1, 
                                                       global_step=tf.train.get_or_create_global_step(),
                                                       decay_steps=20000,
                                                       decay_rate=0.1,
                                                       staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss)
    
    def _fully_layer(self, x, dims, last_activation=False, name=""):
        layer = x
        for i, dim in enumerate(dims):
            with tf.variable_scope("fully_layer_{}_{}".format(name, i)):
                layer = tf.nn.dropout(layer, keep_prob=self.keep_prob)
                if last_activation==False and i==len(dims)-1:
                    dense = AssignableDense(layer, units=dim, activation=None)
                else:
                    dense = AssignableDense(layer, units=dim, activation=tf.nn.relu)
                self.layers.append(dense)
                layer = dense.apply(layer)
        return layer

    
class DEC(object):
    def __init__(self, params):
        self.n_cluster = params["n_clusters"]
        self.kmeans = KMeans(n_clusters=params["n_clusters"], n_init=20)
        self.ae = AutoEncoder(params["encoder_dims"], params['input_dim'], encode_activation=False, decode_activation=False)
        self.alpha = params['alpha']

        with tf.name_scope("distribution"):
            self.mu = tf.Variable(tf.zeros(shape=(params["n_clusters"], params["encoder_dims"][-1])), name="mu")

            self.z = self.ae.encoder
            self.q = self._soft_assignment(self.z, self.mu)
            self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))
    
            self.pred = tf.argmax(self.q, axis=1)
        
        with tf.name_scope("dec-train"):
            self.loss = self._kl_divergence(self.p, self.q)
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
#             self.optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(self.loss)
            
    def get_assign_cluster_centers_op(self, features):
        # init mu
        print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.
        
        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)
            
        Return:
            q_i_j: (num_points, num_cluster)
        """
        def _pairwise_euclidean_distance(a,b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.ae.input_batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+1.0)/2.0)
        q = (q/tf.reduce_sum(q, axis=1, keepdims=True))
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p
    
    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target*tf.log(target/(pred)), axis=1))
    
    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        num_ind = [w[i, j] for i, j in zip(*ind)]
        return sum(num_ind) * 1.0 / y_pred.size


def discriminator(z, discriminator_dims, keep_prob, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        layer = z
        w_init = tf.keras.initializers.glorot_normal()

        b_init = tf.constant_initializer(0.)
        # initializers
        w = list()
        b = list()
        for i, dim in enumerate(discriminator_dims):
            # 1st hidden layer
            w.append(tf.get_variable('w{}'.format(i), [layer.get_shape()[1], dim], initializer=w_init))
            b.append(tf.get_variable('b{}'.format(i), [dim], initializer=b_init))
            layer = tf.matmul(layer, w[i]) + b[i]
            if i != len(discriminator_dims) - 1:
                layer = tf.nn.relu(layer)
                layer = tf.nn.dropout(layer, keep_prob=keep_prob)

    return tf.sigmoid(layer), layer


class DEC_AAE(object):
    def __init__(self, params):
        self.dec = DEC(params={
            "encoder_dims": params["encoder_dims"],
            "n_clusters": params["n_clusters"],
            "input_dim": params['input_dim'],
            "alpha": 1.0
        })
        learn_rate = params["learn_rate"]
        discriminator_dims = params["discriminator_dims"]
        self.input_ = self.dec.ae.input_
        self.keep_prob = self.dec.ae.keep_prob
        self.batch_size = self.dec.ae.input_batch_size
        self.z = self.dec.z
        self.z_dim = params["encoder_dims"][-1]
        self.z_sample = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='prior_sample')
        # self.z_id = tf.placeholder(tf.float32, shape=[None, params["n_clusters"]], name='prior_sample_label')
        # self.x_id = tf.placeholder(tf.float32, shape=[None, params["n_clusters"]], name='input_img_label')
        # 1 不考虑label信息
        z_real = self.z_sample
        z_fake = self.z
        # 2 融入label信息
        # z_real = tf.concat([self.z_sample, self.z_id], axis=1)
        # z_fake = tf.concat([self.z, self.x_id], axis=1)
        self.D_fake, self.D_fake_logits = discriminator(z_fake, discriminator_dims, self.keep_prob)
        self.D_real, self.D_real_logits = discriminator(z_real, discriminator_dims, self.keep_prob, reuse=True)
        self.ae_loss = self.dec.ae.loss
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits,
                                                    labels=tf.ones_like(self.D_real_logits)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,
                                                    labels=tf.zeros_like(self.D_fake_logits)))
        self.D_loss = self.D_loss_real + self.D_loss_fake

        # generator loss
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,
                                                    labels=tf.ones_like(self.D_fake_logits)))
        self.D_loss = tf.reduce_mean(self.D_loss)
        self.G_loss = tf.reduce_mean(self.G_loss)
        self.dec_loss = self.dec.loss
        self.idec_loss = tf.reduce_mean(self.dec_loss+self.ae_loss)
        # self.adec_loss = tf.reduce_mean(self.dec_loss+self.ae_loss+0.01*self.G_loss)
        self.adec_loss = tf.reduce_mean(self.dec_loss+self.ae_loss+0.1*self.G_loss+0.1*self.D_loss)
        self.adec_loss_s = tf.reduce_mean(0.1*self.dec_loss+self.ae_loss+0.01*self.G_loss+0.01*self.D_loss)

        # optimization
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if "discriminator" in var.name]
        self.g_vars = [var for var in t_vars if "encoder" in var.name]
        self.de_vars = [var for var in t_vars if "decoder" in var.name]
        self.ae_vars = self.de_vars + self.g_vars
        self.dec_vars = [var for var in t_vars if "distribution" in var.name] + self.g_vars
        self.idec_vars = self.dec_vars + self.de_vars
        # self.adec_vars = self.dec_vars + self.de_vars
        self.adec_vars = self.dec_vars + self.de_vars + self.d_vars

        # 预训练阶段
        pre_train_ae_learning_rate = 1e-1
        self.train_op_ae = tf.train.MomentumOptimizer(pre_train_ae_learning_rate, 0.9).minimize(self.ae_loss, var_list=self.ae_vars)
        self.train_op_d = tf.train.MomentumOptimizer(pre_train_ae_learning_rate/5., 0.9).minimize(self.D_loss, var_list=self.d_vars)
        self.train_op_g = tf.train.MomentumOptimizer(pre_train_ae_learning_rate/5, 0.9).minimize(self.G_loss, var_list=self.g_vars)
        # ADEC阶段
        # self.train_op_ae = tf.train.MomentumOptimizer(learn_rate, 0.99).minimize(self.ae_loss, var_list=self.ae_vars)
        # self.train_op_d = tf.train.MomentumOptimizer(learn_rate / 5, 0.99).minimize(self.D_loss, var_list=self.d_vars)
        # self.train_op_g = tf.train.MomentumOptimizer(learn_rate, 0.99).minimize(self.G_loss, var_list=self.g_vars)
        # self.train_op_dec = tf.train.MomentumOptimizer(learn_rate/10, 0.99).minimize(self.dec_loss, var_list=self.dec_vars)

        self.train_op_dec = tf.train.AdamOptimizer(learn_rate, beta1=0.9, beta2=0.999).minimize(self.dec_loss, var_list=self.dec_vars)
        self.train_op_idec = tf.train.AdamOptimizer(learn_rate, beta1=0.9, beta2=0.999).minimize(self.idec_loss, var_list=self.idec_vars)
        self.train_op_adec = tf.train.MomentumOptimizer(learn_rate, 0.99).minimize(self.adec_loss, var_list=self.adec_vars)
        self.train_op_adec_s = tf.train.MomentumOptimizer(learn_rate, 0.99).minimize(self.adec_loss_s, var_list=self.adec_vars)

        self.y = self.dec.ae.decoder
