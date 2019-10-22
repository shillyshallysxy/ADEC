# -*- encoding:utf8 -*-import tensorflow as tf
from dec.dataset import *
import os
import configargparse
from dec.model import *
import prior_factory as prior
import logging
import plot_utils as pu
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logging.basicConfig(filename="base.log",
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filemode='a',
                    level=logging.INFO)


def train(dataset,
          batch_size=256,
          encoder_dims=[1000, 1000, 10],
          discriminator_dims=[10, 1000, 1],
          initialize_iteration=50000,
          finetune_iteration=100000,
          learn_rate=1e-3,
          prior_type='uniform',
          pretrained_ae_ckpt_path=None,
          pretrained_aae_ckpt_path=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logging.info("using prior: {}".format(prior_type))

    if dataset == 'MNIST':
        data = MNIST()
    else:
        assert False, "Undefined dataset."

    dec_aae_model = DEC_AAE(params={
        "encoder_dims": encoder_dims,
        "n_clusters": data.num_classes,
        "input_dim": data.feature_dim,
        "alpha": 1.0,
        "discriminator_dims": discriminator_dims,
        "learn_rate": learn_rate
    })

    ae_saver = tf.train.Saver(var_list=dec_aae_model.ae_vars, max_to_keep=None)
    aae_saver = tf.train.Saver(var_list=dec_aae_model.d_vars+dec_aae_model.ae_vars, max_to_keep=20)
    dec_saver = tf.train.Saver(var_list=dec_aae_model.dec_vars, max_to_keep=None)
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
    # phase 1: parameter initialization
    log_interval = 5000
    if pretrained_ae_ckpt_path is None:
        logging.info("pre training auto encoder")
        sae = StackedAutoEncoder(encoder_dims=encoder_dims, input_dim=data.feature_dim)
        ae_ckpt_path = os.path.join('ae_ckpt', 'model.ckpt')

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            # initialize sae
            next_ = data.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)
            cur_ae_data = data.train_x
            for i, sub_ae in enumerate(sae.layerwise_autoencoders):
                # train sub_ae
                for iter_, (batch_x, _, _) in enumerate(next_):
                    _, loss = sess.run([sub_ae.optimizer, sub_ae.loss], feed_dict={sub_ae.input_: batch_x,
                                                                                   sub_ae.keep_prob: 0.8})
                    if iter_%log_interval==0:
                        logging.info("[SAE-{}] iter: {}\tloss: {}".format(i, iter_, loss))

                # assign pretrained sub_ae's weight
                encoder_w_assign_op, encoder_b_assign_op = dec_aae_model.dec.ae.layers[i].get_assign_ops( sub_ae.layers[0] )
                decoder_w_assign_op, decoder_b_assign_op = dec_aae_model.dec.ae.layers[(i+1)*-1].get_assign_ops( sub_ae.layers[1] )
                _ = sess.run([encoder_w_assign_op, encoder_b_assign_op,
                              decoder_w_assign_op, decoder_b_assign_op])

                # get next sub_ae's input
                cur_ae_data = sess.run(sub_ae.encoder, feed_dict={sub_ae.input_: cur_ae_data,
                                                                   sub_ae.keep_prob: 1.0})
                embedding = Dataset(train_x=cur_ae_data, train_y=cur_ae_data)
                next_ = embedding.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)

            # finetune AE
            for iter_, (batch_x, _, _) in enumerate(data.gen_next_batch(batch_size=batch_size, is_train_set=True,
                                                                        iteration=finetune_iteration)):
                _, loss = sess.run([dec_aae_model.dec.ae.optimizer, dec_aae_model.dec.ae.loss], feed_dict={dec_aae_model.dec.ae.input_: batch_x,
                                                                                                           dec_aae_model.dec.ae.keep_prob: 1.0})
                if iter_%log_interval==0:
                    logging.info("[AE-finetune] iter: {}\tloss: {}".format(iter_, loss))
            ae_saver.save(sess, ae_ckpt_path)

    else:
        ae_ckpt_path = pretrained_ae_ckpt_path

    if pretrained_aae_ckpt_path is None:
        logging.info("pre training adversarial auto encoder")
        aae_ckpt_path = os.path.join('aae_ckpt', 'model.ckpt')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # ae_saver.restore(sess, ae_ckpt_path)
            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                       is_train_set=True, iteration=100000)):
                train_dec_feed = {dec_aae_model.input_: batch_x,
                                  dec_aae_model.batch_size: batch_x.shape[0],
                                  dec_aae_model.keep_prob: 0.9,}
                z_sample, z_id_one_hot, z_id_ = \
                    prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim)
                train_dec_feed.update({
                    dec_aae_model.z_sample: z_sample,
                })
                # reconstruction loss
                _, ae_loss = sess.run(
                    (dec_aae_model.train_op_ae, dec_aae_model.ae_loss), feed_dict=train_dec_feed)
                #
                # discriminator loss
                _, d_loss = sess.run(
                    (dec_aae_model.train_op_d, dec_aae_model.D_loss), feed_dict=train_dec_feed)
                #
                # generator loss
                for _ in range(2):
                    _, g_loss = sess.run(
                        (dec_aae_model.train_op_g, dec_aae_model.G_loss),
                        feed_dict=train_dec_feed)
                #
                tot_loss = ae_loss + d_loss + g_loss
                #
                if iter_ % 2500 == 0:
                    # logging.info cost every epoch
                    logging.info("[ADVER] epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (
                        iter_, tot_loss, ae_loss, d_loss, g_loss))
                if iter_ % 5000 == 0:
                    # logging.info cost every epoch
                    xmlr_x = data.train_x[:10000, :]
                    xmlr_id = data.train_y[:10000]
                    z = sess.run(dec_aae_model.z,
                                 feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0})
                    pu.save_scattered_image(z, xmlr_id, "./results/z_map_{}.jpg".format(iter_))

                    aae_saver.save(sess, aae_ckpt_path, global_step=iter_)
    else:
        aae_ckpt_path = pretrained_aae_ckpt_path
    # phase 2: parameter optimization

    # exit()
    dec_ckpt_path = os.path.join('dec_ckpt', 'model.ckpt')
    t_ckpt_path = os.path.join('adver_ckpt', 'model2.ckpt')
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # ae_saver.restore(sess, ae_ckpt_path)
        aae_saver.restore(sess, aae_ckpt_path)
        # dec_saver.restore(sess, dec_ckpt_path)
        # saver.restore(sess, t_ckpt_path)

        # initialize mu
        z = sess.run(dec_aae_model.z, feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
        assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
        _ = sess.run(assign_mu_op)

        # # show z space
        # q, mu = sess.run([dec_aae_model.dec.q, dec_aae_model.dec.mu], feed_dict={dec_aae_model.input_: data.train_x,
        #                                              dec_aae_model.batch_size: data.train_x.shape[0],
        #                                              dec_aae_model.keep_prob: 1.0})
        # p = dec_aae_model.dec.target_distribution(q)
        # total_z = np.concatenate((mu,  z[:1000, :]), axis=0)
        # X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(total_z)
        # plt.figure(1)
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[0]*10 + list(data.train_y[:1000]))
        # plt.colorbar()
        # plt.show()
        # plt.figure(2)
        # plt.scatter(X_tsne[:10, 0], X_tsne[:10, 1], c=[0] * 10)
        # plt.colorbar()
        # plt.show()
        for cur_epoch in range(100):
            # z = sess.run(dec_aae_model.z,
            #              feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
            # assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
            # _ = sess.run(assign_mu_op)

            total_y = list()
            total_pred = list()
            # per one epoch
            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                       is_train_set=True, epoch=1)):
                if iter_ % 100 == 0:
                    q = sess.run(dec_aae_model.dec.q, feed_dict={dec_aae_model.input_: data.train_x,
                                                                 dec_aae_model.batch_size: data.train_x.shape[0],
                                                                 dec_aae_model.keep_prob: 1.0})
                    p = dec_aae_model.dec.target_distribution(q)

                batch_p = p[batch_idxs]
                train_dec_feed = {dec_aae_model.input_: batch_x,
                                  dec_aae_model.batch_size: batch_x.shape[0],
                                  dec_aae_model.dec.p: batch_p,
                                  dec_aae_model.keep_prob: 0.8,}

                _, loss, pred = sess.run([dec_aae_model.train_op_dec,
                                             dec_aae_model.dec_loss, dec_aae_model.dec.pred],
                                         feed_dict=train_dec_feed)

                total_y.append(batch_y)
                total_pred.append(pred)

                # ==========================adversial part ============================
                z_sample, z_id_one_hot, z_id_ = \
                    prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim)
                train_dec_feed.update({
                                  dec_aae_model.z_sample: z_sample,
                                  })

                # discriminator loss
                # _, d_loss = sess.run(
                #     (dec_aae_model.train_op_d, dec_aae_model.D_loss), feed_dict=train_dec_feed)
                d_loss = 0

                # generator loss
                for _ in range(2):
                    _, g_loss = sess.run(
                        (dec_aae_model.train_op_g, dec_aae_model.G_loss),
                        feed_dict=train_dec_feed)

                # reconstruction loss
                _, ae_loss = sess.run(
                    (dec_aae_model.train_op_ae, dec_aae_model.ae_loss), feed_dict=train_dec_feed)
                tot_loss = ae_loss + d_loss + g_loss

                if iter_ % 100 == 0:
                    # logging.info cost every epoch
                    logging.info("[ADVER] epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (
                        cur_epoch, tot_loss, ae_loss, d_loss, g_loss))
                    # ==========================adversial part ============================
                    logging.info("[DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch, loss,
                                                                  dec_aae_model.dec.cluster_acc(batch_y, pred)))
            total_y = np.reshape(np.array(total_y), [-1])
            total_pred = np.reshape(np.array(total_pred), [-1])
            logging.info("[Total DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch, loss,
                                                              dec_aae_model.dec.cluster_acc(total_y, total_pred)))
            # dec_saver.save(sess, dec_ckpt_path)
            saver.save(sess, t_ckpt_path)


def eval(dataset,
         batch_size=256, encoder_dims=[500, 500, 2000, 10],
         discriminator_dims=[10, 1000, 1], prior_type='mixGaussian',):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if dataset == 'MNIST':
        data = MNIST()
    else:
        assert False, "Undefined dataset."

    dec_aae_model = DEC_AAE(params={
        "encoder_dims": encoder_dims,
        "n_clusters": data.num_classes,
        "input_dim": data.feature_dim,
        "alpha": 1.0,
        "discriminator_dims": discriminator_dims,
        "learn_rate": 0
    })
    t_ckpt_path = os.path.join('adver_ckpt', 'model2.ckpt')
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, t_ckpt_path)

        total_y = list()
        total_pred = list()
        total_z = list()
        # per one epoch
        for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                   is_train_set=False, epoch=1)):
            train_dec_feed = {dec_aae_model.input_: batch_x,
                              dec_aae_model.batch_size: batch_x.shape[0],
                              dec_aae_model.keep_prob: 1,}

            pred, enz = sess.run([dec_aae_model.dec.pred,dec_aae_model.z],
                                     feed_dict=train_dec_feed)
            total_z.append(enz)
            total_y.append(batch_y)
            total_pred.append(pred)

            # # ==========================adversial part ============================
            z_sample, z_id_one_hot, z_id_ = \
                prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim)
            train_dec_feed.update({
                              dec_aae_model.z_sample: z_sample,
                              })

        total_y = np.reshape(np.array(total_y), [-1])
        total_pred = np.reshape(np.array(total_pred), [-1])
        total_z = np.reshape(np.array(total_z), [-1, dec_aae_model.z_dim])
        logging.info("[Total DEC EVAL] acc: {}".format(dec_aae_model.dec.cluster_acc(total_y, total_pred)))
        total_z = total_z[:1000, :]
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(total_z)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        plt.colorbar()
        plt.show()


if __name__=="__main__":
    parser = configargparse.ArgParser()
    parser.add("--batch-size", dest="batch_size", help="Train Batch Size", default=256, type=int)
    parser.add("--gpu-index", dest="gpu_index", help="GPU Index Number", default="0", type=str)
    parser.add("--prior-type", dest="prior_type", help="Prior Type", default="mixGaussian", type=str)

    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_index']

    train(batch_size=args['batch_size'],
          dataset="MNIST",
          pretrained_ae_ckpt_path="./ae_ckpt/model.ckpt",
          # pretrained_ae_ckpt_path=None,
          pretrained_aae_ckpt_path="./aae_ckpt/model.ckpt-100000",
          # pretrained_aae_ckpt_path=None,
          )

    # eval(batch_size=args['batch_size'],
    #       dataset="MNIST",)
