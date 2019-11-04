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
from multiprocessing.pool import Pool


logging.basicConfig(filename="base.log",
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filemode='a',
                    level=logging.INFO)
tf.reset_default_graph()


def train(dataset,
          batch_size=256,
          # encoder_dims=[1000, 1000, 10],
          encoder_dims=[500, 500, 2000, 10],
          discriminator_dims=[1000, 1],
          initialize_iteration=50000,
          finetune_iteration=100000,
          learn_rate=1e-3,
          prior_type='uniform_lab',
          pretrained_ae_ckpt_path=None,
          pretrained_aae_ckpt_path=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logging.info("using prior: {}".format(prior_type))

    pool_ = Pool(4)

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
    aae_saver = tf.train.Saver(var_list=dec_aae_model.d_vars+dec_aae_model.ae_vars, max_to_keep=None)
    dec_saver = tf.train.Saver(var_list=dec_aae_model.dec_vars, max_to_keep=None)
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
    # phase 1: ae parameter initialization
    log_interval = 5000
    if pretrained_ae_ckpt_path is None:
        logging.info("pre training auto encoder")
        sae = StackedAutoEncoder(encoder_dims=encoder_dims, input_dim=data.feature_dim)
        # tttttt = tf.trainable_variables()
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
                if iter_ % (2*log_interval) == 0:
                    xmlr_x = data.train_x[:10000, :]
                    xmlr_id = data.train_y[:10000]
                    z = sess.run(dec_aae_model.z,
                                 feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0})
                    pool_.apply_async(pu.save_scattered_image, (z, xmlr_id, "./results/z_ae_map_{}.jpg".format(iter_)))
                    # pu.save_scattered_image(z, xmlr_id, "./results/z_ae_map_{}.jpg".format(iter_))
            ae_saver.save(sess, ae_ckpt_path)
        pool_.close()  # 关闭进程池，表示不能在往进程池中添加进程
        pool_.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        exit()

    else:
        ae_ckpt_path = pretrained_ae_ckpt_path

    # exit()
    # phase 2: aae parameter initialization
    if pretrained_aae_ckpt_path is None:
        logging.info("pre training adversarial auto encoder")
        aae_ckpt_path = os.path.join('aae_ckpt', 'model.ckpt')
        # aae_ckpt_path = os.path.join('aae_ckpt', 'model.ckpt-100000')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ae_saver.restore(sess, ae_ckpt_path)
            g_loss, d_loss, ae_loss = 0, 0, 0
            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                       is_train_set=True, iteration=20000)):
                z_sample, z_id_one_hot, z_id_ = \
                    prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim)
                train_dec_feed = {dec_aae_model.input_: batch_x,
                                  dec_aae_model.batch_size: batch_x.shape[0],
                                  dec_aae_model.keep_prob: 0.9,
                                  dec_aae_model.z_sample: z_sample,}
                # reconstruction loss
                _, ae_loss = sess.run(
                    (dec_aae_model.train_op_ae, dec_aae_model.ae_loss), feed_dict=train_dec_feed)
                #
                if iter_ % 2 == 0:
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
                    aae_saver.save(sess, aae_ckpt_path)

                    xmlr_x = data.train_x[:10000, :]
                    xmlr_id = data.train_y[:10000]
                    z = sess.run(dec_aae_model.z,
                                 feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0})
                    # pu.save_scattered_image(z, xmlr_id, "./results/z_map_{}.jpg".format(iter_))
                    pool_.apply_async(pu.save_scattered_image, (z, xmlr_id, "./results/z_aae_map_{}.jpg".format(iter_)))
        pool_.close()  # 关闭进程池，表示不能在往进程池中添加进程
        pool_.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        exit()
    else:
        aae_ckpt_path = pretrained_aae_ckpt_path



    # phase 3: parameter optimization
    dec_ckpt_path = os.path.join('dec_ckpt', 'model.ckpt')
    t_ckpt_path = os.path.join('adver_ckpt', 'model.ckpt')
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        retrain = False
        dec_mode = True
        idec_mode = False
        #  dec_mode = False, idec_mode = True  ==> IDEC
        #  dec_mode = True, idec_mode = False  ==> DEC
        #  dec_mode = False, idec_mode = False  ==> ADEC
        if dec_mode:
            if retrain:
                logging.info("retraining the dec")
                saver.restore(sess, t_ckpt_path)
                bais = 100
            else:
                logging.info("training the dec")
                ae_saver.restore(sess, ae_ckpt_path)
                bais = 0
                # initialize mu
                z = sess.run(dec_aae_model.z, feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
                assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
                _ = sess.run(assign_mu_op)
                # xmlr_x = data.train_x[:10000, :]
                # xmlr_id = data.train_y[:10000]
                # z, xmlr_pred_id = sess.run([dec_aae_model.z, dec_aae_model.dec.pred],
                #                            feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0,
                #                                       dec_aae_model.batch_size: xmlr_x.shape[0]})
                # pool_.apply_async(pu.save_scattered_image,
                #                   (z, xmlr_id, "./results/z_init_map_{}.jpg".format(0 + bais), xmlr_pred_id))
                # pool_.close()  # 关闭进程池，表示不能在往进程池中添加进程
                # pool_.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
                # exit()
        else:
            if retrain:
                logging.info("retraining the adec")
                bais = 100
                saver.restore(sess, t_ckpt_path)
            else:
                logging.info("training the adec")
                # aae_saver.restore(sess, aae_ckpt_path)
                ae_saver.restore(sess, ae_ckpt_path)
                bais = 0
                # initialize mu
                z = sess.run(dec_aae_model.z, feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
                assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
                _ = sess.run(assign_mu_op)

        for cur_epoch in range(100):
            # if cur_epoch < 2:
            #     z = sess.run(dec_aae_model.z,
            #                  feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
            #     assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
            #     _ = sess.run(assign_mu_op)

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

                # ==========================adversial part ============================
                z_sample, z_id_one_hot, z_id_ = \
                    prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim)
                train_dec_feed.update({
                                  dec_aae_model.z_sample: z_sample,
                                  })
                # if not dec_mode:
                    # # reconstruction loss
                    # _, ae_loss = sess.run(
                    #     (dec_aae_model.train_op_ae, dec_aae_model.ae_loss), feed_dict=train_dec_feed)
                    #
                    # # discriminator loss
                    # # _, d_loss = sess.run(
                    # #     (dec_aae_model.train_op_d, dec_aae_model.D_loss), feed_dict=train_dec_feed)
                    #
                    # # generator loss
                    # _, g_loss = sess.run(
                    #     (dec_aae_model.train_op_g, dec_aae_model.G_loss),
                    #     feed_dict=train_dec_feed)


                # tot_loss = ae_loss + d_loss + g_loss
                # ==========================adversial part ============================

                if dec_mode:
                    # logging.info("DEC mode")
                    _, loss, pred = sess.run([dec_aae_model.train_op_dec,
                                              dec_aae_model.dec_loss, dec_aae_model.dec.pred],
                                             feed_dict=train_dec_feed)
                elif idec_mode:
                    # logging.info("IDEC mode")
                    _, loss, pred = sess.run([dec_aae_model.train_op_idec,
                                              dec_aae_model.idec_loss, dec_aae_model.dec.pred],
                                             feed_dict=train_dec_feed)
                else:
                    # logging.info("ADEC mode")
                    _, loss, pred = sess.run([dec_aae_model.train_op_adec,
                                              dec_aae_model.adec_loss, dec_aae_model.dec.pred],
                                             feed_dict=train_dec_feed)

                total_y.append(batch_y)
                total_pred.append(pred)

                if iter_ % 100 == 0:
                    # logging.info cost every epoch
                    # logging.info("[ADVER] epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (
                    #     cur_epoch, tot_loss, ae_loss, d_loss, g_loss))
                    # ==========================adversial part ============================
                    logging.info("[DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch+bais, loss,
                                                                  dec_aae_model.dec.cluster_acc(batch_y, pred)))
            if (cur_epoch+1) % 5 == 0 or cur_epoch == 0:
                xmlr_x = data.train_x[:10000, :]
                xmlr_id = data.train_y[:10000]
                z, xmlr_pred_id = sess.run([dec_aae_model.z, dec_aae_model.dec.pred],
                                           feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0,
                                                      dec_aae_model.batch_size: xmlr_x.shape[0]})
                pool_.apply_async(pu.save_scattered_image, (z, xmlr_id, "./results/z_adec_map_{}.jpg".format(cur_epoch+bais), xmlr_pred_id))

            total_y = np.reshape(np.array(total_y), [-1])
            total_pred = np.reshape(np.array(total_pred), [-1])
            logging.info("[Total DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch+bais, loss,
                                                              dec_aae_model.dec.cluster_acc(total_y, total_pred)))
            # dec_saver.save(sess, dec_ckpt_path)
        saver.save(sess, t_ckpt_path)

    pool_.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool_.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用


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
          pretrained_aae_ckpt_path="./aae_ckpt/model.ckpt",
          # pretrained_aae_ckpt_path=None,
          )

    # eval(batch_size=args['batch_size'],
    #       dataset="MNIST",)
