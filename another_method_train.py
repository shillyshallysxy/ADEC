# -*- encoding:utf8 -*-import tensorflow as tf
from dec.dataset import *
import os
import argparse
from dec.model import *
import prior_factory as prior
import logging
import plot_utils as pu
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from multiprocessing.pool import Pool


logging.basicConfig(filename="./base.log",
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filemode='a',
                    level=logging.INFO)

logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
file_handler = logging.FileHandler("./base.log")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# logger.info = print
tf.reset_default_graph()


def train(dataset,
          learn_rate=1e-4,
          prior_type='uniform',
          pretrained_ae_ckpt_path=None,
          pretrained_aae_ckpt_path=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logger.info("using prior: {}".format(prior_type))

    pool_ = Pool(4)

    if dataset == 'MNIST':
        data = MNIST()
        data_name = "MNIST"
        w_init = "kaiming_uniform"
        encoder_dims = [500, 500, 1000, 10]
        discriminator_dims = [1000, 1]
        stack_ae = True
        update_interval = 100
        update_aae_mu_interval = 10000
        aae_finetune_iteration = 30000
        initialize_iteration = 50000
        finetune_iteration = 100000
        finetune_epoch = 200
        aae_finetune_epoch = 40
        batch_size = 256
        aae_ae_enhance = 1
    elif dataset == "StackOverflow":
        data = StackOverflow()
        data_name = dataset
        encoder_dims = [500, 500, 2000, 20]
        discriminator_dims = [1000, 1]
        w_init = "glorot_uniform"
        stack_ae = False
        update_interval = 500
        aae_finetune_iteration = 5000
        update_aae_mu_interval = 5000
        finetune_epoch = 15
        aae_finetune_epoch = None
        batch_size = 64
        aae_ae_enhance = 1
        finetune_iteration = finetune_epoch*(data.train_y.shape[0]/batch_size)
    else:
        assert False, "Undefined dataset."
    logger.info("running on data set: {}".format(dataset))

    dec_aae_model = DEC_AAE(params={
        "encoder_dims": encoder_dims,
        "n_clusters": data.num_classes,
        "input_dim": data.feature_dim,
        "alpha": 1.0,
        "discriminator_dims": discriminator_dims,
        "learn_rate": learn_rate,
        "w_init": w_init
    })
    if dataset == 'MNIST':
        # learning_rate = tf.train.exponential_decay(learning_rate=0.1,
        #                                            global_step=tf.train.get_or_create_global_step(),
        #                                            decay_steps=20000,
        #                                            decay_rate=0.1,
        #                                            staircase=True)
        # dec_aae_model.dec.ae.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).\
        #     minimize(dec_aae_model.dec.ae.loss)
        dec_aae_model.dec.ae.optimizer = tf.train.AdamOptimizer(0.0001). \
                minimize(dec_aae_model.dec.ae.loss)
    elif dataset == "StackOverflow":
        dec_aae_model.dec.ae.optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).\
            minimize(dec_aae_model.dec.ae.loss)

    ae_saver = tf.train.Saver(var_list=dec_aae_model.ae_vars, max_to_keep=None)
    aae_saver = tf.train.Saver(var_list=dec_aae_model.d_vars+dec_aae_model.ae_vars, max_to_keep=None)
    dec_saver = tf.train.Saver(var_list=dec_aae_model.dec_vars, max_to_keep=None)
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
    # phase 1: ae parameter initialization
    log_interval = 500
    if pretrained_ae_ckpt_path is None:
        logger.info("pre training auto encoder")
        sae = StackedAutoEncoder(encoder_dims=encoder_dims, input_dim=data.feature_dim)
        ae_ckpt_path = os.path.join('ae_ckpt', 'model{}.ckpt'.format(data_name))

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if stack_ae:
                # initialize sae
                next_ = data.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)
                cur_ae_data = data.train_x
                for i, sub_ae in enumerate(sae.layerwise_autoencoders):
                    # train sub_ae
                    for iter_, (batch_x, _, _) in enumerate(next_):
                        _, loss = sess.run([sub_ae.optimizer, sub_ae.loss], feed_dict={sub_ae.input_: batch_x,
                                                                                       sub_ae.keep_prob: 0.8})
                        if iter_%log_interval==0:
                            logger.info("[SAE-{}] iter: {}\tloss: {}".format(i, iter_, loss))

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
                                                                        # iteration=finetune_iteration,
                                                                        epoch=finetune_epoch
                                                                        )):
                _, loss = sess.run([dec_aae_model.dec.ae.optimizer, dec_aae_model.dec.ae.loss], feed_dict={dec_aae_model.dec.ae.input_: batch_x,
                                                                                                           dec_aae_model.dec.ae.keep_prob: 1.0})
                if iter_%log_interval==0:
                    logger.info("[AE-finetune] iter: {}\tloss: {}".format(iter_, loss))
                if iter_ % (10*log_interval) == 0:
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
        logger.info("pre training adversarial auto encoder")
        aae_ckpt_path = os.path.join('aae_ckpt', 'model{}.ckpt'.format(data_name))
        # aae_ckpt_path = os.path.join('aae_ckpt', 'model.ckpt-100000')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ae_saver.restore(sess, ae_ckpt_path)
            # aae_saver.restore(sess, aae_ckpt_path)
            z = sess.run(dec_aae_model.z, feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
            assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
            _ = sess.run(assign_mu_op)
            mu = sess.run(dec_aae_model.dec.mu)
            total_y = data.train_y
            total_pred = sess.run(dec_aae_model.dec.pred, feed_dict={dec_aae_model.input_: data.train_x,
                                                                     dec_aae_model.batch_size: data.train_x.shape[
                                                                         0],
                                                                     dec_aae_model.keep_prob: 1.0})
            logger.info("[Total DEC] epoch: {}\tacc: {}".
                        format(-1, dec_aae_model.dec.cluster_acc(total_y, total_pred)))

            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                       is_train_set=True,
                                                                                       # iteration=aae_finetune_iteration,
                                                                                       epoch=aae_finetune_epoch,
                                                                                       )):
                # if iter_ % update_aae_mu_interval == 0 and iter_ != 0:
                #     z = sess.run(dec_aae_model.z,
                #                  feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
                #     assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
                #     _ = sess.run(assign_mu_op)
                #     mu = sess.run(dec_aae_model.dec.mu)

                z_sample, z_id_one_hot, z_id_ = \
                    prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim, n_labels=data.num_classes, mu=mu)
                train_dec_feed = {dec_aae_model.input_: batch_x,
                                  dec_aae_model.batch_size: batch_x.shape[0],
                                  dec_aae_model.keep_prob: 1,
                                  dec_aae_model.z_sample: z_sample,}

                # if iter_ < 100:
                #     # discriminator loss
                #     _, d_loss = sess.run(
                #         (dec_aae_model.train_op_d, dec_aae_model.D_loss), feed_dict=train_dec_feed)
                #     logger.info("[ADVER] epoch %d:  d_loss %03.2f" % (
                #         iter_, d_loss))
                #     continue
                for _ in range(aae_ae_enhance):
                    # reconstruction loss
                    _, ae_loss = sess.run(
                        (dec_aae_model.train_op_ae, dec_aae_model.ae_loss), feed_dict=train_dec_feed)
                    #
                # discriminator loss
                _, d_loss = sess.run(
                    (dec_aae_model.train_op_d, dec_aae_model.D_loss), feed_dict=train_dec_feed)
                #
                # generator loss
                _, g_loss = sess.run(
                    (dec_aae_model.train_op_g, dec_aae_model.G_loss),
                    feed_dict=train_dec_feed)
                #
                tot_loss = ae_loss + d_loss + g_loss
                #
                if iter_ % 500 == 0:
                    # logger.info cost every epoch
                    logger.info("[ADVER] epoch %d: L_tot %03.4f L_likelihood %03.4f d_loss %03.2f g_loss %03.4f" % (
                        iter_, tot_loss, ae_loss, d_loss, g_loss))
                if iter_ % 2500 == 0:
                    # logger.info cost every epoch

                    xmlr_x = data.train_x[:10000, :]
                    xmlr_id = data.train_y[:10000]
                    z = sess.run(dec_aae_model.z,
                                 feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
                    # pu.save_scattered_image(z, xmlr_id, "./results/z_map_{}.jpg".format(iter_))

                    # pred_y = sess.run(dec_aae_model.dec.pred,
                    #              feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0,
                    #                         dec_aae_model.batch_size: data.train_x.shape[0]
                    #                         })
                    # logger.info("[Total DEC] iteration: {}\targ_acc: {}".
                    #             format(iter_, dec_aae_model.dec.cluster_acc(data.train_y, pred_y)))
                    #
                    # kmeans = KMeans(n_clusters=data.num_classes, n_init=20)
                    # pred_y = kmeans.fit_predict(z)
                    # logger.info("[Total DEC] iteration: {}\tkmeans_acc: {}".
                    #             format(iter_, dec_aae_model.dec.cluster_acc(data.train_y, pred_y)))
                    z = z[:10000]
                    pool_.apply_async(pu.save_scattered_image, (z, xmlr_id, "./results/z_aae_map_{}.jpg".format(iter_)))

            aae_saver.save(sess, aae_ckpt_path)

        pool_.close()  # 关闭进程池，表示不能在往进程池中添加进程
        pool_.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        exit()
    else:
        aae_ckpt_path = pretrained_aae_ckpt_path

    # phase 3: parameter optimization
    dec_ckpt_path = os.path.join('dec_ckpt', 'model{}.ckpt'.format(data_name))
    t_ckpt_path = os.path.join('adver_ckpt', 'model{}.ckpt'.format(data_name))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        retrain = False
        dec_mode = True
        idec_mode = False
        adec_mode = False
        best_score = 0.
        if dec_mode or idec_mode:
            if retrain:
                logger.info("retraining the dec")
                saver.restore(sess, t_ckpt_path)
                bais = 100
            else:
                logger.info("training the dec")
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

                total_y = data.train_y
                total_pred = sess.run(dec_aae_model.dec.pred, feed_dict={dec_aae_model.input_: data.train_x,
                                                                         dec_aae_model.batch_size: data.train_x.shape[
                                                                             0],
                                                                         dec_aae_model.keep_prob: 1.0})
                logger.info("[Total DEC] epoch: {}\tacc: {}".
                            format(-1, dec_aae_model.dec.cluster_acc(total_y, total_pred)))

                # print("sstart")
                # total_y = total_y[:10000]
                # z = z[:10000]
                # from sklearn.manifold import TSNE
                # z = TSNE(n_components=2, learning_rate=100).fit_transform(z)
                # kmeans = KMeans(n_clusters=data.num_classes, n_init=20)
                # pred_y = kmeans.fit_predict(z)
                # print(pu.cluster_acc(total_y, pred_y))
                # exit()

        else:
            if retrain:
                logger.info("retraining the adec")
                bais = 100
                saver.restore(sess, t_ckpt_path)
            else:
                logger.info("training the adec")
                aae_saver.restore(sess, aae_ckpt_path)
                # ae_saver.restore(sess, ae_ckpt_path)
                bais = 0
                # initialize mu
                z = sess.run(dec_aae_model.z, feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
                assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
                _ = sess.run(assign_mu_op)

                total_y = data.train_y
                total_pred = sess.run(dec_aae_model.dec.pred, feed_dict={dec_aae_model.input_: data.train_x,
                                                                         dec_aae_model.batch_size: data.train_x.shape[
                                                                             0],
                                                                         dec_aae_model.keep_prob: 1.0})
                logger.info("[Total ADEC] epoch: {}\tacc: {}".
                            format(-1, dec_aae_model.dec.cluster_acc(total_y, total_pred)))
                pool_.apply_async(pu.save_scattered_image,
                                  (z[:10000, ], total_y[:10000], "./results/z_adec_map_{}.jpg".format(-1), total_pred[:10000]))

        mu = sess.run(dec_aae_model.dec.mu)
        p = None
        for cur_epoch in range(100):
            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                       is_train_set=True,
                                                                                       epoch=1,
                                                                                       # iteration=50000
                                                                                       )):
                if cur_epoch % 10 == 0 and iter_ == 0:
                    q = sess.run(dec_aae_model.dec.q, feed_dict={
                        dec_aae_model.input_: data.train_x,
                        dec_aae_model.batch_size: data.train_x.shape[0],
                        dec_aae_model.keep_prob: 1.0})
                    p = dec_aae_model.dec.target_distribution(q)

                # if (iter_+1) % 10000 == 0:
                #     z = sess.run(dec_aae_model.z,
                #                  feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
                #     assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
                #     _ = sess.run(assign_mu_op)
                #     mu = sess.run(dec_aae_model.dec.mu)

                batch_p = p[batch_idxs]
                train_dec_feed = {dec_aae_model.input_: batch_x,
                                  dec_aae_model.batch_size: batch_x.shape[0],
                                  dec_aae_model.dec.p: batch_p,
                                  dec_aae_model.keep_prob: 1.,}

                # ==========================adversial part ============================
                z_sample, z_id_one_hot, z_id_ = \
                    prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim, n_labels=data.num_classes, mu=mu)
                train_dec_feed.update({
                                  dec_aae_model.z_sample: z_sample,
                                  })
                # ==========================adversial part ============================

                if dec_mode:
                    # logger.info("DEC mode")
                    _, loss, pred = sess.run([dec_aae_model.train_op_dec,
                                              dec_aae_model.dec_loss, dec_aae_model.dec.pred],
                                             feed_dict=train_dec_feed)
                elif idec_mode:
                    # logger.info("IDEC mode")
                    _, loss, pred = sess.run([dec_aae_model.train_op_idec,
                                              dec_aae_model.idec_loss, dec_aae_model.dec.pred],
                                             feed_dict=train_dec_feed)
                elif adec_mode:
                    # logger.info("ADEC mode")
                    _, loss, pred = sess.run([dec_aae_model.train_op_adec,
                                              dec_aae_model.adec_loss, dec_aae_model.dec.pred],
                                             feed_dict=train_dec_feed)
                    ae_loss, g_loss, d_loss = \
                        sess.run([dec_aae_model.ae_loss, dec_aae_model.G_loss, dec_aae_model.D_loss],
                                 feed_dict=train_dec_feed)
                    tot_loss = ae_loss+g_loss+d_loss
                else:
                    raise ValueError("没有这个模式！")

                # if iter_ % 100 == 0:
                    # logger.info cost every epoch
                    # logger.info("[ADVER] epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (
                    #     cur_epoch, tot_loss, ae_loss, d_loss, g_loss))
                    # ==========================adversial part ============================
                    # logger.info("[DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch+bais, loss,
                    #                                               dec_aae_model.dec.cluster_acc(batch_y, pred)))
                if iter_ % 2500 == 0:
                    total_y = data.train_y
                    total_pred = sess.run(dec_aae_model.dec.pred,
                                          feed_dict={dec_aae_model.input_: data.train_x,
                                                     dec_aae_model.batch_size: data.train_x.shape[0],
                                                     dec_aae_model.keep_prob: 1.0})
                    now_score = pu.cluster_acc(total_y, total_pred)
                    now_nmi = pu.cluster_nmi(total_y, total_pred)
                    if adec_mode:
                        logger.info("[ADVER] epoch %d: L_tot %03.4f L_likelihood %03.4f d_loss %03.2f g_loss %03.4f" % (
                            cur_epoch, tot_loss, ae_loss, d_loss, g_loss))
                    logger.info("[Total DEC] iteration: {}\tloss: {}\tacc: {}\tnmi: {}".
                                format(iter_, loss, now_score, now_nmi))
                    if now_score > best_score:
                        best_score = now_score
                        saver.save(sess, t_ckpt_path)
                if iter_ % 5000 == 0:
                    xmlr_x = data.train_x[:10000, :]
                    xmlr_id = data.train_y[:10000]
                    z, xmlr_pred_id = sess.run([dec_aae_model.z, dec_aae_model.dec.pred],
                                               feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0,
                                                          dec_aae_model.batch_size: xmlr_x.shape[0]})
                    pool_.apply_async(pu.save_scattered_image, (z, xmlr_id, "./results/z_adec_map_{}.jpg".format(iter_), xmlr_pred_id))

            total_y = data.train_y
            total_pred = sess.run(dec_aae_model.dec.pred, feed_dict={dec_aae_model.input_: data.train_x,
                                                         dec_aae_model.batch_size: data.train_x.shape[0],
                                                         dec_aae_model.keep_prob: 1.0})
            logger.info("[Total DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch+bais, loss,
                                                              dec_aae_model.dec.cluster_acc(total_y, total_pred)))
            # dec_saver.save(sess, dec_ckpt_path)

    pool_.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool_.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用


if __name__ == "__main__":
    desc = "Tensorflow implementation of (ADEC)"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--batch-size", dest="batch_size", help="Train Batch Size", default=256, type=int)
    parser.add_argument("--gpu-index", dest="gpu_index", help="GPU Index Number", default="0", type=str)
    parser.add_argument("--prior_type", dest="prior_type",
                        help="[mixGaussian, uniform, swiss_roll, normal, dirichlet, loc_normal]",
                        default="normal", type=str)
    parser.add_argument("--data_name", dest="data_name", help="[MNIST, StackOverflow]", default="MNIST", type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    train(dataset=args.data_name,
          pretrained_ae_ckpt_path='./ae_ckpt/model{}.ckpt'.format(args.data_name),
          # pretrained_ae_ckpt_path=None,
          pretrained_aae_ckpt_path='./aae_ckpt/model{}.ckpt'.format(args.data_name),
          # pretrained_aae_ckpt_path=None,
          prior_type=args.prior_type,
          )

    # eval(batch_size=args['batch_size'],
    #       dataset="MNIST",)
