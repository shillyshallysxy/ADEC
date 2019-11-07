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

    pool_ = Pool(6)

    if dataset == 'MNIST':
        data = MNIST()
        data_name = ""
    elif dataset == "StackOverflow":
        data = StackOverflow()
        data_name = dataset
    else:
        assert False, "Undefined dataset."

    print("running on data set: {}".format(dataset))
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
    ae_ckpt_path = pretrained_ae_ckpt_path

    aae_ckpt_path = pretrained_aae_ckpt_path

    # phase 3: parameter optimization
    dec_ckpt_path = os.path.join('dec_ckpt', 'model{}.ckpt'.format(data_name))
    t_ckpt_path = os.path.join('adver_ckpt', 'model{}.ckpt'.format(data_name))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        retrain = False

        if retrain:
            logging.info("retraining the adec")
            bais = 0
            saver.restore(sess, t_ckpt_path)
        else:
            logging.info("training the adec")
            # aae_saver.restore(sess, aae_ckpt_path)
            ae_saver.restore(sess, ae_ckpt_path)
            bais = 0
            # initialize mu
            z = sess.run(dec_aae_model.z, feed_dict={dec_aae_model.input_: data.train_x, dec_aae_model.keep_prob: 1.0})
            # np.save("./results/z.npy", z)
            # np.save("./results/label.npy", data.train_y)
            # exit()
            assign_mu_op = dec_aae_model.dec.get_assign_cluster_centers_op(z)
            _ = sess.run(assign_mu_op)
            # for i in range(1):
            #     z_ = z[i*10000: (i+1)*10000, :]
            #     y_true = data.train_y[i*10000: (i+1)*10000]
            #     from sklearn.manifold import TSNE
            #     z_ = TSNE(n_components=2, learning_rate=100).fit_transform(z_)
            #     print(z_.shape)
            #     kmeans = KMeans(n_clusters=10, n_init=20)
            #     y_pred = kmeans.fit_predict(z_)
            #     print("acc {}: {}:".format(i, pu.cluster_acc(y_true, y_pred)))
            #     pu.save_scattered_image(z_, y_true, './results/scattered_image_10d_{}.jpg'.format(i), y_pred)
            # print(z.shape)
            # exit()

        iter_switch = 201
        for cur_epoch in range(100):

            total_y = list()
            total_pred = list()
            # per one epoch
            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                       is_train_set=True, iteration=iter_switch)):
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

            iter_switch_ae = iter_switch * min(9, cur_epoch//2)
            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
                                                                                       is_train_set=True, iteration=iter_switch_ae)):
                train_dec_feed = {dec_aae_model.input_: batch_x,
                                  dec_aae_model.batch_size: batch_x.shape[0],
                                  dec_aae_model.keep_prob: 0.8, }
                _, loss = sess.run([dec_aae_model.train_op_ae,
                                    dec_aae_model.ae_loss],
                                   feed_dict=train_dec_feed)

            if (cur_epoch+1) % 5 == 0 or cur_epoch == 0:
                xmlr_x = data.train_x[:10000, :]
                xmlr_id = data.train_y[:10000]
                z, xmlr_pred_id = sess.run([dec_aae_model.z, dec_aae_model.dec.pred],
                                           feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0,
                                                      dec_aae_model.batch_size: xmlr_x.shape[0]})
                pool_.apply_async(pu.save_scattered_image, (z, xmlr_id, "./results/z_adec_map_ae_{}.jpg".format(cur_epoch+bais), xmlr_pred_id))

            total_y = np.reshape(np.array(total_y), [-1])
            total_pred = np.reshape(np.array(total_pred), [-1])
            logging.info("[Total DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch+bais, loss,
                                                              dec_aae_model.dec.cluster_acc(total_y, total_pred)))
            # dec_saver.save(sess, dec_ckpt_path)
            # saver.save(sess, t_ckpt_path)

        # for cur_epoch in range(90):
        #     bais = 10
        #     total_y = list()
        #     total_pred = list()
        #     # per one epoch
        #     for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,
        #                                                                                is_train_set=True, epoch=1)):
        #         if iter_ % 100 == 0:
        #             q = sess.run(dec_aae_model.dec.q, feed_dict={dec_aae_model.input_: data.train_x,
        #                                                          dec_aae_model.batch_size: data.train_x.shape[0],
        #                                                          dec_aae_model.keep_prob: 1.0})
        #             p = dec_aae_model.dec.target_distribution(q)
        #
        #         batch_p = p[batch_idxs]
        #         train_dec_feed = {dec_aae_model.input_: batch_x,
        #                           dec_aae_model.batch_size: batch_x.shape[0],
        #                           dec_aae_model.dec.p: batch_p,
        #                           dec_aae_model.keep_prob: 0.8,}
        #
        #         # ==========================adversial part ============================
        #         z_sample, z_id_one_hot, z_id_ = \
        #             prior.get_sample(prior_type, batch_size, dec_aae_model.z_dim)
        #         train_dec_feed.update({
        #                           dec_aae_model.z_sample: z_sample,
        #                           })
        #
        #         # logging.info("ADEC mode")
        #         _, loss, pred = sess.run([dec_aae_model.train_op_adec_s,
        #                                   dec_aae_model.adec_loss_s, dec_aae_model.dec.pred],
        #                                  feed_dict=train_dec_feed)
        #
        #         total_y.append(batch_y)
        #         total_pred.append(pred)
        #
        #         if iter_ % 100 == 0:
        #             # logging.info cost every epoch
        #             # logging.info("[ADVER] epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (
        #             #     cur_epoch, tot_loss, ae_loss, d_loss, g_loss))
        #             # ==========================adversial part ============================
        #             logging.info("[DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch+bais, loss,
        #                                                           dec_aae_model.dec.cluster_acc(batch_y, pred)))
        #     total_y = np.reshape(np.array(total_y), [-1])
        #     total_pred = np.reshape(np.array(total_pred), [-1])
        #     logging.info("[Total DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch + bais, loss,
        #                                                                    dec_aae_model.dec.cluster_acc(total_y,
        #                                                                                                  total_pred)))
        #
        #     if (cur_epoch+1) % 5 == 0 or cur_epoch == 0:
        #         xmlr_x = data.train_x[:10000, :]
        #         xmlr_id = data.train_y[:10000]
        #         z, xmlr_pred_id = sess.run([dec_aae_model.z, dec_aae_model.dec.pred],
        #                                    feed_dict={dec_aae_model.input_: xmlr_x, dec_aae_model.keep_prob: 1.0,
        #                                               dec_aae_model.batch_size: xmlr_x.shape[0]})
        #         pool_.apply_async(pu.save_scattered_image, (z, xmlr_id, "./results/z_adec_map_{}.jpg".format(cur_epoch+bais), xmlr_pred_id))

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

    data_name = "StackOverflow"
    # data_name = "MNIST"

    train(batch_size=args['batch_size'],
          dataset=data_name,
          # pretrained_ae_ckpt_path='./ae_ckpt/model{}.ckpt'.format(data_name),
          pretrained_ae_ckpt_path=None,
          # pretrained_aae_ckpt_path='./aae_ckpt/model{}.ckpt'.format(data_name),
          pretrained_aae_ckpt_path=None,
          )

    # eval(batch_size=args['batch_size'],
    #       dataset="MNIST",)
