# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import time
import json
import tensorflow as tf
from src.dataset.data_loader import DataLoader
from src.dataset.data_batcher import DataBatcher
from src.dataset.knowledge_loader import KnowledgeLoader
from src.kbqa.model.kbqa_model import KbqaModel
from src.model.model import TransDGModel
from src.model.model_multistep import TransDGModelMultistep
from src.utils.trainer import Trainer
from src.utils.generator import Generator


parser = argparse.ArgumentParser(description='TransDG Model')

# data config arguments
parser.add_argument('--data_dir', type=str, help='dataset dir')
parser.add_argument('--kd_dir', type=str, help='knowledge dir')
parser.add_argument('--model_dir', type=str, help='KBQA model dir')
parser.add_argument('--log_dir', type=str, help='Output log dir')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--multi_step', action="store_true")
parser.add_argument('--no_trans_repr', action="store_true")
parser.add_argument('--no_trans_select', action='store_true')
parser.add_argument('--no_use_guiding', action='store_true')

# model config arguments
parser.add_argument('--cell_class', type=str, default='GRU', choices=['RNN', 'GRU', 'LSTM'])
parser.add_argument('--num_units', type=int, default=512, help='num units of RNN cell')
parser.add_argument('--num_layers', type=int, default=2, help='num layers of RNN cell')
parser.add_argument('--dim_emb', type=int, default=300, help='word embedding dimension')
parser.add_argument('--dim_trans', type=int, default=100, help='transE embedding dimension')
parser.add_argument('--vocab_size', type=int, default=30000, help='vocabulary size')
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--max_dec_len', type=int, default=60, help='max length of decoder')

# training config arguments
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--max_epoch', type=int, default=20, help='max epochs')
parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='clip gradients to this norm')
parser.add_argument('--save_per_step', type=int, default=1000, help='steps to save checkpoint')
parser.add_argument('--ckpt', type=str, default='best.model', help='checkpoint for inference')
parser.add_argument('--verbose', type=int, default=1, help='verbose level')

# runtime arguments
parser.add_argument('--allow_soft_placement', type=bool, default=True)
parser.add_argument('--allow_gpu_growth', type=bool, default=True)


def main(args):
    config = tf.ConfigProto(allow_soft_placement=args.allow_soft_placement,
                            gpu_options=tf.GPUOptions(allow_growth=args.allow_gpu_growth))

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.no_trans_repr:
        kbqa_model = None
        trans_sess = None
    else:
        # load transferred model params
        config_path = "%s/config.json" % args.model_dir
        with open(config_path, 'r') as fr:
            kbqa_model_config = json.load(fr)

        trans_graph = tf.Graph()
        with trans_graph.as_default():
            kbqa_model = KbqaModel(**kbqa_model_config)
            trans_saver = tf.train.Saver()

        trans_sess = tf.Session(config=config, graph=trans_graph)
        model_path = '%s/model_best/best.model' % args.model_dir
        trans_saver.restore(trans_sess, save_path=model_path)

    if args.no_trans_select:
        feed_parm = None
    else:
        param_path = '%s/detail/param.best.pkl' % args.model_dir
        with open(param_path, 'rb') as fr:
            param_dict = pickle.load(fr)
            if len(param_dict.keys()) == 1:
                feed_parm = {'bilinear_mat': param_dict['rm_task/rm_forward/bilinear_mat']}
                print("bilinear_mat:", feed_parm['bilinear_mat'].shape)
            else:
                feed_parm = {'fc1_weights': param_dict['rm_task/rm_forward/fc1/weights'],
                             'fc1_biases': param_dict['rm_task/rm_forward/fc1/biases'],
                             'fc2_weights': param_dict['rm_task/rm_forward/fc2/weights'],
                             'fc2_biases': param_dict['rm_task/rm_forward/fc2/biases']}
                print("fc1_weights:", feed_parm['fc1_weights'].shape)
                print("fc1_biases:", feed_parm['fc1_biases'].shape)
                print("fc2_weights:", feed_parm['fc2_weights'].shape)
                print("fc2_biases:", feed_parm['fc2_biases'].shape)

    # load knowledge
    kd_loader = KnowledgeLoader(args.kd_dir)
    word_vocab, word_embed = kd_loader.load_vocab(vocab_size=args.vocab_size, embed_dim=args.dim_emb)
    kd_vocab, kd_embed = kd_loader.load_entity_relation()
    csk_entity_list = kd_loader.load_csk_entities()

    main_graph = tf.Graph()
    with main_graph.as_default():
        use_trans_repr = False if args.no_trans_repr else True
        use_trans_select = False if args.no_trans_select else True
        use_guiding = False if args.no_use_guiding else True
        if args.multi_step:
            model = TransDGModelMultistep(word_embed, kd_embed, feed_parm,
                                          use_trans_repr=use_trans_repr, use_trans_select=use_trans_select,
                                          use_guiding=use_guiding,
                                          vocab_size=args.vocab_size, dim_emb=args.dim_emb, dim_trans=args.dim_trans,
                                          cell_class=args.cell_class, num_units=args.num_units,
                                          num_layers=args.num_layers,
                                          max_length=args.max_dec_len, lr_rate=args.lr_rate,
                                          max_grad_norm=args.max_grad_norm, drop_rate=args.drop_rate)
        else:
            model = TransDGModel(word_embed, kd_embed, feed_parm,
                                 use_trans_repr=use_trans_repr, use_trans_select=use_trans_select,
                                 vocab_size=args.vocab_size, dim_emb=args.dim_emb, dim_trans=args.dim_trans,
                                 cell_class=args.cell_class, num_units=args.num_units, num_layers=args.num_layers,
                                 max_length=args.max_dec_len, lr_rate=args.lr_rate,
                                 max_grad_norm=args.max_grad_norm, drop_rate=args.drop_rate)
        saver = tf.train.Saver(max_to_keep=5)
        best_saver = tf.train.Saver()

    sess = tf.Session(config=config, graph=main_graph)

    if args.mode == 'train':
        if tf.train.get_checkpoint_state("%s/models" % args.log_dir):
            model_path = tf.train.latest_checkpoint("%s/models" % args.log_dir)
            print("model restored from [%s]" % model_path)
            saver.restore(sess, model_path)
        else:
            print("create model with init parameters...")
            with main_graph.as_default():
                sess.run(tf.global_variables_initializer())
                model.set_vocabs(sess, word_vocab, kd_vocab)
                if args.verbose > 0:
                    model.show_parameters()

        train_chunk_list = []
        with open("%s/all_list" % args.data_dir, 'r') as fr:
            for line in fr:
                train_chunk_list.append(line.strip())
        train_batcher = DataBatcher(data_dir=args.data_dir, file_list=train_chunk_list,
                                    batch_size=args.batch_size, num_epoch=args.max_epoch, shuffle=True)
        # wait for train_batcher queue caching
        print("Loading data from [%s/all_list]" % args.data_dir)
        while not train_batcher.full():
            time.sleep(5)
            print("loader queue caching...")

        valid_loader = DataLoader(batch_size=args.batch_size)
        valid_path = "%s/valid.pkl" % args.data_dir
        valid_loader.load_data(file_path=valid_path)

        # train model
        trainer = Trainer(model=model, sess=sess, trans_model=kbqa_model, trans_sess=trans_sess,
                          saver=saver, best_saver=best_saver, log_dir=args.log_dir, save_per_step=args.save_per_step)

        for epoch_idx in range(args.max_epoch):
            print("Epoch %d:" % (epoch_idx+1))
            trainer.train(train_batcher, valid_loader, epoch_idx=epoch_idx)

    else:
        if args.ckpt == 'best.model':
            model_path = "%s/models/best_model/best.model" % args.log_dir
        else:
            model_path = "%s/models/model-%s" % (args.log_dir, args.ckpt)
        print("model restored from [%s]" % model_path)
        saver.restore(sess, model_path)

        test_loader = DataLoader(batch_size=args.batch_size)
        test_path = "%s/test.pkl" % args.data_dir
        test_loader.load_data(file_path=test_path)

        with open('%s/stopwords' % args.kd_dir, 'r') as f:
            stop_words = json.loads(f.readline())

        # test model on test set
        generator = Generator(model=model, sess=sess, trans_model=kbqa_model, trans_sess=trans_sess,
                              log_dir=args.log_dir, ckpt=args.ckpt, stop_words=stop_words, csk_entities=csk_entity_list)
        generator.generate(test_loader)


if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
