# -*- coding: utf-8 -*-
import os
import shutil
import pickle
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.kbqa.dataset.freebase_helper import FreebaseHelper
from src.kbqa.dataset.feature_helper import FeatureHelper
from src.kbqa.dataset.schema_dataset import SchemaDataset
from src.kbqa.dataset.schema_builder import SchemaBuilder
from src.kbqa.model.kbqa_model import KbqaModel
from src.kbqa.learner.evaluator import Evaluator
from src.kbqa.learner.optimizer import Optimizer
from src.kbqa.utils.embedding_util import WordEmbeddingUtil
from src.kbqa.utils.log_util import LogInfo
from src.kbqa.utils import model_util


parser = argparse.ArgumentParser(description='KBQA Model Training')

# data config arguments
parser.add_argument('--data_dir', type=str, help='dataset dir')
parser.add_argument('--candgen_dir', type=str, help='dataset candidates dir')
parser.add_argument('--emb_dir', type=str, help='word/mid embedding dir')
parser.add_argument('--fb_meta_dir', type=str, help='Freebase metadata dir')
parser.add_argument('--output_dir', type=str, help='output dir, including results, models and others')

# schema config arguments
parser.add_argument('--schema_level', type=str, default='strict', choices=['strict', 'elegant', 'coherent', 'general'])
parser.add_argument('--qw_max_len', type=int, default=20, help='max length of question')
parser.add_argument('--pw_max_len', type=int, default=8, help='max length of path at word level')
parser.add_argument('--path_max_size', type=int, default=3, help='max size of path')
parser.add_argument('--pseq_max_len', type=int, default=3, help='max length of path at sequence level')

# negative sampling config arguments
parser.add_argument('--neg_f1_ths', type=float, default=0.1, help='negative sampling threshold')
parser.add_argument('--neg_max_sample', type=int, default=20, help='maximum negative sampling')
parser.add_argument('--neg_strategy', type=str, default='Fix', choices=['Fix', 'Dyn'])

# model config arguments
parser.add_argument('--cell_class', type=str, default='GRU', choices=['RNN', 'GRU', 'LSTM'])
parser.add_argument('--num_units', type=int, default=256, help='num units of RNN cell')
parser.add_argument('--num_layers', type=int, default=1, help='num layers of RNN cell')
parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
parser.add_argument('--n_words', type=int, default=61814, help='num words')
parser.add_argument('--n_mids', type=int, default=3561, help='num mids')
parser.add_argument('--n_paths', type=int, default=3561, help='num paths')
parser.add_argument('--w_emb_fix', type=str, default='Upd', choices=['Upd', 'Fix'])
parser.add_argument('--dep_simulate', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--att_func', type=str, default='noAtt', choices=['noAtt', 'dot', 'bahdabau', 'bilinear', 'bdot'])
parser.add_argument('--dim_att_hidden', type=int, default=128, help='attention dimension')
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--path_usage', type=str, default='BH', choices=['X|B|R X|B|R|H'])
parser.add_argument('--sent_usage', type=str, default='mSum', choices=['mSum', 'mMax', 'mlp', 'qwOnly', 'depOnly'])
parser.add_argument('--seq_merge_mode', type=str, default='fwbw', choices=['fwbw', 'avg', 'max'])
parser.add_argument('--scoring_mode', type=str, default='separated', choices=['separated', 'compact'])
parser.add_argument('--final_func', type=str, default='fc512', help="matching function, ['bilinear', 'fcxxx']")

# training config arguments
parser.add_argument('--loss_margin', type=float, default=0.5, help='hinge loss margin')
parser.add_argument('--optm_name', type=str, default='Adam', choices=['Adam', 'Adadelta', 'Adagrad', 'GradientDescent'])
parser.add_argument('--lr_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--optm_batch_size', type=int, default=128, help='optm_batch size')
parser.add_argument('--eval_batch_size', type=int, default=512, help='eval_batch size')
parser.add_argument('--max_epoch', type=int, default=20, help='max epochs')
parser.add_argument('--max_patience', type=int, default=20, help='max patience')
parser.add_argument('--resume_model_name', default='model_best',
                    help='the directory name of the model which you wan to resume learning')
parser.add_argument('--gpu_fraction', type=float, default=0.25, help='GPU fraction limit')
parser.add_argument('--verbose', type=int, default=1, help='verbose level')


def main(args):

    # ==== Loading Necessary Utils ====
    LogInfo.begin_track('Loading Utils ... ')
    wd_emb_util = WordEmbeddingUtil(emb_dir=args.emb_dir, dim_emb=args.dim_emb)
    freebase_helper = FreebaseHelper(meta_dir=args.fb_meta_dir)
    LogInfo.end_track()

    # ==== Loading Dataset ====
    LogInfo.begin_track('Creating Dataset ... ')
    schema_dataset = SchemaDataset(data_dir=args.data_dir, candgen_dir=args.candgen_dir,
                                   schema_level=args.schema_level, freebase_helper=freebase_helper)
    schema_dataset.load_all_data()
    active_dicts = schema_dataset.active_dicts
    qa_list = schema_dataset.qa_list
    feature_helper = FeatureHelper(active_dicts, qa_list, freebase_helper, path_max_size=args.path_max_size,
                                   qw_max_len=args.qw_max_len, pw_max_len=args.pw_max_len,
                                   pseq_max_len=args.pseq_max_len)
    ds_builder = SchemaBuilder(schema_dataset=schema_dataset, feature_helper=feature_helper,
                               neg_f1_ths=args.neg_f1_ths, neg_max_sample=args.neg_max_sample,
                               neg_strategy=args.neg_strategy)
    LogInfo.end_track()

    # ==== Building Model ====
    LogInfo.begin_track('Building Model and Session ... ')
    model_config = {'qw_max_len': args.qw_max_len,
                    'pw_max_len': args.pw_max_len,
                    'path_max_size': args.path_max_size, 
                    'pseq_max_len': args.pseq_max_len,
                    'dim_emb': args.dim_emb, 
                    'w_emb_fix': args.w_emb_fix, 
                    'n_words': args.n_words, 
                    'n_mids': args.n_mids, 
                    'n_paths': args.n_paths, 
                    'drop_rate': args.drop_rate,
                    'rnn_config': {'cell_class': args.cell_class,
                                   'num_units': args.num_units,
                                   'num_layers': args.num_layers}, 
                    'att_config': {'att_func': args.att_func,
                                   'dim_att_hidden': args.dim_att_hidden},
                    'path_usage': args.path_usage, 
                    'sent_usage': args.sent_usage, 
                    'seq_merge_mode': args.seq_merge_mode, 
                    'scoring_mode': args.scoring_mode,
                    'final_func': args.final_func,
                    'loss_margin': args.loss_margin, 
                    'optm_name': args.optm_name, 
                    'learning_rate': args.lr_rate
                    }
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    with open("%s/config.json" % args.output_dir, 'w') as fw:
        json.dump(model_config, fw)

    kbqa_model = KbqaModel(**model_config)

    LogInfo.logs('Showing final parameters: ')
    for var in tf.global_variables():
        LogInfo.logs('%s: %s', var.name, var.get_shape().as_list())

    LogInfo.end_track()

    # ==== Focused on specific params ====
    if args.final_func == 'bilinear':
        focus_param_name_list = ['rm_task/rm_forward/bilinear_mat']
    else:   # mlp
        focus_param_name_list = ['rm_task/rm_forward/fc1/weights',
                                 'rm_task/rm_forward/fc1/biases',
                                 'rm_task/rm_forward/fc2/weights',
                                 'rm_task/rm_forward/fc2/biases']
    focus_param_list = []
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for param_name in focus_param_name_list:
            try:
                var = tf.get_variable(name=param_name)
                focus_param_list.append(var)
            except ValueError:
                LogInfo.logs("ValueError occured for %s!" % param_name)
                pass
    LogInfo.begin_track('Showing %d concern parameters: ', len(focus_param_list))
    for name, tensor in zip(focus_param_name_list, focus_param_list):
        LogInfo.logs('%s --> %s', name, tensor.get_shape().as_list())
    LogInfo.end_track()

    # ==== Initializing model ====
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            intra_op_parallelism_threads=8))
    LogInfo.begin_track('Running global_variables_initializer ...')
    start_epoch = 0
    best_valid_f1 = 0.
    resume_flag = False
    model_dir = None
    if args.resume_model_name not in ('', 'None'):
        model_dir = '%s/%s' % (args.output_dir, args.resume_model_name)
        if os.path.exists(model_dir):
            resume_flag = True
    if resume_flag:
        start_epoch, best_valid_f1 = model_util.load_model(saver=saver, sess=sess, model_dir=model_dir)
    else:
        dep_simulate = True if args.dep_simulate == 'True' else False
        wd_emb_mat = wd_emb_util.produce_active_word_embedding(
            active_word_dict=schema_dataset.active_dicts['word'],
            dep_simulate=dep_simulate
        )
        pa_emb_mat = np.random.uniform(low=-0.1, high=0.1,
                                       size=(model_config['n_paths'],
                                             model_config['dim_emb'])).astype('float32')
        mid_emb_mat = np.random.uniform(low=-0.1, high=0.1,
                                        size=(model_config['n_mids'],
                                              model_config['dim_emb'])).astype('float32')
        LogInfo.logs('%s random path embedding created.', pa_emb_mat.shape)
        LogInfo.logs('%s random mid embedding created.', mid_emb_mat.shape)
        sess.run(tf.global_variables_initializer(),
                 feed_dict={kbqa_model.w_embedding_init: wd_emb_mat,
                            kbqa_model.p_embedding_init: pa_emb_mat,
                            kbqa_model.m_embedding_init: mid_emb_mat})
    LogInfo.end_track('Model build complete.')

    # ==== Running optm / eval ====
    optimizer = Optimizer(model=kbqa_model, sess=sess)
    evaluator = Evaluator(model=kbqa_model, sess=sess)
    optm_data_loader = ds_builder.build_optm_dataloader(optm_batch_size=args.optm_batch_size)
    eval_data_list = ds_builder.build_eval_dataloader(eval_batch_size=args.eval_batch_size)

    if not os.path.exists('%s/detail' % args.output_dir):
        os.mkdir('%s/detail' % args.output_dir)
    if not os.path.exists('%s/result' % args.output_dir):
        os.mkdir('%s/result' % args.output_dir)

    LogInfo.begin_track('Learning start ...')

    patience = args.max_patience
    for epoch in range(start_epoch+1, args.max_epoch+1):
        if patience == 0:
            LogInfo.logs('Early stopping at epoch = %d.', epoch)
            break
        update_flag = False
        disp_item_dict = {'Epoch': epoch}

        LogInfo.begin_track('Epoch %d / %d', epoch, args.max_epoch)

        LogInfo.begin_track('Optimizing ... ')
        optimizer.optimize_all(optm_data_loader=optm_data_loader)

        LogInfo.logs('loss = %.6f', optimizer.ret_loss)
        disp_item_dict['rm_loss'] = optimizer.ret_loss
        LogInfo.end_track()

        LogInfo.begin_track('Evaluation:')
        for mark, eval_dl in zip(['train', 'valid', 'test'], eval_data_list):
            LogInfo.begin_track('Eval-%s ...', mark)
            disp_key = '%s_F1' % mark
            detail_fp = '%s/detail/%s.tmp' % (args.output_dir, mark)
            result_fp = '%s/result/%s.%03d.result' % (args.output_dir, mark, epoch)
            disp_item_dict[disp_key] = evaluator.evaluate_all(
                eval_data_loader=eval_dl,
                detail_fp=detail_fp,
                result_fp=result_fp
            )
            LogInfo.end_track()
        LogInfo.end_track()

        # Display & save states (results, details, params)
        cur_valid_f1 = disp_item_dict['valid_F1']
        if cur_valid_f1 > best_valid_f1:
            best_valid_f1 = cur_valid_f1
            update_flag = True
            patience = args.max_patience
            save_best_dir = '%s/model_best' % args.output_dir
            model_util.delete_dir(save_best_dir)
            model_util.save_model(saver=saver, sess=sess, model_dir=save_best_dir,
                                  epoch=epoch, valid_metric=best_valid_f1)
        else:
            patience -= 1
        LogInfo.logs('Model %s, best valid_F1 = %.6f [patience = %d]',
                     'updated' if update_flag else 'stayed', cur_valid_f1, patience)
        disp_item_dict['Status'] = 'UPDATE' if update_flag else str(patience)
        disp_item_dict['Time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        status_fp = '%s/status.txt' % args.output_dir
        disp_header_list = model_util.construct_display_header()
        if epoch == 1:
            with open(status_fp, 'w') as bw:
                write_str = ''.join(disp_header_list)
                bw.write(write_str + '\n')
        with open(status_fp, 'a') as bw:
            write_str = ''
            for item_idx, header in enumerate(disp_header_list):
                if header.endswith(' ') or header == '\t':
                    write_str += header
                else:
                    val = disp_item_dict.get(header, '--------')
                    if isinstance(val, float):
                        write_str += '%8.6f' % val
                    else:
                        write_str += str(val)
            bw.write(write_str + '\n')

        LogInfo.logs('Output concern parameters ... ')
        # don't need any feeds, since we focus on parameters
        param_result_list = sess.run(focus_param_list)
        param_result_dict = {}
        for param_name, param_result in zip(focus_param_name_list, param_result_list):
            param_result_dict[param_name] = param_result

        with open(args.output_dir + '/detail/param.%03d.pkl' % epoch, 'wb') as bw:
            pickle.dump(param_result_dict, bw)
        LogInfo.logs('Concern parameters saved.')

        if update_flag:
            with open(args.output_dir + '/detail/param.best.pkl', 'wb') as bw:
                pickle.dump(param_result_dict, bw)
            # save the latest details
            for mode in ['train', 'valid', 'test']:
                src = '%s/detail/%s.tmp' % (args.output_dir, mode)
                dest = '%s/detail/%s.best' % (args.output_dir, mode)
                if os.path.isfile(src):                                             
                    shutil.move(src, dest)

        LogInfo.end_track()     # end of epoch
    LogInfo.end_track()         # end of learning


if __name__ == '__main__':
    LogInfo.begin_track('KBQA running ...')
    _args = parser.parse_args()
    main(_args)
    LogInfo.end_track('All Done.')
