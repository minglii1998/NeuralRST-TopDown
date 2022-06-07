import sys, time
import os
import numpy as np
from datetime import datetime
import random

from zmq import device

sys.path.append(".")

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adamax
from torch.nn.utils import clip_grad_norm_

from in_out.reader_lm import Reader
from in_out.util import load_embedding_dict, get_logger
from in_out.preprocess_lm import construct_embedding_table
from in_out.preprocess_lm import create_alphabet
from in_out.preprocess_lm import validate_gold_actions, validate_gold_top_down
from in_out.preprocess_lm import batch_data_variable
from models.metric import Metric
from models.vocab_lm import Vocab
from models.config import Config
from models.architecture_sa_lm import MainArchitecture as SAMainArchitecture
from models.architecture_lm import MainArchitecture


# from torch.utils.tensorboard import SummaryWriter  
from in_out.logging_utils import SelfLogger


UNK_ID=0
main_path=''
device = 'cuda'

def set_label_action(dictionary, instances):
    for i in range(len(instances)):
        for j in range(len(instances[i].gold_actions)):
            instances[i].gold_actions[j].set_label_id(dictionary)
    return instances


def predict(network, instances, vocab, config, logger):
    time_start = datetime.now()
    span = Metric(); nuclear = Metric(); relation = Metric(); full = Metric() # evaluation for RST parseval
    span_ori = Metric(); nuclear_ori = Metric(); relation_ori = Metric(); full_ori = Metric() # evaluation for original parseval (recommended by Morey et al., 2017)
    predictions = []; predictions_ori = [];
    total_data = len(instances)
    for i in range(0, total_data, config.batch_size):
        end_index = i+config.batch_size
        if end_index > total_data:
            end_index = total_data
        indices = np.array((range(i, end_index)))
        subset_data = batch_data_variable(instances, indices, vocab, config)
        predictions_of_subtrees_ori, prediction_of_subtrees = network.loss(subset_data, None)
        predictions += prediction_of_subtrees
        predictions_ori += predictions_of_subtrees_ori
    for i in range(len(instances)):
        span, nuclear, relation, full = instances[i].evaluate(predictions[i], span, nuclear, relation, full) 
        span_ori, nuclear_ori, relation_ori, full_ori = instances[i].evaluate_original_parseval(predictions_ori[i], span_ori, nuclear_ori, relation_ori, full_ori)
    time_elapsed = datetime.now() - time_start
    m,s = divmod(time_elapsed.seconds, 60)
    
    logger.info('Finished in {} mins {} secs'.format(m,s))
    logger.info("S (rst): " + span.print_metric())
    logger.info("N (rst): " + nuclear.print_metric())
    logger.info("R (rst): " + relation.print_metric())
    logger.info("F (rst): " + full.print_metric())
    logger.info("------------------------------------------------------------------------------------")
    logger.info("S (ori): " + span_ori.print_metric())
    logger.info("N (ori): " + nuclear_ori.print_metric())
    logger.info("R (ori): " + relation_ori.print_metric())
    logger.info("F (ori): " + full_ori.print_metric())
    return span, nuclear, relation, full, span_ori, nuclear_ori, relation_ori, full_ori


def get_test_loss(network, instances, vocab, config):
    network.eval()
    network.training = False

    costs = []
    nuc_rel_loss_list = []
    seg_loss_list = []

    total_data = len(instances)
    for i in range(0, total_data, config.batch_size):
        end_index = i+config.batch_size
        if end_index > total_data:
            end_index = total_data
        indices = np.array((range(i, end_index)))
        subset_data = batch_data_variable(instances, indices, vocab, config)
        cost, cost_val, nuc_rel_loss, seg_loss = network.loss(subset_data, None, loss_valid=True)
        costs.append(cost_val)
        nuc_rel_loss_list.append(nuc_rel_loss)
        seg_loss_list.append(seg_loss)

    return np.mean(costs), np.mean(nuc_rel_loss_list), np.mean(seg_loss_list)

def main():
    start_a = time.time()
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--word_embedding', default='t5', help='Embedding for words')
    args_parser.add_argument('--train_path', default=main_path+'NeuralRST/rst.train312')  
    args_parser.add_argument('--test_path', default=main_path+'NeuralRST/rst.test38')  
    args_parser.add_argument('--dev_path', default=main_path+'NeuralRST/rst.dev35')  
    args_parser.add_argument('--model_path', default=main_path+'logs_new')
    args_parser.add_argument('--max_iter', type=int, default=1000, help='maximum epoch')
   
    args_parser.add_argument('--word_dim', type=int, default=768, help='768 for base, 1024 for large')
    args_parser.add_argument('--etype_dim', type=int, default=100, help='Dimension of Etype embeddings')
    args_parser.add_argument('--freeze', default=True, help='frozen the word embedding (disable fine-tuning).')
    
    args_parser.add_argument('--max_sent_size', type=int, default=100, help='maximum word size in 1 edu')
    # two args below are not used in top-down discourse parsing
    args_parser.add_argument('--max_edu_size', type=int, default=400, help='maximum edu size')
    args_parser.add_argument('--max_state_size', type=int, default=1024, help='maximum decoding steps')
    
    args_parser.add_argument('--hidden_size', type=int, default=256, help='')
    args_parser.add_argument('--hidden_size_tagger', type=int, default=128, help='')

    args_parser.add_argument('--drop_prob', type=float, default=0.5, help='default drop_prob')
    args_parser.add_argument('--num_layers', type=int, default=1, help='number of RNN layers')
    
    args_parser.add_argument('--batch_size', type=int, default=1, help='Number of sentences in each batch')
    args_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args_parser.add_argument('--ada_eps', type=float, default=1e-6, help='epsilon for adam or adamax')
    args_parser.add_argument('--opt', default='adam', help='Optimization, choose between adam, sgd, and adamax')
    args_parser.add_argument('--start_decay', type=int, default=0, help='')
    args_parser.add_argument('--grad_accum', type=int, default=2, help='gradient accumulation setting')

    args_parser.add_argument('--beta1', type=float, default = 0.9, help='beta1 for adam')
    args_parser.add_argument('--beta2', type=float, default = 0.999, help='beta2 for adam')
    args_parser.add_argument('--momentum', type=float, default = 0.9, help='momentum')
    args_parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    args_parser.add_argument('--clip', type=float, default=10.0, help='gradient clipping')

    args_parser.add_argument('--loss_nuc_rel', type=float, default = 1.0, help='weight for nucleus and relation loss')
    args_parser.add_argument('--loss_seg', type=float, default = 1.0, help='weight for segmentation loss')
    args_parser.add_argument('--activate_nuc_rel_loss', type=int, default = 0, help='set the starting epoch for nuclear loss')
    args_parser.add_argument('--activate_seg_loss', type=int, default = 0, help='set the starting epoch for segmentation loss')

    args_parser.add_argument('--decay', type=int, default=0, help='')
    args_parser.add_argument('--oracle_prob', type=float, default=0.66666, help='')
    args_parser.add_argument('--flag_oracle', default=False, help='')
    args_parser.add_argument('--start_dynamic_oracle', type=int, default=20, help='')
    args_parser.add_argument('--use_dynamic_oracle', type=int, default=0, help='')
    args_parser.add_argument('--early_stopping', type=int, default=50, help='')
    
    args_parser.add_argument('--beam_search', type=int, default=1, help='assign parameter k for beam search')
    args_parser.add_argument('--depth_alpha', type=float, default=0.0, help='multiplier of loss based on depth of the subtree')
    args_parser.add_argument('--elem_alpha', type=float, default=0.35, help='multiplier of loss based on number of element in a subtree')
    args_parser.add_argument('--seed', type=int, default=999, help='random seed')

    args_parser.add_argument('--model', type=str, default='sa', help='what model to use, sa ori')
    args_parser.add_argument('--special_tag', type=str, default='', help='special tag for logging')
    args_parser.add_argument('--quick_embedding', type=str, default='', help='pass of saved embeddings')
    args_parser.add_argument('--decode_layer', type=str, default='lstm', help='segmentor type in decoder')
    args_parser.add_argument('--resume', type=str, default='', help='checkpoint path')
    args_parser.add_argument('--milestone', type=int, nargs='+', default=[],help='Decrease learning rate at these epochs.')
    args_parser.add_argument('--keep_lstm', type=bool, default=False, help='use lstm instead of doc sa')
    args_parser.add_argument('--using_etype', type=bool, default=False, help='whether use etype')

    # not used in this language model (lm) version model
    args_parser.add_argument('--word_embedding_file', default=main_path+'NeuralRST/glove.6B.200d.txt.gz')
    args_parser.add_argument('--train_syn_feat_path', default=main_path+'NeuralRST/SyntaxBiaffine/train.conll.dump.results')  
    args_parser.add_argument('--test_syn_feat_path', default=main_path+'NeuralRST/SyntaxBiaffine/test.conll.dump.results')  
    args_parser.add_argument('--dev_syn_feat_path', default=main_path+'NeuralRST/SyntaxBiaffine/dev.conll.dump.results') 
    args_parser.add_argument('--tag_dim', type=int, default=200, help='Dimension of POS tag embeddings')
    args_parser.add_argument('--syntax_dim', type=int, default=1200, help='Dimension of Sytax embeddings')
    args_parser.add_argument('--experiment', help='Name of your experiment', type=str, default='exp_try')
    args_parser.add_argument('--model_name', default='network.pt')

    config = args_parser.parse_args()
    config.betas = (config.beta1, config.beta2) 

    np.random.seed(config.seed)
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    logger = SelfLogger('lm',config)
    config.alphabet_path = logger.get_alphabet_path()
    config.model_name = logger.get_model_path()
    if config.use_dynamic_oracle:
        logger.info("This is using DYNAMIC oracle, and will be activated at Epoch %d" %(config.start_dynamic_oracle))
    else:
        logger.info("This is using STATIC oracle")
    
    # logger.info("Load word embedding, will take 2 minutes")
    # pretrained_embed, word_dim = load_embedding_dict(config.word_embedding, config.word_embedding_file)
    # assert (word_dim == config.word_dim)

    if config.quick_embedding != '':
        quick_train_path = os.path.join(config.quick_embedding,config.word_embedding+'_train_instances_'+str(config.word_dim)+'.pt')
        quick_dev_path = os.path.join(config.quick_embedding,config.word_embedding+'_dev_instances_'+str(config.word_dim)+'.pt')
        quick_test_path = os.path.join(config.quick_embedding,config.word_embedding+'_test_instances_'+str(config.word_dim)+'.pt')


    logger.info("Reading Train start")
    if config.quick_embedding == '':
        reader = Reader(config.train_path, config.train_syn_feat_path, config.word_embedding)
        train_instances = reader.read_data()
    else:
        train_instances = torch.load(quick_train_path)
    logger.info('Finish reading training instances: ' + str(len(train_instances)))
    logger.info('Max sentence size: ' + str(config.max_sent_size))

    logger.info('Creating Alphabet....')
    gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha, etype_alpha = create_alphabet(train_instances, config.alphabet_path, logger)
    vocab = Vocab(etype_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha)
    set_label_action(action_label_alpha.alpha2id, train_instances) 

    # logger.info('Checking Gold Actions for transition-based parser....')
    # validate_gold_actions(train_instances, config.max_state_size)
    logger.info('Checking Gold Labels for top-down parser....')
    validate_gold_top_down(train_instances)
    
    # word_table = construct_embedding_table(word_alpha, config.word_dim, config.freeze, pretrained_embed)
    # tag_table = construct_embedding_table(tag_alpha, config.tag_dim, config.freeze)
    etype_table = construct_embedding_table(etype_alpha, config.etype_dim, config.freeze)
    
    logger.info("Finish reading train data by:" + str(round(time.time() - start_a, 2)) + 'sec')

    # DEV data processing
    if config.quick_embedding == '':
        reader = Reader(config.dev_path, config.dev_syn_feat_path, config.word_embedding)
        dev_instances = reader.read_data()
    else:
        dev_instances = torch.load(quick_dev_path)
    logger.info('Finish reading dev instances')
    
    # TEST data processing
    if config.quick_embedding == '':
        reader = Reader(config.test_path, config.test_syn_feat_path, config.word_embedding)
        test_instances  = reader.read_data()
    else:
        test_instances = torch.load(quick_test_path)
    logger.info('Finish reading test instances')

    torch.set_num_threads(4)
    if config.model == 'sa':
        network = SAMainArchitecture(vocab, config, etype_table) 
    elif config.model == 'ori':
        network = MainArchitecture(vocab, config, etype_table) 
    logger.write_model(network)
   
    # if config.freeze:
    #     network.word_embedd.freeze()
    if device == 'cpu':
        config.use_gpu = False
    else:
        config.use_gpu = True

    if config.use_gpu:
        network.to(device)
    
    # Set-up Optimizer
    def generate_optimizer(config, params):
        params = filter(lambda param: param.requires_grad, params)
        if config.opt == 'adam':
            return Adam(params, lr=config.lr, betas=config.betas, weight_decay=config.gamma, eps=config.ada_eps)
        elif config.opt == 'sgd':
            return SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.start_decay, nesterov=True)
        elif config.opt == 'adamax':
            return Adamax(params, lr=config.lr, betas=config.betas, weight_decay=config.start_decay, eps=config.ada_eps)
        else:
            raise ValueError('Unknown optimization algorithm: %s' % config.opt)

    optim = generate_optimizer(config, network.parameters())
    opt_info = 'opt: %s, ' % config.opt
    if config.opt == 'adam':
        opt_info += 'betas=%s, eps=%.1e, lr=%.5f, weight_decay=%.1e' % (config.betas, config.ada_eps, config.lr, config.gamma)
    elif config.opt == 'sgd':
        opt_info += 'momentum=%.2f' % config.momentum
    elif config.opt == 'adamax':
        opt_info += 'betas=%s, eps=%.1e, lr=%f' % (config.betas, config.ada_eps, config.lr)
    if len(config.milestone) > 0:
        scheduler = optim.lr_scheduler.MultiStepLR(optim, milestones=config.milestone, gamma=0.1)

    logger.info(opt_info)

    def get_subtrees(data, indices):
        subtrees = []
        for i in indices:
            subtrees.append(data[i].result)
        return subtrees

    # START TRAINING
    batch_size = config.batch_size
    logger.info('Start doing training....')
    total_data = len(train_instances)
    logger.info('Batch size: %d' % batch_size)
    num_batch = total_data / batch_size + 1
    es_counter = 0
    best_S = 0; best_S_ori = 0;
    best_N = 0; best_N_ori = 0;
    best_R = 0; best_R_ori = 0;
    best_F = 0; best_F_ori = 0;
    iteration = -1

    if config.resume == '':
        start_epo = 0
    else:
        checkpoint = torch.load(config.resume)
        network.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epo = checkpoint['epoch'] + 1
    
    for epoch in range(start_epo, config.max_iter):
        if len(config.milestone) > 0:
            scheduler.step(epoch)

        logger.info('Epoch %d ' % (epoch))
        logger.info("Current learning rate: %.5f" %(config.lr))
        
        if epoch == config.start_dynamic_oracle and config.use_dynamic_oracle:
            logger.info("In this epoch, dynamic oracle is activated!")
            config.flag_oracle = True
        
        permutation = torch.randperm(total_data).long()  
        network.metric_span.reset()
        network.metric_nuclear_relation.reset()
        time_start = datetime.now()
        costs = []
        nuc_rel_loss_list = []
        seg_loss_list = []
        counter_acc = 0
        for i in range(0, total_data, batch_size):
            network.train()
            network.training = True
            
            indices = permutation[i: i+batch_size]
            subset_data = batch_data_variable(train_instances, indices, vocab, config)
            gold_subtrees = get_subtrees(train_instances, indices)
            
            cost, cost_val, nuc_rel_loss, seg_loss = network.loss(subset_data, gold_subtrees, epoch=epoch)
            costs.append(cost_val)
            nuc_rel_loss_list.append(nuc_rel_loss)
            seg_loss_list.append(seg_loss)
            cost.backward()
            counter_acc += 1
            
            if config.grad_accum > 1 and counter_acc == config.grad_accum:
                clip_grad_norm_(network.parameters(), config.clip)
                optim.step()
                network.zero_grad()
                counter_acc = 0
            elif config.grad_accum == 1:
                optim.step()
                network.zero_grad()
                counter_acc = 0

            time_elapsed = datetime.now() - time_start
            m,s = divmod(time_elapsed.seconds, 60)
            logger.info('Epoch %d, Batch %d, AvgCost: %.2f, CorrectSpan: %.2f, CorrectNuclearRelation: %.2f - {} mins {} secs'.format(m,s) % (epoch, (i+batch_size) / batch_size, 
                    np.mean(costs), network.metric_span.get_accuracy(), network.metric_nuclear_relation.get_accuracy()))

        logger.tb_writer.add_scalar('1_training/loss',np.mean(costs),epoch)
        logger.tb_writer.add_scalar('2_evaluating/nuc_rel_loss',np.mean(nuc_rel_loss_list),epoch)
        logger.tb_writer.add_scalar('2_evaluating/seg_loss',np.mean(seg_loss_list),epoch)
        logger.record_loss([np.mean(costs),np.mean(nuc_rel_loss_list),np.mean(seg_loss_list)],'train',epoch)

        logger.save_checkpoint(network, optim, epoch)
            
        # # Perform evaluation if span accuracy is at leas 0.8 OR when dynamic oracle is activated
        # if network.metric_span.get_accuracy() < 0.8 and not config.flag_oracle:
        #     logger.info('We only perform test for DEV and TEST set if the span accuracy >= 0.80')
        #     continue
        
        logger.info('Batch ends, performing test for DEV and TEST set')
        # START EVALUATING DEV:
        network.eval()
        network.training = False

        torch.cuda.empty_cache()
        logger.info('Evaluate DEV:')
        span, nuclear, relation, full, span_ori, nuclear_ori, relation_ori, full_ori =\
                predict(network, dev_instances, vocab, config, logger)
        
        mean_loss, nuc_rel_loss, seg_loss = get_test_loss(network, dev_instances, vocab, config)
        logger.tb_writer.add_scalar('2_evaluating/loss',mean_loss,epoch)
        logger.tb_writer.add_scalar('2_evaluating/nuc_rel_loss',nuc_rel_loss,epoch)
        logger.tb_writer.add_scalar('2_evaluating/seg_loss',seg_loss,epoch)
        logger.record_loss([mean_loss,nuc_rel_loss,seg_loss],'valid',epoch)


        # mean_loss, nuc_rel_loss, seg_loss = get_test_loss(network, test_instances, vocab, config)
        # tb_writer.add_scalar('3_testing/loss',mean_loss,epoch)
        # tb_writer.add_scalar('3_testing/nuc_rel_loss',nuc_rel_loss,epoch)
        # tb_writer.add_scalar('3_testing/seg_loss',seg_loss,epoch)
        
        if best_F < full.get_f_measure():
            best_S = span.get_f_measure(); best_S_ori = span_ori.get_f_measure()
            best_N = nuclear.get_f_measure(); best_N_ori = nuclear_ori.get_f_measure()
            best_R = relation.get_f_measure(); best_R_ori = relation_ori.get_f_measure()
            best_F = full.get_f_measure(); best_F_ori = full_ori.get_f_measure()
            iteration = epoch
            #save the model
            torch.save(network.state_dict(), config.model_name)
            logger.info('Model is successfully saved')
            es_counter = 0
        else:
            logger.info("NOT exceed best Full F-score: history = %.2f, current = %.2f" % (best_F, full.get_f_measure()))
            logger.info("Best dev performance in Iteration %d with result S (rst): %.4f, N (rst): %.4f, R (rst): %.4f, F (rst): %.4f" %(iteration, best_S, best_N, best_R, best_F))
            #logger.info("Best dev performance in Iteration %d with result S (ori): %.4f, N (ori): %.4f, R (ori): %.4f, F (ori): %.4f" %(iteration, best_S_ori, best_N_ori, best_R_ori, best_F_ori))
            if es_counter > config.early_stopping:
                logger.info('Early stopping after getting lower DEV performance in %d consecutive epoch. BYE, Assalamualaikum!' %(es_counter) )
                sys.exit()
            es_counter += 1
        
        # START EVALUATING TEST:
        torch.cuda.empty_cache()
        logger.info('Evaluate TEST:')
        span, nuclear, relation, full, span_ori, nuclear_ori, relation_ori, full_ori =\
                predict(network, test_instances, vocab, config, logger)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
