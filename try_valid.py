import torch

# from in_out.reader import Reader


# train_path = 'NeuralRST/rst.train312'
# train_syn_feat_path = 'NeuralRST/SyntaxBiaffine/train.conll.dump.results'

# reader = Reader(train_path, train_syn_feat_path)
# train_instances  = reader.read_data()


def check_one_instance(instance_ori,instance_t5):

    # check edu
    assert(len(instance_ori.edus) == len(instance_t5.edus))
    for i in range(len(instance_ori.edus)):
        edu_i_ori = instance_ori.edus[i]
        edu_i_t5 = instance_t5.edus[i]
        assert(edu_i_ori.etype == edu_i_t5.etype)


    # check gold_actions
    assert(len(instance_ori.gold_actions) == len(instance_t5.gold_actions))
    for i in range(len(instance_ori.gold_actions)):
        i_ori = instance_ori.gold_actions[i]
        i_t5 = instance_t5.gold_actions[i]
        assert(i_ori.__dict__ == i_t5.__dict__)

    # check gold_top_down
    assert(instance_ori.gold_top_down.__dict__ == instance_t5.gold_top_down.__dict__)

    # check result
    dic_ori = instance_ori.result.__dict__
    dic_t5 = instance_t5.result.__dict__
    assert(instance_ori.result.__dict__.keys() == instance_t5.result.__dict__.keys())
    for k in dic_ori.keys():
        i_ori = instance_ori.result.__dict__[k]
        i_t5 = instance_t5.result.__dict__[k]
        assert(len(i_ori) == len(i_t5))
        for j in range(len(i_ori)):
            assert(i_ori[j].__dict__ == i_t5[j].__dict__)
        pass

    # check tree
    def check_tree(t1,t2):
        if t1 == None and t2 == None:
            return 0
        assert(t1.__dict__.keys()==t2.__dict__.keys())
        assert(t1.__dict__['edu_span']==t2.__dict__['edu_span'])
        assert(t1.__dict__['nuclear']==t2.__dict__['nuclear'])
        assert(t1.__dict__['relation']==t2.__dict__['relation'])
        check_tree(t1.__dict__['left'],t2.__dict__['left'])
        check_tree(t1.__dict__['right'],t2.__dict__['right'])
    check_tree(instance_ori.tree,instance_t5.tree)

    pass


instances_ori = torch.load('instances_ori.pt')
instances_t5 = torch.load('instances_t5.pt')

assert(len(instances_ori) == len(instances_t5))

wrong_num = 0
for i in range(len(instances_ori)):
    try:
        check_one_instance(instances_ori[i],instances_t5[i])
    except:
        wrong_num += 0
        print(i)
    
if wrong_num == 0:
    print('All clear')
else:
    print('Wrong:',wrong_num)
    
pass