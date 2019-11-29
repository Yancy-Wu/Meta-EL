'''
    test zel on zeshel datasets.
'''

import torch
from zel.trainer import Trainer
from zel.similar_net import SimilarNet
from zel.prediction import Prediction
from zel.adapter import Adapter
from zel.datasets.zeshel import Zeshel
from tensorizer.bert_tokenizer import EasyBertTokenizer
from modules.bert import Bert

def main():
    '''
        zel pipeline example.
    '''

    # create datasets. provide train and eval data.
    dataset = Zeshel('../datasets/zeshel')

    # tensorizer. convert an example to tensors.
    tensorizer = EasyBertTokenizer.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'FIXED_LEN': 16
    })

    # adapter. call tensorizer, convert a batch of examples to big tensors.
    adapter = Adapter(tensorizer, tensorizer)

    # embedding model. for predication.
    bert = Bert.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'POOLING_METHOD': 'avg',
        'FINETUNE_LAYER_RANGE': '10:12'
    })

    # siamese bert for training.
    model = SimilarNet(bert, bert, bert.config.hidden_size, {
        'DROP_OUT_PROB': 0.1,
        'ACT_NAME': 'relu',
        'USE_BIAS': False
    })

    # trainer. to train siamese bert.
    trainer = Trainer({
        'dataset': dataset,
        'adapter': adapter,
        'model': model,
        'DEVICE': torch.device('cuda:2'),
        'TRAIN_BATCH_SIZE': 100,
        'VALID_BATCH_SIZE': 100,
        'ROUND': 5
    })

    # train start here.
    trainer.train()

    # train done, fetch bert model to prediction.
    prediction = Prediction(model, adapter, {
        'TOP_K': 1,
        'EMB_BATCH_SIZE': 1000,
        'DEVICE': torch.device('cuda:0')
    })
    # add candidates.
    prediction.add_candidate(*dataset.all_candidates())

    # we start predict query.
    print(prediction.predict(['Washton', 'apple']))

    prediction.save('../trained/zel_zeshel_bert_35388.pkl')

if __name__ == '__main__':
    main()
