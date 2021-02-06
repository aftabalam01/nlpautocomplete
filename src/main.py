#!/usr/bin/env python
import os
import string
import random
import torch
import pickle
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from model import model, data_generator, engine, tokenize_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_SIZE=512
DATA_PATH = os.path.join(Path(__file__).parent, '.', 'data')
class MyModel_Runner:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        tokenize_data.tokenize_prepare_data()

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        try:
            with open(fname) as f:
                for line in f:
                    inp = line[:-1]  # the last character is a newline
                    data.append(inp)
            return data
        except FileNotFoundError:
            print(f"Test Input file {fname} is not present. Using default input")
            return ['Happ','Happy Ne','Happy New Yea','That’s one small ste','That’s one sm','That’','Th']

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        automodel, vocab, device = engine.train_eval(work_dir)

    def run_pred(self, model, data):
        # your code here
        preds = []
        for inp in data:
            # this model just predicts a random character each time
            preds.append(engine.predict_next_characters(model, device, inp, vocab=None, num_chars=3))
        return preds

    def save(self, work_dir):
        # your code here
        # copies best model in work_dir
        os.system('cp {work_dir}/*.checkpoint {work_dir}/model.checkpoint')
        return os.path.isfile('{work_dir}/model.checkpoint') 

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        try:
            with open(f'{DATA_PATH}/vocabulary.pkl','rb') as f:
                cls.vocab = pickle.load(f)
            automodel = model.AutoCompleteNet(len(cls.vocab), FEATURE_SIZE)
            automodel.load_model(f'{work_dir}/model.checkpoint')
            automodel.to(device)
    
            return automodel
        except FileNotFoundError:
            print("Trained model.checkpoint is not present")
            #exit()

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        runner = MyModel_Runner()
        print('Loading training data')
        train_data = runner.load_training_data()
        print('Training')
        runner.run_train(train_data, args.work_dir)
        print('Saving model')
        runner.save(args.work_dir)
    elif args.mode == 'test':
        runner = MyModel_Runner()
        print('Loading model')
        model = runner.load(args.work_dir)
        print(model)
        print('Loading test data from {}'.format(args.test_data))
        test_data = runner.load_test_data(args.test_data)
        print(test_data)
        print('Making predictions')
        pred = runner.run_pred(model, test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        runner.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
