import os
from utils.config import *
from data import load_data
from models import models
import transformation
import random
import pandas as pd
import art
import time
import json
import utils.measure as measure
import gc
import cv2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class MyClass:
    def __init__(self, debug=False):
        _, (self.test_x, self.test_y) = load_data('mnist')
        self.debug = debug
        if not debug:
            self.model_names = os.listdir('data/models')
        else:
            self.model_names = os.listdir('data/models')[:2]
        self.model_models = {}
        print("Loading models")
        for tmp in self.model_names:
            self.model_models[tmp] = models.load_model(tmp)
        print("Finished loading models!")

    def re_trans(self, x, trans_type):
        if trans_type == TRANSFORMATION.rotate90:
            x = transformation.transform_images(x, TRANSFORMATION.rotate270)
        elif trans_type == TRANSFORMATION.rotate270:
            x = transformation.transform_images(x, TRANSFORMATION.rotate90)
        elif trans_type == TRANSFORMATION.rotate180:
            x = transformation.transform_images(x, TRANSFORMATION.rotate180)
        elif trans_type == TRANSFORMATION.flip_horizontal:
            x = transformation.transform_images(x, TRANSFORMATION.flip_horizontal)
        elif trans_type == TRANSFORMATION.flip_both:
            x = transformation.transform_images(x, TRANSFORMATION.flip_both)
        elif trans_type == TRANSFORMATION.flip_vertical:
            x = transformation.transform_images(x, TRANSFORMATION.flip_vertical)
        return x

    def attack(self, model_name, x, attack_params):
        wrap_model = art.classifiers.KerasClassifier(self.model_models[model_name], use_logits=False)
        if attack_params['type'].lower() == 'fgsm':
            attacker = art.attacks.FastGradientMethod(wrap_model, eps=attack_params['eps'],
                                                      batch_size=attack_params['batch_size'])
        else:
            raise NotImplementedError("Only FGSM supported for now!")
        trans_name = model_name.split('.')[0].split('-')[-1]
        trans_x = transformation.transform_images(x, trans_name)
        adv_x = attacker.generate(trans_x)
        return self.re_trans(x=adv_x, trans_type=trans_name)

    def inference(self, model_name, x):
        trans_name = model_name.split('.')[0].split('-')[-1]
        trans_x = transformation.transform_images(x, trans_name)
        return self.model_models[model_name].predict(trans_x)

    def save_to_json(self, dict, path, name):
        tmp = json.dumps(dict)
        with open(os.path.join(path, name + '.json'), 'w+') as fin:
            fin.write(tmp)
        print("Saved {} to {}!".format(name, path))

    def evaluation_transferability_single(self, i, attack_params, path):
        cur_x = np.expand_dims(self.test_x[i], axis=0)
        transferability_record = {}
        dissimilarity_record = {}
        for model_name in self.model_names:
            adv_x = self.attack(model_name, cur_x, attack_params)
            adv_x = np.clip(adv_x, 0, 1)
            for cand_name in self.model_names:
                if cand_name == model_name:
                    pass
                pred = self.inference(cand_name, adv_x)
                if pred.argmax(axis=-1) != self.test_y.argmax(axis=-1)[i]:
                    try:
                        transferability_record['number of models fooled by {}'.format(model_name)] += 1
                    except KeyError:
                        transferability_record['number of models fooled by {}'.format(model_name)] = 1
            dissimilarity_record[model_name] = measure.frobenius_norm(adv_x, cur_x)
        self.save_to_json(transferability_record, path, 'transferability record for {}th example'.format(i))
        self.save_to_json(dissimilarity_record, path, 'dissimilarity record for {}th example'.format(i))
        print("Finished Evaluating {}th example!".format(i))

    def evaluate_transferability(self, attack_params, path):
        if not os.path.exists(path):
            os.mkdir(os.path.join(os.curdir, path))
        if self.debug:
            cnt = 10
        else:
            cnt = self.test_x.shape[0]
        for i in range(cnt):
            print("Evaluating {}th example".format(i))
            self.evaluation_transferability_single(i, attack_params, path)
        print('Done evaluating!')

def main():
    params = {
        'type': 'fgsm',
        'eps': 0.1,
        'batch_size': 128
    }
    model_class = MyClass(debug=False)
    model_class.evaluate_transferability(params, 'data/transferability_test')

if __name__ == '__main__':
    main()