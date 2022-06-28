#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import os
import logging
import benchmark
from utils import helper


class StealthNet:
    def __init__(self, bench, model1, model2, device):
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.train_loader = bench.get_dataloader("Flower102", split='train')
        self.test_loader = bench.get_dataloader("Flower102", split='test')

    def fine_tuning(self):
        pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    args = helper.get_args()
    from benchmark import ImageBenchmark
    bench = ImageBenchmark(
        datasets_dir=os.path.join(args.benchmark_dir, 'data'),
        models_dir=os.path.join(args.benchmark_dir, 'models')
    )
    model1 = None
    model2 = None
    model_strs = []
    for model_wrapper in bench.list_models():
        if not model_wrapper.torch_model_exists():
            continue
        if model_wrapper.__str__() == args.model1:
            model1 = model_wrapper
        if model_wrapper.__str__() == args.model2:
            model2 = model_wrapper
        model_strs.append(model_wrapper.__str__())

    if model1 is None or model2 is None:
        print(f'model not found: {args.model1} {args.model2}')
        print(f'find models in the list:')
        print('\n'.join(model_strs))
        return

    # 规定model1是最source模型，model2是reuse模型
    net = StealthNet(bench, model1=model1, model2=model2, device=args.device)
    net.fine_tuning()



if __name__ == "__main__":
    main()