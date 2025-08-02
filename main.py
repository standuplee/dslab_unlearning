import argparse
import numpy as np
from config import InsParam
from Experiments.oracle_instance import OracleInstance
from Experiments.baseline_instance import BaselineInstance
from config import Instance  # 기존 Instance
from logger_setup import setup_logger, init_wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset name')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--worker', type=int, default=8, help='number of CPU workers')
parser.add_argument('--verbose', type=int, default=2, help='verbose type')
parser.add_argument('--group', type=int, default=10, help='number of groups')
parser.add_argument('--layer', nargs='+', default=[64, 32], help='setting of layers')
parser.add_argument('--learn', type=str, default='sisa', help='type of learning and unlearning')
parser.add_argument('--delper', nargs='+', type=float, default=[0.5], help='deleted user proportion(s)')
parser.add_argument('--deltype', type=str, default='random', help='unlearn data selection')
parser.add_argument('--model', type=str, default='wmf', help='rec model')
parser.add_argument('--mode', type=str, default='instance', help='instance type: oracle | baseline | instance')
parser.add_argument('--methods', nargs='+', help='specific baseline methods to run (baseline mode only)')
parser.add_argument('--origin_model_path', type=str, default='na', help='task vector model')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
parser.add_argument('--beta', type=float, default=0.3, help='beta')
parser.add_argument('--rank_ratio', type=float, default=0.8, help='rank ratio')


def main():
    args = parser.parse_args()
    
    # Logger
    log_path = f"./log/{args.model}_{args.dataset}_{args.learn}_{args.deltype}_{args.delper}.log"
    logger = setup_logger(log_path)
    logger.info("Logger initialized")

    # wandb
    init_wandb(args)

    param = InsParam(
        dataset=args.dataset,
        model=args.model,
        epochs=args.epoch,
        n_worker=args.worker,
        layers=args.layer,
        n_group=args.group,
        del_per=args.delper[0],  # 첫 값 기본 저장
        learn_type=args.learn,
        del_type=args.deltype
    )

    del_per_list = args.delper

    # ---------------- Oracle Instance ---------------- #
    if args.mode == 'oracle':
        logger.info("Running OracleInstance...")
        oracle_ins = OracleInstance(param, del_per_list=del_per_list)
        oracle_ins.run(verbose=args.verbose)

    # ---------------- Baseline Instance ---------------- #
    elif args.mode == 'baseline':
        logger.info("Running BaselineInstance...")
        baseline_ins = BaselineInstance(param, del_per_list=del_per_list)
        baseline_ins.run(verbose=args.verbose, methods=args.methods)

    # ---------------- Default Instance ---------------- #
    elif args.mode == 'instance':
        logger.info("Running default Instance...")
        ins = Instance(param)
        ins.run(verbose=args.verbose)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == '__main__':
    main()