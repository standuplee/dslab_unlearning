import argparse
import os
from config import InsParam
from logger_setup import setup_logger, init_wandb
from Experiments.instance import Instance
from Experiments.task_instance import task_instance, task_instance2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--worker', type=int, default=8, help='number of CPU workers')
    parser.add_argument('--verbose', type=int, default=2, help='verbose type')
    parser.add_argument('--group', type=int, default=10, help='number of groups')
    parser.add_argument('--layer', nargs='+', type=int, default=[64, 32], help='setting of layers')
    parser.add_argument('--learn_type', type=str, default='instance', 
                        choices=['instance', 'task_vector', 'joint_task_vector'], 
                        help='type of learning/unlearning')
    parser.add_argument('--learn', type=str, default='retrain', help='algorithm name (retrain, sisa, ultrare, recformer)')
    parser.add_argument('--delper', type=float, default=0.5, help='deleted user proportion')
    parser.add_argument('--deltype', type=str, default='random', help='unlearn data selection')
    parser.add_argument('--model', type=str, default='wmf', help='rec model')
    parser.add_argument('--origin_model_path', type=str, default='na', help='task vector model')
    parser.add_argument('--oracle_model_path', type=str, default='na', help='oracle model')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--beta', type=float, default=0.3, help='beta')
    parser.add_argument('--rank_ratio', type=float, default=0.8, help='rank ratio')
    parser.add_argument('--use_degree_weighting', type=bool, default=False, help='degree weighting for SVD')
    parser.add_argument('--degree_weight_power', type=float, default=0.5)
    parser.add_argument('--normalize_weights', type=bool, default=False)

    args = parser.parse_args()

    # Logger & Wandb
    log_path = f"./log/{args.model}_{args.dataset}_{args.learn_type}_{args.learn}_{args.deltype}_{args.delper}.log"
    os.makedirs("./log", exist_ok=True)
    logger = setup_logger(log_path)
    logger.info("Logger initialized")
    init_wandb(args)

    # 파라미터 세팅
    param = InsParam(args.dataset, args.model, args.epoch, args.worker, args.layer,
                     args.group, args.delper, args.learn, args.deltype,
                     args.origin_model_path, args.oracle_model_path,
                     args.alpha, args.beta, args.rank_ratio,
                     args.use_degree_weighting, args.degree_weight_power, args.normalize_weights)

    # learn_type에 따라 실행 클래스 선택
    if args.learn_type == 'instance':
        runner = Instance(param)
    elif args.learn_type == 'task_vector':
        runner = task_instance(param)
    elif args.learn_type == 'joint_task_vector':
        runner = task_instance2(param)
    else:
        raise ValueError(f"Unknown learn_type: {args.learn_type}")

    logger.info(f"Starting experiment: dataset={args.dataset}, model={args.model}, learn_type={args.learn_type}, learn={args.learn}")
    runner.run(verbose=args.verbose)
    logger.info("Experiment finished")

if __name__ == '__main__':
    main()