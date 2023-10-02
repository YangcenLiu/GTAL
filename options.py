import argparse

parser = argparse.ArgumentParser(description='DELU')
parser.add_argument('--path_dataset', type=str, default='path/to/Thumos14', help='the path of data feature')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model_name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature_size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num_class', type=int, default=20, help='number of classes (default: )')
parser.add_argument('--dataset_name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--max_seqlen', type=int, default=320,
                    help='maximum sequence length during training (default: 750)')
parser.add_argument('--num_similar', default=3, type=int,
                    help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--seed', type=int, default=3552, help='random seed (default: 1)')
parser.add_argument('--max_iter', type=int, default=20000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--feature_type', type=str, default='I3D',
                    help='type of feature to be used I3D or UNT (default: I3D)')
parser.add_argument('--use_model', type=str, default='DELU', help='model used to train the network')
parser.add_argument('--eval_interval', type=int, default=200, help='time interval of performing the test')
parser.add_argument('--similar_size', type=int, default=2)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dataset', type=str, default='SampleDataset')
parser.add_argument('--proposal_method', type=str, default='multiple_threshold_hamnet')

parser.add_argument('--work_dir', type=str, default='work_dir') # to save all results
parser.add_argument('--ckpt_path', type=str, default='ckpt/best_delu_thumos.pkl') # for test and ood_test only

# for proposal genration
parser.add_argument('--scale', type=float, default=1)
parser.add_argument("--feature_fps", type=int, default=25)
parser.add_argument('--gamma_oic', type=float, default=0.2)

parser.add_argument('--k', type=float, default=7)
# for testing time usage
parser.add_argument("--topk2", type=float, default=10)
parser.add_argument("--topk", type=float, default=60)

parser.add_argument('--dropout_ratio', type=float, default=0.7)
parser.add_argument('--reduce_ratio', type=int, default=16)
# for pooling kernel size calculate
parser.add_argument('--t', type=int, default=5)

# -------------loss weight---------------
parser.add_argument("--alpha1", type=float, default=0.8)
parser.add_argument("--alpha2", type=float, default=0.8)  # origin guide_loss coefficient
parser.add_argument("--alpha3", type=float, default=1)
parser.add_argument('--alpha4', type=float, default=1)

parser.add_argument("--AWM", type=str, default='BWA_fusion_dropout_feat_v2')

parser.add_argument("--without_wandb", action='store_true')
parser.add_argument("--alpha_edl", type=float, default=1)
parser.add_argument("--rat_atn", type=int, default=9)
parser.add_argument("--amplitude", type=float, default=0.7)
parser.add_argument("--alpha_uct_guide", type=float, default=0.4)

# -------------ddgnet---------------
parser.add_argument('--alpha5', type=float, default=1)  # feature consistency
parser.add_argument('--alpha6', type=float, default=1)  # complementary learning
parser.add_argument('--weight', type=float, default=2)
parser.add_argument('--temperature', type=float, default=0.5)

parser.add_argument('--action_threshold', type=float, default=0.5)
parser.add_argument('--background_threshold', type=float, default=0.5)
parser.add_argument('--similarity_threshold', type=float, default=0.8)
parser.add_argument("--top_k_rat", type=int, default=10)

# -------------adv attack---------------
parser.add_argument("--adv_attack", action="store_true", default=False)
parser.add_argument("--epsilon", type=float, default=1e-5, help="epsilon for attack")
parser.add_argument("--num_pgd_iter", type=int, default=40)

# random resize
parser.add_argument("--random_resize", action="store_true", default=False)
parser.add_argument("--resize_min_ratio", type=float, default=0.5)
parser.add_argument("--resize_max_ratio", type=float, default=1.5)

# -------------ood test-----------------
parser.add_argument("--model_num_classes", type=int, default=20)
parser.add_argument("--class_mapping", type=str, default="class_mapping/t2a_class_mapping.json")

# -------------multi scale setting-----------------
parser.add_argument("--scales", nargs='+', type=int, default=[1])


# -------------sub_class setting-----------------
parser.add_argument("--num_sub_class", type=int, default=50) # description="total number of visual concepts"
parser.add_argument("--num_memory_bank_size", type=int, default=100) # maximum number of snippet feature memorized for a sub_class
