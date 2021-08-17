import argparse
import sys
import os
def _parse_args():
    #获取训练的一些数据信息之类的
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    #有没有config文件
    # parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
    #                     help='YAML config file specifying default arguments')
    #input_size
    parser.add_argument('--model_path', type=str, default='/home/gukai/research/Models-based-on-yolo/models/Homography2.yaml',
                        help='model config file)')
    parser.add_argument('--input_size', type=tuple, default=[488,488],
                        help='input_size default(3,244,244)')
    #batch_size
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    #训练路径
    parser.add_argument('--train_dir', type=str, default='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train',
                        help='train_data dir')
    #数据文件夹名
    parser.add_argument('--data_dir', type=str, default='un_rectify',
                        help='name of data_folder')
    #label文件夹名(可以与数据文件夹名相同)
    parser.add_argument('--label_dir', type=str, default='label3',
                        help='name of label_folder')
    #验证路径
    parser.add_argument('--val_dir', type=str, default='/fast_data2/India/TranData/RussiaBiscuits/RussiaBiscuits_wft_0715',
                        help='val_dir')
    #训练epochs
    parser.add_argument('--epochs', type=int, default=10000,
                        help='epochs')
    #模型文件
    parser.add_argument('--config_file', type=str,default='models/Homography.yaml',
                        help='model config-file')

    #输出地址
    parser.add_argument('--output_dir', type=str,default='output/210809',
                        help='output_dir to save model')
                    
    #几张卡
    parser.add_argument('--num_gpus', type=int,default=2,
                        help='gpus used for training')

    #几台服务器
    parser.add_argument('--num_machines', type=int,default=1,
                        help='machines used for training')

    #第几台机器作为主机
    parser.add_argument('--machine_rank', type=int,default=0,
                        help='machine index as main')
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    #用于卡之间交流的一个设置
    parser.add_argument('--dist_url',
        #default='auto',
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",)

    #是否继续上次训练
    parser.add_argument('--resume', action='store_true',default=False,
                        help='whether to resume')

    args=config_parser.parse_args()
    return args