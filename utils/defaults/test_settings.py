import argparse
import sys
import os
def test_parse_args():
    #获取测试数据参数
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

    #input_size
    parser.add_argument('--input_size', type=tuple, default=[488,488],
                        help='input_size default(3,244,244)')
    #测试数据路径
    parser.add_argument('--test_dir', type=str, default='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train/img2test',
                        help='test_data dir')
    #模型文件
    parser.add_argument('--model_path', type=str,default='/home/gukai/research/Models-based-on-yolo/models/reg.yaml',
                        help='model config-file')

    #模型权重文件保存地址
    parser.add_argument('--weights_path', type=str,default='/home/gukai/research/Models-based-on-yolo/output/210809/last.pt',
                    help='model weights-file')
    #输出地址
    parser.add_argument('--output_dir', type=str,default='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train/pred_label2',
                        help='output_dir of predicted label')
    #是否要预测结果与原label对比求loss
    parser.add_argument('--val', action='store_true',
                        help='whether to validate')
    #如果要对比，原label的路径
    parser.add_argument('--label_dir', type=str,default='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train/label3',
                    help='original label dir')
    #显卡还是cpu，哪几张卡                
    parser.add_argument('--device', default='4', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args=config_parser.parse_args()
    return args