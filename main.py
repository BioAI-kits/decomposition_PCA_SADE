import argparse
from preprocess.data import split_data_to_matrix
from preprocess.pca import decomposition_pca
from sdae import SadeTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--raw_file', help="原始的一维度数据集")
    parser.add_argument('-n', '--n_components', type=int, help="PCA保留维度数目")
    parser.add_argument('--encoder_hidden', type=list, default=[64, 32] ,help="SADE编码器隐含层神经元")
    parser.add_argument('--decoder_hidden', type=list, default=[32, 64] ,help="SADE解码器隐含层神经元")
    parser.add_argument('-e', '--epoch', type=int, default=100, help="SADE模型训练的Epoch数")
    parser.add_argument('-b', '--batchsize', type=int, default=32, help="SADE模型训练的BatchSize数")
    parser.add_argument('--lr', type=float, default=0.001, help="SADE模型训练的学习率lr")
    args = parser.parse_args()
    args = vars(args)
    
    # step1：构建矩阵类型数据
    data = split_data_to_matrix(file_path=args['raw_file'], dimen_len=500, save_file="./result/matrix.txt") # dimen_len指的是将一维数据拆分为矩阵，矩阵列数为500
    
    # step2：执行PCA降维
    data_pca = decomposition_pca(data=data, n_components=args['n_components'], save_file='./result/matrix_pca.txt')

    # step3：执行SADE降维
    trainer = SadeTrainer(in_features=args['n_components'], out_features=args['n_components'], 
                          encoder_hidden=args['encoder_hidden'], decoder_hidden=args['decoder_hidden'], 
                          epoch=args['epoch'], batch_size=args['batchsize'],lr=args['lr'],
                          sade_output_file="./result/SADE_out.txt"
                          )
    trainer.start_train()
    trainer.save_data()


if __name__ == "__main__":
    main()