# decomposition_PCA_SADE
先采用PCA，然后采用SADE（堆叠式自编码）进行降维


## 文件目录结构

- data

    -- rawdata.csv   # 原始的一维数据集（注：只是作为样本数据，实际使用中本身就是二维度矩阵形式的数据，不需要进行转换。）
    
    -- matrix.txt    # 基于一维数据转换的矩阵形式
    
- network

    -- sade_network.py  # 堆叠式自编码器降维的神经网络模型架构

- preprocess

    -- data.py  # 将1维度数据转换为矩阵
    
    -- pca.py   # PCA降维
    
    -- preprocess_for_sade.py  # 堆叠式自编码器模型数据的预处理

-sade.py  # 堆叠式自编码器模型构建

-main.py  # 程序运行接口，通过它实现PCA和自编码降维


## 程序实现方式

1. 查看程序帮助信息

```py
python main.py -h
```

2. 执行代码：具体参数可以自行调整

```sh
python .\main.py -f .\data\rawdata.csv -n 100 --encoder_hidden [64,32] --decoder_hidden [32,64] -e 10 -b 32 --lr 0.01
```

## 结果

代码执行完毕后，会生成一个 Result 文件目录：

--matrix_pca.txt ：PCA降维后，得到的矩阵

--SADE_out.txt ：自编码器降维后，得到的结果

重构数据与原数据的差异指标为MSE，会在模型train的过程中输出，选择最后一个epoch的就行。
