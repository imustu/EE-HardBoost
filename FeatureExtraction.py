# -*- coding: utf-8 -*-
# @Time : 2021/7/29 9:12
# @Author : chenshuai
# @File : FeatureExtraction.py
import os
import openpyxl
import pandas as pd

# kflodDataAddress="D:\multiLabelClassification\dataSet\kflod\\"
# kflodFeatureAddress="D:\multiLabelClassification\dataSet\\kflodFeatures\\"
trainSetAddress ="D:\pycharm\PycharmProjects\protein_sequence\\out_all_sequence31.fasta"
# testSetAddress="D:\multiLabelClassification\dataSet\\testSet.fasta"
trainFeatureAddress = "D:\pycharm\PycharmProjects\protein_sequence\\trainFeatures\\"


features = ["AAC", "CTriad", "DPC", "QSOrder", "GTPC"]


def command(dataAddress, feature, featureAddress):
	"""
	通过cmd命令执行脚本文件提取特征
	:param dataAddress: 输入：数据集地址
	:param feature: 输入：特征
	:param featureAddress: 输出：特征向量地址
	:return: 提取特征的命令
	"""

	return "python D:/python/iLearn-master/iLearn-protein-basic.py --file " + dataAddress + " --method " + feature + "  " \
																													 "--format svm --out " + featureAddress + feature + ".txt"


def execute_cmd_command(command):
	return os.popen(command).read()

def featureExtraction(dataAddress, features, featureAddress):
	"""
	提取特征
	:param dataAddress: 输入：数据集地址
	:param features: 输入：特征
	:param featureAddress: 输出：特征集保存地址
	"""

	for feature in features:
		result = execute_cmd_command(command(dataAddress, feature, featureAddress))
		print(result)

def writeinexcel( ):
	txtname = 'GAAC.txt'
	excelname = 'GAAC.xlsx'

	fopen = open(txtname, 'r', encoding='utf-8')
	lines = fopen.readlines()
	# 写入 excel表
	file = openpyxl.Workbook()
	sheet = file.active
	# 新建一个sheet
	sheet.title = "data"

	i = 0
	for line in lines:
		# strip 移出字符串头尾的换行
		line = line.strip('\n')
		# 用','替换掉'\t',很多行都有这个问题，导致不能正确把各个特征值分开
		#line = line.replace("\t", ",")
		line = line.split()
		# 一共7个字段
		for index in range(len(line)):
			a = line[index]
			a = a.split(':')[-1]
			sheet.cell(i + 1, index + 1, a)
		# 行数递增
		i = i + 1



	file.save(excelname)
def integrate():
	x = pd.read_excel('A_network_features.xlsx')
	y = pd.read_excel('A_sequence_features.xlsx')
	A_features = pd.merge(x, y, on="Protein")
	A_features.to_excel('A_features.xlsx' , index=False)





if __name__ == '__main__':


	feature = ["AAC"]
	print(type(trainSetAddress))
	# kflodData_process(kflod2Address,feature,kflod2FeatureAddress)
	featureExtraction(trainSetAddress, feature, trainFeatureAddress)
	# featureExtraction(demoAddress,feature,demoFeaturesAddress)
	#writeinexcel()
	#integrate()



