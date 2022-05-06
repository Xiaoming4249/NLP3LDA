import math
import jieba
import os
import random
import gensim
from gensim import models
"""
NLP第三次作业：
    任务：
    1.对语料库所给的16本小说分别进行lda建模
    2.从所给语料库（16本金庸小说）中均匀抽取200个段落，使用上面得到的lda模型计算每个段落的主题分布。
    3.根据段落主题分布进行分类，并验证结果的准确性。
"""

class Article:
    lable = ''
    wordList = []
    topicDistribute = []

    def __init__(self, name, wordList):
        self.lable = name
        self.wordList = wordList

    def setTopicDist(self, topicDist):
        self.topicDistribute = topicDist

def readNovels(path):
    # 16篇小说，即16个文章
    articles = []
    names = os.listdir(path)
    for name in names:
        novelPath = path + '\\' + name
        with open(novelPath, 'r', encoding='UTF-8') as f:
            # 对文本字符串过滤后，使用结巴分词得到该小说的分词列表
            words = list(jieba.lcut(filter(f.read())))
            articles.append(Article(name.replace('.txt', ''), words))
        f.close()
    return articles

# 读取语料内容，并完成预处理
def readTestNovels(path):
    # 208个段落，即208个Article
    articles = []
    names = os.listdir(path)
    for name in names:
        novelPath = path + '\\' + name
        with open(novelPath, 'r', encoding='UTF-8') as f:
            # 对文本字符串过滤后，使用结巴分词得到该小说的分词列表
            words = list(jieba.lcut(filter(f.read())))
            # 每篇小说等间隔选取600个词，作为一篇文章，由于每篇小说包含作者的前言和结语，
            # 与小说内容无关，因此抛弃第一分段和最后一分段
            step = int(len(words) // 15)
            for i in range(1, 14):
                articles.append(Article(name.replace('.txt', ''), words[i*step: i*step + 600]))
        f.close()
    return articles


# 语料清理，去除无用的标点符号
def filter(novel):
    strs = ['。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”',
          '‘', '’', '［', '］', '....', '......', '『', '』', '（', '）', '…', '「', '」',
          '＜', '＞', '+', '【', '】', '(', '.', '?', 'com', 'cr173', ')', 'www']
    for str in strs:
        novel = novel.replace(str, '')
    return novel


if __name__ == '__main__':
    # 获取语料库
    print("获取训练集语料库中...")
    articles = readNovels('.//novels')
    wordsDocuments = [article.wordList for article in articles]
    # print(wordsDocuments)

    # 根据语料词语列表，构建DT矩阵（词频向量），作为lda模型建立的输入
    from gensim import corpora
    dictionary = corpora.Dictionary(wordsDocuments)
    docTermMatrix = [dictionary.doc2bow(words) for words in wordsDocuments]
    print("训练集DT矩阵构建完成")
    #print(docTermMatrix)

    # 建立lda模型
    Lda = models.ldamodel.LdaModel
    # 设置topic个数为40
    ldamodel = Lda(docTermMatrix, num_topics= 40, id2word=dictionary)
    # 打印40个topic和其中10个词出现的概率
    print(ldamodel.print_topics(num_topics=40, num_words=10))
    # 打印16篇文章的主题分布，并保存在novelDistribute中
    novelDistribute = []
    i = 0
    for doc in docTermMatrix:
        doclda = ldamodel[doc]
        ad = [0.0] * 40
        for t in doclda:
            ad[t[0]] = t[1]
        novelDistribute.append(ad)
        i += 1
        print("第%d篇文章主题分布：" % i, doclda)
    print(novelDistribute)
    # 获取语料库
    print("获取测试集语料库中...")
    articlesTest = readTestNovels('.//novels')
    wordsDocumentsTest = [article.wordList for article in articlesTest]
    # print(wordsDocuments)

    # 根据语料词语列表，构建DT矩阵（词频向量），作为lda模型建立的输入
    docTermMatrixTest = [dictionary.doc2bow(words) for words in wordsDocumentsTest]
    print("测试集DT矩阵建立完成")
    # print(docTermMatrixTest)

    # 将DT矩阵作为输入，根据训练集建立的模型，输出测试集文章的主题分布
    # 打印16篇文章的主题分布，并保存在novelDistribute中
    novelTestDistribute = []
    i = 0
    for doc in docTermMatrixTest:
        doclda = ldamodel[doc]
        ad = [0.0] * 40
        for t in doclda:
            ad[t[0]] = t[1]
        novelTestDistribute.append(ad)
        i += 1
        print("第%d篇测试文章主题分布：" % i, doclda)
    print(novelTestDistribute)

    # 计算分布间距离，判断段落类型
    for k in range(208):
        dis = novelTestDistribute[k]
        deta = [0.0]*16
        for i in range(16):
            for j in range(40):
                deta[i] += (math.sqrt(novelDistribute[i][j]) - math.sqrt(dis[j])) * (math.sqrt(novelDistribute[i][j]) - math.sqrt(dis[j]))
        novelTestDistribute[k].append(deta.index(min(deta)))
    # 正确率计算
    count = 0
    for i in range(208):
        if (novelTestDistribute[i][40]+1)*13 > i:
            count += 1
    print("正确率为%f" % (count/208))

    # 将结果保存在excel表格中
    import xlwt
    wbk = xlwt.Workbook()
    sheet1 = wbk.add_sheet('训练集分布')
    for i in range(16):
        for j in range(40):
            sheet1.write(i, j, "%f" % novelDistribute[i][j])
    sheet2 = wbk.add_sheet('测试集分布')
    for i in range(208):
        for j in range(41):
            sheet2.write(i, j, "%f" % novelTestDistribute[i][j])
    wbk.save('result.xls')

    # 分类结果散点图绘制
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("测试集段落分类结果统计", fontsize=24)
    plt.xlabel("段落序号", fontsize=14)
    plt.ylabel("分类结果（小说序号）", fontsize=14)
    plt.axis([0, 210, 0, 17])
    # classresult = []
    for i in range(0,208):
        # classresult[i] = novelTestDistribute[i][40]+1
        if (novelTestDistribute[i][40]+1)*13 > i:
            plt.scatter(i + 1, novelTestDistribute[i][40]+1, c='green', s=20)
        else:
            plt.scatter(i + 1, novelTestDistribute[i][40] + 1, c='red', s=20)
    plt.show()
