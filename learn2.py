import math
import jieba
import os  # 用于处理文件路径
import random
import numpy as np

"""
NLP第三次作业：
    任务：
    1.从所给语料库（16本金庸小说）中均匀抽取200个段落，每个段落不少于500词，建立LDA主题模型，得到每个段落的主题分布。
    2.根据段落主题分布进行分类，并验证结果的准确性。
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


# 读取语料内容，并完成预处理
def readNovels(path):
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
          '＜', '＞', '+', '【', '】']
    for str in strs:
        novel = novel.replace(str, '')
    return novel

# 建立一个m*n的列表
def listBuilder(m, n):
    if m == 1:
        return [0] * n
    aList = [[0 for i in range(n)] for j in range(m)]
    return aList

# 迭代过程随机投掷模型
def setNewTopic(p):
    for i in range(1, len(p)):
        p[i] += p[i - 1]
    u = random.random()*p[len(p) - 1]
    index = len(p) - 1
    for i in range(len(p)):
        if p[i] > u:
            index = i
            break
    return index

# 模型初始参数
K = 20
alpha = 0.5
beta = 0.5
iter_times = 200
# top_words_num = 1000

if __name__ == '__main__':
    # 获取语料库
    print("获取语料库中...")
    articles = readNovels('.//novels')

    # 申请统计量
    # 词汇表，包含词汇和其id
    VocabularyWI = {}
    VocabularyIW = {}
    # 词汇表初始化
    print("词汇表初始化...")
    wordIndex = 0
    for article in articles:
        for word in article.wordList:
            if VocabularyWI.get(word) == None :
                VocabularyWI[word] = wordIndex
                VocabularyIW[wordIndex] = word
                wordIndex += 1

    # 每个词在某主题中出现的次数，V*K，V表示词语总数，K表示主体总数
    phi = listBuilder(len(VocabularyWI), K)
    # 对于所有词语，每种主题出现的总个数，K
    topicSum = listBuilder(1, K)
    # 第i篇文章j主题词出现的次数，M*K，M表示文章总数（208）
    artTopicSum = listBuilder(len(articles), K)
    # 文章词语数总计
    artWordSum = listBuilder(1, len(articles))
    # 每个文章里每个词被指定的主题序号，M*每篇文章词汇数
    wordTopic = []

    # LDA模型初始化
    print("lda初始化...")
    i = 0
    for article in articles:
        aWord = []
        for word in article.wordList:
            topic = random.randint(0, K - 1)
            aWord.append(topic)
            print("%d,%d"%(VocabularyWI[word], topic))
            phi[VocabularyWI[word]][topic] += 1
            topicSum[topic] += 1
            artTopicSum[i][topic] += 1
            artWordSum[i] += 1
        i += 1
        wordTopic.append(aWord)

    # Collasped Gibbs Sampling 迭代
    # -1-->采样公式重新分配-->+1
    print("开始迭代...")
    V = len(VocabularyWI)
    for iter in range(iter_times):
        for i in range(len(articles)):
            for j in range(len(articles[i].wordList)):
                word = articles[i].wordList[j]
                wordId = VocabularyWI[word]
                topic = wordTopic[i][j]
                phi[wordId][topic] -= 1
                topicSum[topic] -= 1
                artTopicSum[i][topic] -= 1
                p = [1.0] * K
                for k in range(K):
                    p[k] = (((phi[wordId][k] + beta)/(topicSum[k] + V*beta))*
                            ((artTopicSum[i][k] + alpha) / (artWordSum[i] + K*alpha)))
                newTopic = setNewTopic(p)
                phi[wordId][newTopic] += 1
                topicSum[newTopic] += 1
                artTopicSum[i][newTopic] += 1
        print("迭代了%d次..." % iter)

    # 迭代完成，计算各文章的主题分布
    theta = [[1.0 for i in range(K)] for j in range(len(articles))]
    for i in range(208):
        for j in range(K):
            theta[i][j] = (artTopicSum[i][j] + alpha) / (artWordSum[i] + K*alpha)

    # 计算各文章之间主题概率分布的差异距离，并进行分类
    deta = [[1.0 for i in range(len(articles))] for j in range(len(articles))]
    for i in range(208):
        for j in range(208):
            sum = 0.0
            for k in range(K):
                sum += (math.sqrt(theta[i][k]) - math.sqrt(theta[j][k])) * (math.sqrt(theta[i][k]) - math.sqrt(theta[j][k]))
    count = 0
    flag = [1]*208
    result = []
    for i in range(208):
        flag[i] = 0
        temp = [] + deta[i]
        for j in range(208):
            if flag[j] == 0:
                temp[j] = 10000.0
        temp.sort()
        value = temp[12]
        aResult = []
        for j in range(208):
            if flag[j] == 1 and deta[i][j] <= value:
                aResult.append(j)
                flag[j] = 0
        result.append(aResult)
    print(deta)
    print(result)







