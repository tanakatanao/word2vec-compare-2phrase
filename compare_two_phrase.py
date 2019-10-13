import gensim
import numpy as np
import MeCab
from scipy import spatial
mecab = MeCab.Tagger("/usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati")

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('model.vec', binary=False)

def main():
    result = sentence_similarity(
        "彼は昨日、激辛ラーメンを食べてお腹を壊した",
        "昨日、僕も激辛の中華料理を食べてお腹を壊した"
    )
    print(result)

    result = sentence_similarity(
        "ソースコード管理問題",
        "gitトラブル"
    )
    print(result)

    result = sentence_similarity(
        "デプロイ問題",
        "デプロイ"
    )
    print(result)

    result = sentence_similarity(
        "もっと英語で話すべき",
        "音楽がならない"
    )
    print(result)

def extractAndShape(sentence):
    node = mecab.parseToNode(sentence)
    keywords = []
    while node:
        if node.feature.split(",")[0] == u"名詞":
            keywords.append(node.surface)
        elif node.feature.split(",")[0] == u"形容詞":
            keywords.append(node.feature.split(",")[6])
        elif node.feature.split(",")[0] == u"動詞":
            keywords.append(node.feature.split(",")[6])
        node = node.next
    return [s.replace(' \n', '') for s in keywords]

def avg_feature_vector(sentence, model, num_features):
    words = extractAndShape(sentence)
    feature_vec = np.zeros((num_features,), dtype="float32") # 特徴ベクトルの入れ物を初期化
    print(words)
    for word in words:
        feature_vec = np.add(feature_vec, model[word])
    if len(words) > 0:
        feature_vec = np.divide(feature_vec, len(words))
    return feature_vec

def sentence_similarity(sentence_1, sentence_2):
    # 今回使うWord2Vecのモデルは300次元の特徴ベクトルで生成されているので、num_featuresも300に指定
    num_features=300
    sentence_1_avg_vector = avg_feature_vector(sentence_1, word2vec_model, num_features)
    sentence_2_avg_vector = avg_feature_vector(sentence_2, word2vec_model, num_features)
    # １からベクトル間の距離を引いてあげることで、コサイン類似度を計算
    return 1 - spatial.distance.cosine(sentence_1_avg_vector, sentence_2_avg_vector)

if __name__ == "__main__":
    main()