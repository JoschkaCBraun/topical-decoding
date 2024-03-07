import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

N_TOPIC_WORDS = 20


# returns a Pandas DataFrame of the NEWTS training set
def read_train(path="NEWTS/NEWTS_train_2400.csv"):
    out = pd.read_csv(path, encoding="utf-8", index_col=[0])
    assert len(out) == 2400
    return out

# given a path to model, loads it and its dictionary
def read_lda(path="LDA_250/"):
    path = path.strip("/") + "/"
    lda = LdaModel.load(path + "lda.model", mmap="r")
    print("Loaded the LDA model")
    dictionary = Dictionary.load(path + "dictionary.dic", mmap="r")
    print("Loaded dictionary")
    return lda, dictionary

# get the top words for a given topic from the LDA model
def get_topic_words(lda, topic_id, n_topic_words=N_TOPIC_WORDS):
    """
    Returns the top words for a given topic from the LDA model.
    
    :param lda: The LDA model.
    :param topic_id: The topic number to get the top words for.
    :param num_words: The number of top words to return.
    :return: A list of top words for the specified topic.
    """
    try:
        # Get the specified topic. Note: num_words here limits the number of words returned for the topic.
        topic_words = lda.show_topic(topic_id, n_topic_words)
        
        # Extract just the words
        top_words = [word for word, prob in topic_words]
        return top_words
    except Exception as e:
        print(f"Error in getting top topic words: {e}")
        return []


def main():
    # Load the training set and the LDA model
    train = read_train()
    lda, dictionary = read_lda()

    # create list of dictionaries (document and topic words)
    doc_topic_list = []
    for i in range(len(train)):
        document = train.iloc[i]['article']
        tid1 = train.iloc[i]['tid1']
        tid2 = train.iloc[i]['tid2']

        for topic_id in [tid1, tid2]:
            topic_words = [get_topic_words(lda, topic_id) for topic_id in lda.get_document_topics(dictionary.doc2bow(document), minimum_probability=0.0)]
            doc_topic_dict = {"document": document, "topic_words": topic_words}
            doc_topic_list.append(doc_topic_dict)

    return doc_topic_list

if __name__ == "__main__":
    main()