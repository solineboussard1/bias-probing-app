import numpy as np
import json
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Gensim libraries for coherence computation
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return ' '.join([t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2])

def extract_topics(responses, candidate_topics=range(5, 12,2)):
    # Clean texts and prepare tokenized texts for coherence computation
    texts = [clean_text(res['text']) for res in responses]
    tokenized_texts = [text.split() for text in texts]  
    
    # Build a dictionary and corpus for the coherence model
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # Vectorize texts for LDA using scikit-learn
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english', max_features=1000)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    if doc_term_matrix.shape[0] == 0 or doc_term_matrix.shape[1] == 0:
        return { "error": "Empty document-term matrix." }
    
    feature_names = vectorizer.get_feature_names_out()
    
    best_coherence = -1
    best_model = None
    best_n_topics = None
    best_doc_topics = None
    best_topics_full = None
    
    # Grid search for the best number of topics based on coherence
    for n_topics in candidate_topics:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=50, learning_method='batch')
        doc_topics = lda.fit_transform(doc_term_matrix)
        
        topics = []
        for i, topic in enumerate(lda.components_):
            top_word_indices = topic.argsort()[:-11:-1] 
            top_words = [feature_names[idx] for idx in top_word_indices]
            topics.append(top_words)
        
        coherence_model = CoherenceModel(topics=topics, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        
        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_model = lda
            best_n_topics = n_topics
            best_doc_topics = doc_topics
            
            topics_full = []
            for i, topic in enumerate(lda.components_):
                top_word_indices = topic.argsort()[:-11:-1]
                top_words = [feature_names[idx] for idx in top_word_indices]
                topic_weights = topic[top_word_indices].tolist()
                total_weight = sum(topic_weights)
                if total_weight > 0:
                    topic_weights = [w / total_weight for w in topic_weights]
                topics_full.append({ "topic_id": i, "words": top_words, "weights": topic_weights })
            best_topics_full = topics_full

    print("Selected number of topics:", best_n_topics, "with coherence:", best_coherence, file=sys.stderr)

    # Track demographic distributions.
    demographics_data = {}
    for i, res in enumerate(responses):
        for demo in res.get("demographics", []):
            category = demo.get("category", "").lower()
            value = demo.get("value", "").lower()
            if category and value:
                demographics_data.setdefault(category, {}).setdefault(value, []).append(best_doc_topics[i])
                
    demographic_distributions = {
        category: {value: np.mean(arr, axis=0).tolist() for value, arr in values.items()}
        for category, values in demographics_data.items()
    }
    print("Demographic distributions:", demographic_distributions, file=sys.stderr)
    
    return { 
        "topics": best_topics_full, 
        "distributions": best_doc_topics.tolist(), 
        "demographicDistributions": demographic_distributions,
    }

if __name__ == "__main__":
    try:
        input_data = json.loads(sys.stdin.read())
        result = extract_topics(input_data)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
