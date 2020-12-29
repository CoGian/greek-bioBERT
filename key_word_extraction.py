from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import nltk
import unicodedata
import json


def load_model(model_name):
    model = TFBertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # print(model.summary())
    return model, tokenizer


def produce_embeddings(model, tokenizer, text, isDoc):
    input_ids = tokenizer.encode(text,
                                 return_tensors="tf",
                                 padding="max_length",
                                 max_length=512,
                                 truncation=True)
    outputs = model(input_ids).last_hidden_state
    # doc_embedding = tf.reshape(tf.nn.softmax(outputs, axis=1).numpy()[0], -1).numpy()
    if isDoc:
        text_embedding = tf.reshape(tf.nn.softmax(outputs, axis=1).numpy()[0], -1).numpy().reshape(1, -1)
    else:
        text_embedding = tf.reshape(tf.nn.softmax(outputs, axis=1).numpy()[0], -1).numpy()

    return text_embedding


def prepare_stopwords_list():
    stop_words = nltk.corpus.stopwords.words('greek')
    with open("stopwords-el.json", "r", encoding="utf-8") as fin:
        stop_words2 = json.load(fin)

    stop_words.extend(stop_words2)
    stop_words = set(stop_words)

    return stop_words


def strip_accents_and_lowercase(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').lower()


def produce_candidates(doc, n_gram_range, stop_words):
    # unaccented_doc = strip_accents_and_lowercase(doc)

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names()

    return candidates


def extract_keywords(doc):
    nltk.download('stopwords')

    stop_words = []
    candidates = produce_candidates(doc, (3, 4), stop_words)

    model, tokenizer = load_model("greekBERT")
    doc_embedding = produce_embeddings(model, tokenizer, doc, True)

    candidate_embeddings = []
    for candidate in candidates:
        candidate_embeddings.append(produce_embeddings(model, tokenizer, candidate, False))

    top_n = 5
    similarities = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in similarities.argsort()[0][-top_n:]]
    print(keywords[::-1])
    print(similarities)
    # print(similarities.argsort())
    # print(candidates)


if __name__ == '__main__':
    doc = """η διαπίστωση της ύπαρξης γονιδιακών μεταλλάξεων, αλλά και αυξημένης έκφρασης συγκεκριμένων γονιδίων, κατέδειξε την ετερογένεια των περιπτώσεων της οξείας μυελογενούς λευχαιμίας (ομλ). ειδικότερα, κατά την τελευταία δεκαπενταετία έχει διαπιστωθεί η παρουσία μεταλλάξεων σε αρκετά γονίδια καθώς και αυξημένη έκφραση άλλων γονιδίων . οι μεταλλάξεις και η αυξημένη έκφραση των προαναφερθέντων γονιδίων κατά κύριο λόγο ανιχνεύονται σε ασθενείς με φυσιολογικού καρυοτύπου ομλ. η παρακολούθηση της ελάχιστης υπολειμματικής νόσου (ευν) μετά τη χημειοθεραπεία (χ/θ) εφόδου και ιδιαίτερα μετά τη χ/θ εδραίωσης είναι σημαντική για την αξιολόγηση του κινδύνου υποτροπής της νόσου στον κάθε ασθενή με ομλ ξεχωριστά. η επίμονη προσπάθεια πρώιμης ανίχνευσης της υποτροπής (ανοσοφαινοτυπικής ή μοριακής) στους ασθενείς με ομλ δικαιώνεται από τη δυνατότητα αποτελεσματικότερης αντιμετώπισής της σε σύγκριση με την αιματολογική υποτροπή. η δυνατότητα παρακολούθησης της ευν στην ομλ με τις μοριακές τεχνικές ενισχύθηκε σημαντικά τόσο από τη χρήση της real time quantitative polymerase chain reaction (rq-pcr), όσο και από την ταυτοποίηση ενός σημαντικού αριθμού γονιδίων-«στόχων», τα οποία αναφέρθηκαν προηγουμένως. από την ανασκόπηση των δεδομένων της βιβλιογραφίας προκύπτει το συμπέρασμα ότι η και το αποτελούν τα πλέον υποσχόμενα γονίδια-«στόχους» για την παρακολούθηση της ευν στην ομλ με την τεχνική της rq-pcr, πέρα από τα χιμαιρικά γονίδια."""
    extract_keywords(doc)
