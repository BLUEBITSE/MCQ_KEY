import string
import random
from summarizer import Summarizer

class MCQGenerator:
    def __init__(self):
        self.summarizer = None
        self.spacy = None
        self.re = None
        self.random = None
        self.csv = None
        self.sent_tokenize = None
        self.stopwords = None
        self.max_similarity = None
        self.adapted_lesk = None
        self.wordnet = None
        self.requests = None
        self.string = string

    def get_summarizer(self):
        if self.summarizer is None:
            self.summarizer = Summarizer()
        return self.summarizer

    def get_string(self):
        return self.string

    def get_spacy(self):
        if self.spacy is None:
            import spacy
            self.spacy = spacy
        return self.spacy

    def get_re(self):
        if self.re is None:
            import re
            self.re = re
        return self.re

    def get_random(self):
        if self.random is None:
            self.random = random
        return self.random

    def get_csv(self):
        if self.csv is None:
            import csv
            self.csv = csv
        return self.csv

    def get_sent_tokenize(self):
        if self.sent_tokenize is None:
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize = sent_tokenize
        return self.sent_tokenize

    def get_stopwords(self):
        if self.stopwords is None:
            from nltk.corpus import stopwords
            self.stopwords = stopwords
        return self.stopwords

    def get_max_similarity(self):
        if self.max_similarity is None:
            from pywsd.similarity import max_similarity
            self.max_similarity = max_similarity
        return self.max_similarity

    def get_adapted_lesk(self):
        if self.adapted_lesk is None:
            from pywsd.lesk import adapted_lesk
            self.adapted_lesk = adapted_lesk
        return self.adapted_lesk

    def get_wordnet(self):
        if self.wordnet is None:
            import nltk
            nltk.download('wordnet')
            from nltk.corpus import wordnet as wn
            self.wordnet = wn
        return self.wordnet

    def get_requests(self):
        if self.requests is None:
            import requests
            self.requests = requests
        return self.requests

    def summarize_text(self, full_text):
        # Summarize the text using the summarizer
        return self.get_summarizer()(full_text, min_length=60, max_length=500, ratio=0.4)

    def tokenize_sentences(self, text):
        # Tokenize the text into sentences using NLTK's sent_tokenize
        sentences = self.get_sent_tokenize()(text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_sentences_for_keyword(self, keywords, sentences):
        # Filter sentences containing each keyword
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = [sentence for sentence in sentences if word in sentence]
        return keyword_sentences

    def get_nouns_multipartite(self, text):
        out = []
        spacy = self.get_spacy()
        nlp = spacy.load("en_core_web_sm")
        stoplist = set(string.punctuation)
        stoplist |= {'-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'}
        stoplist |= set(self.get_stopwords().words('english'))
        additional_stopwords = {'example', 'examples', 'task', 'entity', 'data', 'use', 'type', 'concepts', 'concept',
                                'learn', 'function', 'method', 'unit', 'fontionality', 'behavior', 'simple', 'ways',
                                'capsule', 'capsules', 'medicines', 'details'}
        stoplist |= additional_stopwords

        # Process text with Spacy's NER and POS tagging
        doc = nlp(text)
        for ent in doc.ents:
            # Extract named entities such as persons, organizations, and locations
            if ent.label_ in {'PERSON', 'ORG', 'GPE'}:
                out.append(ent.text)

        # Filter out non-noun words and include domain-specific terms related to Information Technology
        for token in doc:
            if token.text.lower() not in stoplist and token.pos_ in {'NOUN', 'PROPN'}:
                # Check if the token is a domain-specific term related to Information Technology
                if any(it_term in token.text.lower() for it_term in
                       {'programming', 'software', 'algorithm', 'database', 'network', 'security', 'server', 'cloud',
                        'internet', 'protocol', 'web', 'application', 'interface', 'compiler', 'framework', 'year'}):
                    out.append(token.text)
                # If the token is a named entity, add it regardless of its relation to Information Technology
                elif token.ent_type_ != '':
                    out.append(token.text)

        return out

    def get_distractors_wordnet(self, syn, word):
        # Get distractors from WordNet
        distractors = []
        orig_word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        hypernym = syn.hypernyms()
        if hypernym:
            for item in hypernym[0].hyponyms():
                name = item.lemmas()[0].name().replace("_", " ")
                if name != orig_word:
                    distractors.append(name.capitalize())
        return distractors

    def get_wordsense(self, sent, word):
        # Get word senses from WordNet
        word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        synsets = self.get_wordnet().synsets(word, 'n')
        if synsets:
            wup = self.get_max_similarity()(sent, word, 'wup', pos='n')
            adapted_lesk_output = self.get_adapted_lesk()(sent, word, pos='n')
            lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
            return synsets[lowest_index]
        else:
            return None

    def get_distractors_conceptnet(self, word):
        # Get distractors from ConceptNet
        word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        original_word = word
        distractor_list = []
        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
        obj = self.get_requests().get(url).json()
        for edge in obj['edges']:
            link = edge['end']['term']
            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
            obj2 = self.get_requests().get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2 not in distractor_list and original_word.lower() not in word2.lower():
                    distractor_list.append(word2)
        return distractor_list

    def get_distractors(self, input_file, keyword_sentence_mapping):
        # Get distractors for keywords
        key_distractor_list = {}
        for keyword in keyword_sentence_mapping:
            sentences = keyword_sentence_mapping[keyword]
            if sentences:
                csv_distractors = self.get_distractors_from_csv(input_file, keyword)
                if csv_distractors:
                    key_distractor_list[keyword] = csv_distractors
                else:
                    wordsense = self.get_wordsense(sentences[0], keyword)
                    if wordsense:
                        distractors = self.get_distractors_wordnet(wordsense, keyword)
                        if not distractors:
                            distractors = self.get_distractors_conceptnet(keyword)
                        if distractors:
                            key_distractor_list[keyword] = distractors
        return key_distractor_list

    def get_distractors_from_csv(self, input_file, keyword):
        # Get distractors from a CSV file
        encodings = ['utf-8', 'latin-1', 'utf-16']
        distractors_found = set()
        for encoding in encodings:
            try:
                with open(input_file, 'r', newline='', encoding=encoding) as csvfile:
                    reader = self.get_csv().DictReader(csvfile)
                    for row in reader:
                        key_concept = row['Key Concept']
                        distractors = row['Distractors'].split(', ')
                        if keyword.lower() in key_concept.lower():
                            distractors_found.update([distractor for distractor in distractors if distractor != keyword])
                        if keyword.lower() in distractors:
                            distractors_found.update([distractor for distractor in distractors if distractor != keyword])
                    if distractors_found:
                        return list(distractors_found)
                    break
            except UnicodeDecodeError:
                print(f"Error decoding file with encoding {encoding}. Trying another encoding...")
        return []

    def generate_mcqs(self, text_data):
        # Generate MCQs from the text data
        summarized_text = self.summarize_text(text_data)
        sentences = self.tokenize_sentences(summarized_text)
        keywords = self.get_nouns_multipartite(text_data)
        keyword_sentence_mapping = self.get_sentences_for_keyword(keywords, sentences)
        input_file = 'JAVA.csv'
        key_distractor_list = self.get_distractors(input_file, keyword_sentence_mapping)
        return self.generate_mcqs_from_data(keyword_sentence_mapping, key_distractor_list)

    def generate_mcqs_from_data(self, keyword_sentence_mapping, key_distractor_list):
        # Generate MCQs from keyword-sentence mapping and distractor list
        mcqs = []
        option_choices = ['a', 'b', 'c', 'd']

        for keyword in key_distractor_list:
            sentence = keyword_sentence_mapping[keyword][0]
            pattern = self.get_re().compile(keyword, self.get_re().IGNORECASE)
            output = pattern.sub(" _______ ", sentence)

            if len(key_distractor_list[keyword]) < 3:
                distractors = key_distractor_list[keyword]
                distractors += [''] * (3 - len(distractors))
            else:
                distractors = self.get_random().sample(key_distractor_list[keyword], 3)

            distractors.append(keyword)
            self.get_random().shuffle(distractors)

            mcq = {"question": output, "answer": keyword, "options": dict(zip(option_choices, distractors))}
            mcqs.append(mcq)

        return mcqs
