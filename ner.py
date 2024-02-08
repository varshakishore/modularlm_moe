import spacy


class NamedEntityRecognition:
    def __init__(self):
        self.tagger = spacy.load('en_core_web_sm')

    def __call__(self, text):
        '''
        Given a string returns a list of NER tags
        No entity is represented as an empty string
        '''
        doc = self.tagger(text)
        words = [token.text for token in doc]
        tags = [token.ent_type_ for token in doc]
        return words, tags

if __name__ == '__main__':
    # NOTE: After installing spacy, download the model with `spacy download en_core_web_sm`
    ner = NamedEntityRecognition()
    text = "Apple is looking at buying U.K. startup for $1 billion"
    words, tags = ner(text)
    print(*zip(words, tags))
