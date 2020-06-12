import pandas as pd
import re

class Dataset:

    def __init__(self, posts_file, year, stop_words=None, word_len_filter=False):
        # Load the dataframe
        self.posts_df = pd.read_csv(posts_file, index_col=None)
        # Only extract posts from the year specified
        self.posts_df = self.posts_df[self.posts_df['year']==year]
        # Format titles (lowercase, remove delimiters, etc.)
        self.preprocess()

        self.stop_words = stop_words
        self.word_len_filter = word_len_filter
        self.vocab_list = []

    def preprocess(self, post_col_name='Title_reformatted'):
        reformatted_col = []
        
        for post_title in self.posts_df['Title']:
            # Lowercase
            post_title = post_title.lower()

            # Only keep alphanumerics + some other characters
            post_title = re.sub(r'[^A-Za-z0-9 _-]+', '', post_title)

            # Should not be considered two separate words
            post_title = post_title.replace("ask hn", "ask-hn")
            post_title = post_title.replace("show hn", "show-hn")

            reformatted_col.append(post_title)

        self.posts_df[post_col_name] = reformatted_col

    def generate_vocab(self):
        # Go through each post title
        for post_title in self.posts_df['Title_reformatted']:
            # Extract words, remove empty strings and '-'
            words = post_title.split(' ')
            words = list(filter(None, words))
            words = list(filter(lambda w: w != '-', words))

            # Remove stop words
            if self.stop_words:
                words = list(filter(lambda w: w not in self.stop_words.list_of_words, words))

            # Remove words with len <= 2 or len >= 9
            if self.word_len_filter:
                words = list(filter(lambda w: len(w) > 2 and len(w) < 9, words))

            # Merge list of words into our vocab list
            self.vocab_list += words

        # Remove duplicate words
        self.vocab_list = list(set(self.vocab_list))
        # Sort alphabetically
        self.vocab_list.sort()

    def save_vocab(self, vocab_file):
        # Save each vocab word to the file
        with open(vocab_file, "w") as f:
            for word in self.vocab_list:
                f.write(word + "\n")


