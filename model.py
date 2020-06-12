import numpy as np
from collections import Counter

class Model:

    def __init__(self, dataset, smooth_k=0.5, rem_freq_per=0, freq_thresh=None):
        self.dataset = dataset
        self.smooth_k = smooth_k
        self.rem_freq_per = rem_freq_per
        self.freq_thresh = freq_thresh
        
        # Compute frequencies of words in each story type
        self.story_frequencies = self.compute_frequencies('story')
        self.ask_hn_frequencies = self.compute_frequencies('ask_hn')
        self.show_hn_frequencies = self.compute_frequencies('show_hn')
        self.poll_frequencies = self.compute_frequencies('poll')

        # Conditional probabilites of each word being for each of the classes
        self.word_probabilities = {}

    def compute_frequencies(self, post_type):
        frequencies = {}
        relevant_posts = self.dataset.posts_df[self.dataset.posts_df['Post Type'] == post_type]

        # Go through each post with the specified type and store the word frequencies
        for post_title in relevant_posts['Title_reformatted']:
            words = post_title.split(' ')
            words = list(filter(None, words))
            words = list(filter(lambda w: w != '-', words))

            for word in words:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

        return frequencies

    def frequency_filter(self):
        # If filtering is disabled, no need to proceed
        if not self.freq_thresh and self.rem_freq_per == 0:
            return

        words_to_remove = []

        # Dictionary with keys being words and value being the total frequency across all classes
        word_total_freq = {}

        for word in self.dataset.vocab_list:
            total_freq = self.get_freq(self.story_frequencies, word) + \
                           self.get_freq(self.ask_hn_frequencies, word) + \
                           self.get_freq(self.show_hn_frequencies, word) + \
                           self.get_freq(self.poll_frequencies, word)

            if self.freq_thresh:
                # Remove words that are infrequent (less than given threshold)
                if total_freq <= self.freq_thresh:
                    words_to_remove.append(word)

            word_total_freq[word] = total_freq

        if self.rem_freq_per != 0:
            c = Counter(word_total_freq)
            # Extract most frequent words by percentile
            words_to_remove += [tup[0] for tup in c.most_common(int(len(word_total_freq)/(1/self.rem_freq_per)))]

        # Remove duplicates  
        words_to_remove = list(set(words_to_remove))

        # Remove words from vocab list
        self.dataset.vocab_list = [x for x in self.dataset.vocab_list if x not in words_to_remove]

    def get_freq(self, freq_dict, word):
        if word in freq_dict:
            return freq_dict[word]
        else:
            return 0

    def save(self, model_file):
        line_counter = 1
        # Laplace probability = (Occurrences + k) / (N + k * d)

        # Total number of posts for each class
        N_story = sum(self.story_frequencies.values())
        N_ask_hn = sum(self.ask_hn_frequencies.values())
        N_show_hn = sum(self.show_hn_frequencies.values())
        N_poll = sum(self.poll_frequencies.values())

        # Domain of each story (number of unique words)
        d_story = len(self.story_frequencies)
        d_ask_hn = len(self.ask_hn_frequencies)
        d_show_hn = len(self.show_hn_frequencies)
        d_poll = len(self.poll_frequencies)
        
        with open(model_file, "w") as f:
            for wi in self.dataset.vocab_list:
                # Get the frequency of the word in each class
                story_freq = self.get_freq(self.story_frequencies, wi)
                ask_hn_freq = self.get_freq(self.ask_hn_frequencies, wi)
                show_hn_freq = self.get_freq(self.show_hn_frequencies, wi)
                poll_freq = self.get_freq(self.poll_frequencies, wi)

                story_p = 0
                ask_hn_p = 0
                show_hn_p = 0
                poll_p = 0

                # Compute laplace (smoothed) probabilities
                if len(self.story_frequencies) > 0:
                    story_p = (story_freq + self.smooth_k) / (N_story + d_story * self.smooth_k)

                if len(self.ask_hn_frequencies) > 0:
                    ask_hn_p = (ask_hn_freq + self.smooth_k) / (N_ask_hn + d_ask_hn * self.smooth_k)

                if len(self.show_hn_frequencies):
                    show_hn_p = (show_hn_freq + self.smooth_k) / (N_show_hn + d_show_hn * self.smooth_k)

                if len(self.poll_frequencies):
                    poll_p = (poll_freq + self.smooth_k) / (N_poll + d_poll * self.smooth_k)

                # Save probabilities to model file
                f.write("{0: <10}  {1: <35}  {2: <5}  {3:20.10f}  {4: <5}  {5:20.10f}  {6: <5}  {7:20.10f}  {8: <5}  {9:20.10f}\n".format(
                    line_counter, wi, story_freq, story_p, ask_hn_freq, ask_hn_p, show_hn_freq, show_hn_p,
                    poll_freq, poll_p))
                
                # Store probabilities for further reference
                self.word_probabilities[wi] = [story_p, ask_hn_p, show_hn_p, poll_p]
                
                line_counter += 1
