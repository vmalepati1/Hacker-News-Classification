from collections import OrderedDict
import math

class StandardClassifier:

    def __init__(self, training_dataset, model, test_dataset):
        self.training_dataset = training_dataset
        self.model = model
        self.test_dataset = test_dataset
        self.accuracy = 0

    def predict(self, title):
        # Prediction in log10 space
        # Multiplying very small probabilities is not optimal with a large set of features (words)
        # Computers have a limited precision and cannot store extremely small numbers, so we use log10 here
        # Based from here: http://www.cs.rhodes.edu/~kirlinp/courses/ai/f18/projects/proj3/naive-bayes-log-probs.pdf

        # Get tokenized words in title
        words = title.split(' ')
        words = list(filter(None, words))
        words = list(filter(lambda w: w != '-', words))

        # Class scores dict
        scores = OrderedDict({'story': 0, 'ask_hn': 0, 'show_hn': 0, 'poll': 0})

        df = self.training_dataset.posts_df

        for p_index, post_type in enumerate(scores.keys()):
            score = 0

            # Sum logs of probabilities of each word in the class if they exist in vocab
            for word in words:
                if word in self.model.word_probabilities:
                    word_given_class_p = self.model.word_probabilities[word][p_index]

                    if word_given_class_p > 0:
                        score += math.log(word_given_class_p, 10)

            num_post_type = (df['Post Type'].values == post_type).sum()
            total_num_posts = df.shape[0]

            # Calculate the probability of a post being this type
            post_p = num_post_type / total_num_posts

            # Add to the sum
            if post_p > 0:
                score += math.log(post_p, 10)

            # If post does not occur at all, it has -inf probability
            # Other posts are negative scores due to log, so need to set this to -inf
            if score == 0:
                score = -math.inf

            # Save score
            scores[post_type] = score

        return scores

    def save_results(self, results_file):
        with open(results_file, "w") as f:
            line_counter = 0
            num_right = 0
            
            # Go through each post in the test dataset
            for idx, series in self.test_dataset.posts_df.iterrows():
                post_title = series['Title_reformatted']
                correct_classification = series['Post Type']

                scores = self.predict(post_title)

                # Get the key (post type) with the highest value (score)
                # This is our classifier's classification (argmax)
                classification = max(scores, key=scores.get)

                story_score = scores['story']
                ask_hn_score = scores['ask_hn']
                show_hn_score = scores['show_hn']
                poll_score = scores['poll']

                label = ('right' if classification == correct_classification else 'wrong')

                if label == 'right':
                    num_right += 1

                line_counter += 1

                # Save results
                f.write("{0: <10}  {1: <100}  {2: <10}  {3:10.5f}  {4:10.5f}  {5:10.5f}  {6:10.5f}  {7: <10}  {8: <5}\n".format(
                    line_counter, post_title, classification, story_score, ask_hn_score, show_hn_score, poll_score,
                    correct_classification, label))

            # Calculate accuracy
            self.accuracy = num_right / line_counter
        
