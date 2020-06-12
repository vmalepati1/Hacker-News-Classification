from dataset import Dataset
from model import Model
from standard_classifier import StandardClassifier
from remove_word import RemoveWord
import matplotlib.pyplot as plt

# Baseline experiment
# Load the training dataset with posts from 2018
train_data = Dataset('data/hns_2018_2019.csv', 2018)
# Generate vocab list from post titles
train_data.generate_vocab()
# Save vocab list
train_data.save_vocab('outputs/vocabulary.txt')

# Construct a probabilistic model from the training data
baseline_model = Model(train_data)
# Save class probabilities of each word to the model file
baseline_model.save('outputs/model-2018.txt')

# Load test data with posts from 2019
test_data = Dataset('data/hns_2018_2019.csv', 2019)

# Create classifier
sc = StandardClassifier(train_data, baseline_model, test_data)
# Save results from predictions on test data
sc.save_results('outputs/baseline-result.txt')

print('Baseline accuracy: {0}'.format(sc.accuracy))

# Experiment 1 (Stop-words)

sw = RemoveWord()
sw.load('outputs/remove_word.txt')

sw_train_data = Dataset('data/hns_2018_2019.csv', 2018, stop_words=sw)
sw_train_data.generate_vocab()
sw_train_data.save_vocab('outputs/stopword-vocabulary.txt')

sw_baseline_model = Model(sw_train_data)
sw_baseline_model.save('outputs/stopword-model.txt')

sw_sc = StandardClassifier(sw_train_data, sw_baseline_model, test_data)
sw_sc.save_results('outputs/stopword-result.txt')

print('Stop-word filtering accuracy: {0}'.format(sw_sc.accuracy))

# Experiment 2 (World Length filtering)

wl_train_data = Dataset('data/hns_2018_2019.csv', 2018, word_len_filter=True)
wl_train_data.generate_vocab()
wl_train_data.save_vocab('outputs/wordlength-vocabulary.txt')

wl_model = Model(wl_train_data)
wl_model.save('outputs/wordlength-model.txt')

wl_sc = StandardClassifier(wl_train_data, wl_model, test_data)
wl_sc.save_results('outputs/wordlength-result.txt')

print('Word length filtering accuracy: {0}'.format(wl_sc.accuracy))

# Experiment 3 (Infrequent words filtering)

# X axis is vocab size
th_plot_data_x = []
# Y axis is model accuracy
th_plot_data_y = []

# Frequency thresholds (words with freq < each will be excluded)
frequency_threshs = [1, 5, 10, 15, 20]

# Save each classifier, each one with a different threshold
th_classifiers = []

# Test each model with the frequencies specified above and report results
for thresh in frequency_threshs:
    model_id = 'thresh-{}-'.format(thresh)
    
    th_train_data = Dataset('data/hns_2018_2019.csv', 2018)
    th_train_data.generate_vocab()

    th_baseline_model = Model(th_train_data, freq_thresh=thresh)
    th_baseline_model.frequency_filter()
    th_baseline_model.save('outputs/' + model_id + 'model.txt')

    th_train_data.save_vocab('outputs/' + model_id + 'vocabulary.txt')

    th_sc = StandardClassifier(th_train_data, th_baseline_model, test_data)
    th_sc.save_results('outputs/' + model_id + 'result.txt')

    th_classifiers.append(th_sc)

    print('Threshold = {0} filtering accuracy: {1}'.format(thresh, th_sc.accuracy))

    th_plot_data_x.append(len(th_train_data.vocab_list))
    th_plot_data_y.append(th_sc.accuracy)

# Same as thresholding except now removing the top percent most frequent words
p_plot_data_x = []
p_plot_data_y = []

# List of percents for each model
# What percent of most frequent words will be removed
percents = [0.05, 0.10, 0.15, 0.20, 0.25]

# Save each classifier
p_classifiers = []

# Test each classifier with a different percent to remove
for percent in percents:
    model_id = 'percent-{}-'.format(percent)
    
    p_train_data = Dataset('data/hns_2018_2019.csv', 2018)
    p_train_data.generate_vocab()

    p_baseline_model = Model(p_train_data, rem_freq_per=percent)
    p_baseline_model.frequency_filter()
    p_baseline_model.save('outputs/' + model_id + 'model.txt')

    p_train_data.save_vocab('outputs/' + model_id + 'vocabulary.txt')

    p_sc = StandardClassifier(p_train_data, p_baseline_model, test_data)
    p_sc.save_results('outputs/' + model_id + 'result.txt')

    p_classifiers.append(p_sc)

    print('Percent = {0} filtering accuracy: {1}'.format(percent, p_sc.accuracy))

    p_plot_data_x.append(len(p_train_data.vocab_list))
    p_plot_data_y.append(p_sc.accuracy)

# Plot thresholding results
plt.scatter(th_plot_data_x, th_plot_data_y)
plt.title('Infrequent words threshold filtering')
plt.xlabel('Size of vocabulary (# of words)')
plt.ylabel('Performance (accuracy=TP/Total)')
plt.show()

# Plot thresholding results
plt.scatter(p_plot_data_x, p_plot_data_y)
plt.title('Most frequent percent filtering')
plt.xlabel('Size of vocabulary (# of words)')
plt.ylabel('Performance (accuracy=TP/Total)')
plt.show()
