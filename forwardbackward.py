# Hayden Moore
# 10-301 / 10-601
# HW7 forward backward

import sys
import numpy
import math


def load_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    i = 0
    for line in data:
        data[i] = line.rstrip()
        i += 1

    return data


def word_to_index(data, word_index, tag_index):
    new_data = []

    for line in data:
        row = []
        for word in line:
            word_and_index = word.split('_')

            # get word index
            i = 0
            for comp_word in word_index:
                if word_and_index[0] == comp_word:
                    w_index = i
                i += 1

            # get tag index
            j = 0
            for comp_tag in tag_index:
                if word_and_index[1] == comp_tag:
                    t_index = j
                j += 1

            # append row
            row.append([w_index, t_index])
        new_data.append(row)

    return new_data


def log_likelihood(alpha):
    return math.log(numpy.sum(alpha[-1]))


def forward(data, words, tags, pi, b, a):
    # get alpha
    size = len(data)
    big_alpha = []
    likelihood = []

    j = 0
    for line in data:
        # if j == 10:
        #     return big_alpha, likelihood
        alpha = []
        i = 0
        for word in line:
            # remove tag
            w_index = word[0]

            # only if first word in line
            if i == 0:
                obs = []
                ii = 0
                for p in pi:
                    prod = float(p) * float(b[ii][w_index])
                    obs.append(prod)
                    ii += 1
                #obs = numpy.array(obs)
                alpha.append(obs)


            # otherwise calculate alpha normally
            else:
                obs = []
                jj = 0
                for line in a:
                    iii = 0
                    row = []
                    for prob in line:
                        prod = float(alpha[i - 1][jj]) * float(prob) * float(b[iii][w_index])
                        row.append(prod)
                        iii += 1
                    jj += 1
                    obs.append(row)
                obs = numpy.array(obs)
                obs = obs.sum(axis=0)
                obs = obs.tolist()
                alpha.append(obs)
            i += 1
        big_alpha.append(alpha)
        log_like = log_likelihood(alpha)
        likelihood.append(log_like)
        j += 1
    big_alpha = numpy.array(big_alpha)

    return big_alpha, likelihood


def backward(data, emit, trans):
    big_beta = []
    for line in data:
        size = len(line)
        beta = numpy.zeros((size, len(emit)))
        i = size - 1
        for word in line:
            #print(word)
            # if first item
            if i == size - 1:
                ii = 0
                for b in beta[i]:
                    beta[i][ii] = 1
                    ii += 1

            # otherwise calculate beta normally
            else:
                b_index = line[i + 1][0]
                b_t = emit[:, b_index]
                b_t = b_t.astype(numpy.float)
                trans = trans.astype(numpy.float)
                beta_t = beta[i + 1]
                prod = numpy.multiply(b_t, beta_t)

                beta[i] = numpy.matmul(trans, prod)

            i -= 1
        big_beta.append(beta)
    return big_beta


def main():
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    # load in the test input
    test_data = load_file(test_input)
    i = 0
    for line in test_data:
        test_data[i] = line.split()
        i += 1

    # load in word index
    words = load_file(index_to_word)

    # load in tag idex
    tags = load_file(index_to_tag)

    # load in prior
    prior = load_file(hmmprior)
    prior = numpy.array(prior)

    # load in emit
    emit = load_file(hmmemit)
    i = 0
    for line in emit:
        emit[i] = line.split()
        i += 1
    emit = numpy.array(emit)

    # load in trans
    trans = load_file(hmmtrans)
    i = 0
    for line in trans:
        trans[i] = line.split()
        i += 1
    trans = numpy.array(trans)

    # switch test data to index representation
    test_data = word_to_index(test_data, words, tags)

    # run thee forward part of the algorithm
    alpha, likelihood = forward(test_data, words, tags, prior, emit, trans)

    # run backward part of the algorithm
    beta = backward(test_data, emit, trans)

    big_labels = []
    i = 0
    for a in alpha:
        predictions = numpy.multiply(alpha[i], beta[i])
        #print(predictions)
        labels = []
        for pred in predictions:
            index = numpy.where(pred == numpy.amax(pred))
            labels.append(tags[index[0][0]])
        big_labels.append(labels)
        i += 1

    # find average log likelihood
    avg_likelihood = numpy.average(likelihood)

    # write outputs
    wrong = 0
    with open(predicted_file, 'w') as f:
        ii = 0
        for line in test_data:
            i = 0
            for w in line:
                t = tags[w[1]]
                if i == len(line) - 1:
                    f.write(words[w[0]] + '_' + big_labels[ii][i])
                else:
                    f.write(words[w[0]] + '_' + big_labels[ii][i] + ' ')
                if t != big_labels[ii][i]:
                    wrong += 1
                i += 1
            ii += 1
            f.write('\n')


    total = 0
    i = 0
    for label in big_labels:
        total += len(label)

    accuracy = (total - wrong) / total
    with open(metric_file, 'w') as f:
        f.write("Average Log-Likelihood: " + str(avg_likelihood) + '\n')
        f.write("Accuracy: " + str(accuracy))


if __name__ == "__main__":
    main()
