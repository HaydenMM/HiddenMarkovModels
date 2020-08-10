# Named entity recognition system using Hidden Markov Models
## learnhmm.py
```python learnhmm.py [args...]``` \
Where above [args...] is a placeholder for six command-line arguments: train input, index to word, index to tag, hmmprior, hmmemit, hmmtrans
1. train input: path to the training input .txt file
2. index to word: path to the .txt that specifies the dictionary mapping from words to indices. The tags are ordered by index, with the first word having index of 1, the second word having index of 2, etc.
3. index to tag: path to the .txt that specifies the dictionary mapping from tags to indices. The tags are ordered by index, with the first tag having index of 1, the second tag having index of 2, etc.
4. hmmprior: path to output .txt file to which the estimated prior (π) will be written. The file output to this path should be in the same format as the handout hmmprior.txt
5. hmmemit: path to output .txt file to which the emission probabilities (B) will be written. The file output to this path should be in the same format as the handout hmmemit.txt
6. hmmtrans: path to output .txt file to which the transition probabilities (A) will be written. The file output to this path should be in the same format as the handout hmmtrans.txt

## forwardbackward.py
```python forwardbackward.py [args...]``` \
Where above [args...] is a placeholder for seven command-line arguments:test input, index to word, index to tag, hmmprior, hmmemit, hmmtrans, predicted file, metric file
1. test input: path to the test input .txt file that will be evaluated by your forward backward algorithm
2. index to word: path to the .txt that specifies the dictionary mapping from words to indices. The tags are ordered by index, with the first word having index of 1, the second word having index of 2, etc. This is the same file as was described for learnhmm.{py|java|cpp|m}.
3. index to tag: path to the .txt that specifies the dictionary mapping from tags to indices. The tags are ordered by index, with the first tag having index of 1, the second tag having index of 2, etc. This is the same file as was described for learnhmm.{py|java|cpp|m}.
4. hmmprior: path to input .txt file which contains the estimated prior (π).
5. hmmemit: path to input .txt file which contains the emission probabilities (B).
6. hmmtrans: path to input .txt file which contains transition probabilities (A).
7. predicted file: path to the output .txt file to which the predicted tags will be written. The file should be in the same format as the <test input> file.
8. metric file: path to the output .txt file to which the metrics will be written.
