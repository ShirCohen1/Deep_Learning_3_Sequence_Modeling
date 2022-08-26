r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=250,
        seq_len=65,
        h_dim=1024,
        n_layers=2,
        dropout=0.25,
        learn_rate=0.001,
        lr_sched_factor=0.9,
        lr_sched_patience=4,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Men at some time are masters of their fates: The fault, dear Brutus, is not in our stars, But in ourselves, that we are underlings."
    temperature = 0.4
    # ========================
    return start_seq, temperature


part1_q1 = r"""
The length of the overall corpus is too long. If we wanted to treat the full corpus as a single sequence we would need a matrix of width = length of corpus. 
A matrix of this size would be compatational very heavy to train and calculate weights. 
Also the reason we are using RNN on this sequence is to add an element of memory. But with such a long sequence, there may not be any conneciton between the beginning 
of the sequence and the end. So it does not make sense to treat it as a single sequence where the start will influence the output of the end.
 """

part1_q2 = r"""
The hidden state represents the previous output that is input into the next step. This is the aspect of memory but it does not depend on the length of the sequence.
It does not have to match the length of the sequence. 
"""

part1_q3 = r"""
Although we do not use a full text as a single sequence as we explained in question 1, while training the model we do rely on memory that is passed
on from sequence to sequence, the hidden states. This memory allows the model to better understand the full context of the corpus
and stores the relevant data from previous sequences which also happen to be similar in context to sequences closer together. 
Therefore, it is improtant to preserve the order of the sequences and not shuffle training samples. 
"""

part1_q4 = r"""
1. The temperature variable determines how uniform the distribution will be. Temperate of 1 gives a fully uniform distribution.
In cases where the char distribution is less uniform (some chars appear more than others) we want to account for this differnt distribution in our model. To do this, we can
lower the temperature so that our model will better fit the data.

2. When temperature is very high we get a more uniform distribution of chars. This means that the softmax should just give us the argmax of the probability vector.

3. When the temperature is very low, there is the most non-uniformity among the distributions.
More weight will be give to classes with higher probabilitiy of occurance in corpus. And less weight will be given to classes
that have less proability of occurance. This will make the plot of Tempreature above more spiky. 

"""
# ==============

