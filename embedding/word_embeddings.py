# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

import torch 
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim 

import matplotlib.pyplot as plt 

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2,5)
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

plt.figure()

# N-Gram Language Modeling
CONTEXT_SIZE_LST = [2,4,6,8,10]
for THE_CONTEXT_SIZE in CONTEXT_SIZE_LST:
    CONTEXT_SIZE = THE_CONTEXT_SIZE
    EMBEDDING_DIM = 10
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

    ngrams = [
        (
            [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
            test_sentence[i]
        )
        for i in range(CONTEXT_SIZE, len(test_sentence))
    ]
    print(f"ngrams[:3]: {ngrams[:3]}")

    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    print(f"word_to_ix: {word_to_ix}")

    class NGramLanguageModeler(nn.Module):
        def __init__(self, vocab_size, embedding_dim, context_size):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, 128)
            self.linear2 = nn.Linear(128, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs).view((1,-1))
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            log_probs = F.log_softmax(out, dim=1)
            return log_probs

    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(500):
        total_loss = 0
        for context, target in ngrams:
            # prepare the inputs to be passed to the model (turn the words into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)

    print(losses)
    print(model.embeddings.weight[word_to_ix["beauty"]])

    plt.plot(losses, label=f"CONTEXT_SIZE={CONTEXT_SIZE}")

plt.legend()
plt.savefig("embedding.png")