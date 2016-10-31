---
layout: post
title: "Attention over attention for cloze -style question answering"
date: 2016-10-30
---

Cloze style question answering is the task of taking an article summary, removing one word, then read the article to predict the word that was removed. The task became popular with the paper [Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340) where Deep Mind prepared datasets compiled from CNN and Daily News.

A property of this task is that the word to be predicted usually exists in the article, so an attention mechanism called [Pointer Networks](https://arxiv.org/abs/1506.03134) has been very successfully [applied](https://arxiv.org/abs/1603.01547).

When pointer networks are applied to cloze style question answering tasks each word in the query document (summary) "predicts" the most likely answer from the context document (article), then those predictions are summed up to get a final prediction over all the words in the article. An interesting twist on this was made in [Attention-over-Attention Neural Networks for Reading Comprehension](https://arxiv.org/abs/1607.04423) where they used the same attention mechanism from context to query (each word in the document attending to the query). This attention was used to get a weigted sum of predictions in the context. The authors reported state of the art results without any hyperparameter tuning.

I've implemented their model in tensorflow, and decided to clean it up and post it when I saw this new, more challenging dataset [Who did What: A Large-Scale Person-Centered Cloze Dataset](https://arxiv.org/abs/1608.05457). In the "future work" section of the attention over attention paper, the authors list applying the trick to iteratively reasoning models. This dataset could be a really nice testbed for such a model.

Code is up on [github](https://github.com/OlavHN/attention-over-attention)
