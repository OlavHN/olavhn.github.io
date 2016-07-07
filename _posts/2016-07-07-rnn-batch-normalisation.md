---
layout: post
title: "Batch normalised LSTM for Tensorflow"
date: 2016-07-07
---

I was playing around, trying to implement the model in [Sutskever et al.](https://arxiv.org/abs/1409.3215) paper for translating natural language. Things looked good so I left the computer to do it's thing over the weekend. Unfortunately I came back to wasted GPU cycles as down the line, something went very wrong.

![Model crash](/images/plot-crash.png){: .center-image }

I tried adding a bunch of logging and reloading the model, but couldn't pinpoint why it crashed so badly. Still, I suspect not using any kind of regularization might be part of it.

Having had some success with batch normalization for a convolutional net I wondered how that'd go for a recurrent one and [this](https://arxiv.org/abs/1603.09025) paper by Cooijmans et al. got me really excited. I decided to try and reimplement the results from their paper on the sequential mnist task.

![Cooijmans results](/images/cooijmans-plot.png){: .center-image }

Sequential mnist was harder than I first thought as it actually required decent running times on my awesome 2GB GTX 960 GPU. Finally, after having implemented every single detail of the paper, the tests ran okay.

![Olav results](/images/olav-plot.png){: .center-image }

They seem very similar, except for my vanilla mnist totally falling off the rails and trying to recover towards the end. More importantly the batch normalised LSTM works as reported. Yay!

![Olav results walltime](/images/olav-plot-walltime.png){: .center-image }

A more intereting plot is the two runs plotted against wall time instead of step time. The step times for the batch normalised version was 4 times the vanilla one, and actually converged just as slow as the vanilla LSTM (a lot more stable tho). It could be something crazy bad in my code, but for the sequential mnist the recurrent network is unrolled to 784 steps and calculating the mean and variance statistics for each of those steps is probably heavy.

The code is on [github](https://github.com/OlavHN/bnlstm), and is the only implementation of batch normalised LSTM for Tensorflow I've seen. If you see the performance error I might've done, I'd love to know!
