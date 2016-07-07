---
layout: post
title: "Batch normalised LSTM for Tensorflow"
date: 2016-07-07
---

Having had some success with batch normalization for a convolutional net I wondered how that'd go for a recurrent one and [this](https://arxiv.org/abs/1603.09025) paper by Cooijmans et al. got me really excited. I decided to try and reimplement the results from their paper on the sequential mnist task.

![Cooijmans results](/images/cooijmans-plot.png){: .center-image }

Sequential mnist was more involved than I first thought as it actually required decent running times on my not too awesome 2GB GTX 960 GPU. Finally, after having implemented almost every single detail of the paper, the results came out okay.

![Olav results](/images/olav-plot.png){: .center-image }

They seem very similar, except for my vanilla LSTM totally falling off the rails and is in the middle of trying to recover towards the end. Luckily the batch normalised LSTM works as reported. Yay!

A more intereting plot is the two runs plotted against wall time instead of step time. The step times for the batch normalised version was 4 times the vanilla one, and actually converged just as slow as the vanilla LSTM in wall time. It could be something crazy bad in my code, but for the sequential mnist the recurrent network is unrolled to 784 steps and calculating the mean and variance statistics for each of those steps is probably heavy.

![Olav results walltime](/images/olav-plot-walltime.png){: .center-image }

The code is on [github](https://github.com/OlavHN/bnlstm), and is the only implementation of batch normalised LSTM for Tensorflow I've seen. If you see the performance error I might've done, I'd love to know!
