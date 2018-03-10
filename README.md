Analysis of speech sounds and determination of dialect of those sounds. This project experiments
a how a Gaussian Mixture Models with Hidden Markov-Model (GMM-HMM) can be utilized for dialect
recognition.

<br />

# Description
For the HMM, the hidden states are the speech phonemes and the observations are GMM models
that describe each dialect to classify sound to. An HMM is a sequence model in that it can
be used to find the best output sequence for a given input sequence, or to learn the parameters
for a model. In this project the HMM was used to learn the parameters of the GMMs to be able to
classify the speech sounds to their correct dialect.

<br />

# Experiment
The project was experimented on Arabic and Palestinian dialect datasets. The Arabic dataset contains
speech sounds for Arabic countries and the Palestinian dataset contains speech sounds for four cities
in Palestine.

<br />

# Results
The results were fairly modirate, the recognition of Arabic dialect was better than the recognition
of the Palestinian dialect, and this result is expected since the dialect in a country are close
and difficult to recognize than dialect in different countries.
