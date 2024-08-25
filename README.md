# Glitch-veto-GW-using-chirp
A glitch detection method using Linear Chirp transform and neural networks

This is a project as part of MITACs Globalink GRI'24 under supervision of Dr.Sree Ram Valluri
of Western University,Ontario. This will continue as my Masters thesis too.

I have used Xiyuan Li's Joint Linear Chirp Fourier Transform(JCTFT)-https://arxiv.org/pdf/2209.02673
on glitch as well as merger signals- and used the CNNs on the power distribution at each chirp slice of the 3D spectogram 
to classify as Glitch and Non Glitch.

The latest model trained is the "Model_3".

You could follow other program files for the preprocessing

More will be added and will be finalised as a final complete model as I am working on it.

Currently there is 100 percent accuracy in classifying glitches and merger signals in all training,validation and testing sets.
