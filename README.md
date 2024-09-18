# Glitch-veto-GW-using-chirp
A glitch detection method using Linear Chirp transform and neural networks

This is a project as part of MITACs Globalink GRI'24 under supervision of Dr.Sree Ram Valluri
of Western University,Ontario. This will continue as my Masters thesis too.

I have used Xiyuan Li's Joint Linear Chirp Fourier Transform(JCTFT)-https://arxiv.org/pdf/2209.02673
on glitch as well as merger signals- and used the CNNs on the power distribution at each chirp slice of the 3D spectogram 
to classify as Glitch and Non Glitch.

The trained model is "Model_classification,hdf5".You could use the corresponding program to check the model architecture and run it.

The pipeline should be followed such that-
1)Run the Glitch_data_pipeline to download glitch signals as .hdf5 files having attributes. The csv files uploaded have the url and will download. If system is non-linux change the code accordingly
2)Run the glitch_volume_stats.csv and Merger_volumes.csv for the chirp transform and saving the positive and negative chirp domain's power distribution as csv files in separate directories.
3)Run the Model_classification on the csv files containing distributions to get a classification.

Currently there is 100 percent accuracy in classifying glitches and merger signals in all training,validation and testing sets.

I am attaching the Gltich_signal data directly here for easier download and reference incase you want to skip the initial downloading part or even skip processing the transform and directly just use the model.
