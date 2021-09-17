FESTIV: Fast Extraction of Scene Text in Video

#############################################################################################

In the FESTIV folder you will find the following:
- CRNN (folder)
- data (folder)
- LM (folder)
- SegLink (folder)
- data_synthtext.py
- results.ipynb
- minimize.py (this is not apart of FESTIV, but rather an attempt to find the optimal initial weights for the Depthwise Separable Convolutions) 
- example_output.mov (this is a video demonstrating FESTIV when run on a home video)

#############################################################################################

STEPS TO TRAIN MODELS

1) Download the SynthText dataset from http://www.robots.ox.ac.uk/~vgg/data/scenetext/

2) Place the downloaded TAR file in ./FESTIV/data/ and unzip it

3) Run the script data_synthtext.py. This is done in order to create a lookup table which is used to speed up the training (and testing processes)

4) To train the SegLink model, run ./SegLink/entrypoint.py. To train the CRNN model, run ./CRNN/crnn_entrypoint.py. NOTE: In order to switch between training the Original models and the Depthwise Separable models you will need to go into the entrypoint.py and crnn_entrypoint.py and change the models that are being used.

#############################################################################################

STEPS TO TEST MODELS

1) Download the SynthText dataset from http://www.robots.ox.ac.uk/~vgg/data/scenetext/

2) Place the downloaded TAR file in ./FESTIV/data/ and unzip it

3) Run the script data_synthtext.py. This is done in order to create a lookup table which is used to speed up the training (and testing processes)

4) Open results.ipynb. This notebook contains all of the code used for testing and visualizing the networks.

#############################################################################################

MISCALLANEOUS NOTES

- The actual definition of the SegLink model is in the ./FESTIV/SegLink/ssd_model.py file, not the ./FESTIV/SegLink/sl_model.py file
- Most files in the project are for the pre-processing and post-processing of the SegLink model
- The ./FESTIV/LM/ folder contains the language model and the lexicons. The file clean.py is where the spell correction logic is held, and the segment.py file is where the segmenting logic is held
