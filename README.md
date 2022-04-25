# Bite-Timing Prediction for Robot-Assisted Feeding in a Social Dining Setting


### Code & data organization

* data  \
[reflects the Box “data” folder;  \
not pushed to GitHub;  \
to keep consistency between our setups;  \
this will be published to Harvard Dataverse ]
    * bag
    * annotation
    * questionnaires
    * documentation-pictures
    * bag-extraction-scripts [to extract raw data from bag to the “raw” subfolder]
    * README.md [needs to be updated!]
    * raw [raw data extracted from bag files]
        * audio
        * video
        * depth
        * …
    * processed [data pre-processed from raw data]
        * visual-features
            * openpose
            * openface
            * resnet
            * rt-gene
            * i3d
            * …
        * audio-features
            * binary-speaking
            * sound-direction
            * mfcc
            * logfb
            * log-mel
            * …
        * timing-features
            * last-lifted-times
            * last-to-mouth-times
            * …
* src
    * preprocessing  \
[feature extraction scripts to obtain data in data/processed from data/raw]
    * analysis [data analysis, statistics, visualizations]
    * dataset [dataset loader script; load variations of modalities]
    * models [all ML-related]
        * model1
        * model2
        * …

