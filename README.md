# Software Requirements Analysis

Our suggested approach toward software requirement analysis with BERT-based models consists of three major steps: classification, topic modeling, and keyword extraction. In each step, we take advantage of novel BERT-based models to handle the corresponding task. Below is the flowchart of the proposed pipeline for software requirement analysis.

<p align = "center">
    <img src="https://github.com/vagabondboffin/topicModeling4UserStories/assets/52859501/312ed587-3295-4cb7-84ea-e56ac00a4d89" alt=".." title="RE"  width = "200"/>
</p>

In each step, the state-of-the-art BERT-based model was used:
+ Classification using [NoRBERT](https://github.com/tobhey/NoRBERT)
+ Topic Modeling using [BERTopic](https://github.com/MaartenGr/BERTopic)
+ Keyword Extraction using [KeyBERT](https://github.com/MaartenGr/KeyBERT)
