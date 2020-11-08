# Classifying Medical Notes
------------------------------------------------------------------------------------------
## Project Objective
------------------------------------------------------------------------------------------
Can we build a model to classify the are of medicine to which a medical note belongs to?
## Context
------------------------------------------------------------------------------------------
Doctors dictate medical notes following a visit with a patient and these notes are then attached to the patient's chart. Private practices are already specialized but patients who are hospitalized can be visited by doctors of many specialties. The idea behind this project is to get a feel for medical and see if current NLP methods are capable of distinguishing classes that often have overlap between them.
## Criteria for Success
------------------------------------------------------------------------------------------
High accuracy and precision metric for classification.
## Contraints
------------------------------------------------------------------------------------------
Medical texts are difficult to procure due to HIPAA privacy regulations, therefore it is a relatively new field in NLP. The medical lexicon is vastly different then everyday speech where most of the research for NLP has been. To make things more complicated several specialties have a lot in common in addition to all specialties sharing a core list of terms that could otherwise distinguish it from none medical text. If we were to visualize the clusters the data would overlap quite a bit.
## Data Sources
This dataset was made availble on kaggle.com
https://www.kaggle.com/tboyle10/medicaltranscriptions
