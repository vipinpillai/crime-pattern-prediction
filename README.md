# Predicting Crime Patterns for Pro-active Law Enforcement

In this project, we attempt to learn the spatiotemporal relations between all the features from Baltimore Crime and Arrests datasets respectively. 

Our goal is to predict the Crime Category for a given Premise, Neighborhood, District, Weapon, Crime Hour and Crime Day. Using this predicted Crime Category, we would like to further use the known attributes from the BPD
Arrests dataset such as - District, Neighborhood, Charge, IncidentLocation, IncidentOffense, and attempt to predict the ArrestLocation for the predicted Crime Category. In addition to predicting Crime Category, we would also like to find the Crime Hotspots (Premise) for different crime categories.

We believe this can result in efficient allocation of BPD resources for crime resolution.

## We have used scikit learn and Keras (with Theano backend) for implementing our models which can be installed from the below links:
1. Scikit learn:  http://scikit-learn.org/stable/install.html
2. Keras: https://keras.io/#installation
3. Theano: http://deeplearning.net/software/theano/install.html#install

