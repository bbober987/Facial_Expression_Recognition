# Facial_Expression_Recognition

I've been taking some courses in deep learning on Udemy (Lazy programmer's series). I'll be using the facial expression image dataset obtained from [kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) to test out different architectures and techniques as I learn them as a way to document my progress and solidify my learnings.


# Web app

After getting a reasonably accurate model (61% on a test set -- there is still probably some room for improvement) I decided to learn a little about web development and make this project into a web app. 
The idea is the user can take a picture of themselves on a webcam. I use an out of the box face detector in Open CV to crop the user's face and then resize that image appropriately and send it through the last model that was built in the jupyter notebooks.  
The web app was created using Flask and deployed on pythonanywhere.com, and the code for the app is in the src folder.


You can check out the website [here](bbober.pythonanywhere.com)
