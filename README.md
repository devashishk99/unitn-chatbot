# UniTN Chatbot

In this project, I have created a chatbot that could be beneficial for student visiting the University of Trento website in order get answers to different questions they might have related to the Uni e.g What can I study at UniTN, What facilities are provided at UniTN, etc. The project makes use of the following tech stack:
- PyTorch
- Flask 
- JavaScript

## Initial Setup:
This repo currently contains the starter files.

Clone repo and create a virtual environment
```
$ git clone https://github.com/python-engineer/chatbot-deployment.git
$ cd chatbot-deployment
$ python3 -m venv venv
```
### Activate virtual environment
Mac / Linux:
```
. venv/bin/activate
```
Windows:
```
venv\Scripts\activate
```
Install dependencies
```
$ pip install -r requirement.txt 
```
Install nltk package
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```

Run
```
$ (venv) python train.py
```
This will dump data.pth file. And then run
the following command to test it in the console.
```
$ (venv) python chat.py
```

In order for deployment and usage of Chatbot in local machine, run the following command and go to [localhost](http://127.0.0.1:5000/) in browser
```
$ (venv) python app.py
```
On opening the localhost on port 5000, click on the Chat icon present in the right bottom corner to start interacting with the UniTN Chatbot.
