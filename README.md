# GUIDE TO HAVE THE PROJECT RUNNING
## Step by step commands to get your environment running.

Prepare the AWS instance and get super user permissions.

```
sudo su
```

Install git and Python 3 if not already on the instance

```
yum install git
yum install python3
sudo easy_install pip
```

Install the requirements.txt libraries:

```
pip3 install -r Requirements.txt
```

The code is available in our git repo. Clone to repository and get the code

```
git clone https://
ls
 
cd RepoName/
ls
```

Run the flask application-

```
python3 Project_FinanicalTech.py
```

The application should be running by now.

## Look at Application
Look at the application using the link - http://'your IP address':8080/home

## Services Description 

Book Search: Search Service for books.
Translate Blurb: Translate the blurb in a language you are more familiar with.
Books Similarity: Find similarity score based on blurbs between two titles.
Search Book on Google: Enter the book title, we give you the top links on Google. 
Sentiment Analysis: Given the title we give you the sentiment analysis of the blurb.

