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
yum install python-pip
```

The code is available in our git repo. Clone to repository and get the code

```
git clone https:// --> LINK TO OUR REPO <--
ls
 
cd FIN_TEST/
ls
```

Install the requirements.txt libraries:

```
pip3 install -r requirements.txt
```

Run the flask application-

```
python3 Project_FinanicalTech.py
```

The application should be running by now.

## Look at Application
Look at the application using the link - http://'your IP address':8080/home

## Services Description 

Book Search: Search Service for books based on Author, Title, ISBN, Year and Publisher.\n
Translate Blurb: Translate the blurb in a language you are more familiar with.
Books Similarity: Find similarity score based on blurbs between two titles.
Search Book on Google: Enter the book title, we give you the top links on Google. 
Sentiment Analysis: Given the title we give you the sentiment analysis of the blurb.
Sentiment for Title: Given a number we return the top and bottom book titles.
Sentiment for Blurbs: Given a number we return the top and bottom book blurbs.
Wiki Search: Given the author name we return the author photo and a short description of the author.
Wordcloud from blurb adjectives: Given a book title we return a wordcloud made from the blurb adjectives.
Random Book: Don't know what to read? We will give you a random book form our database.
Random Author: Want to know a new outhor? We will give you a random author form our database.
More Information: Are you not satisfied with the information provided for a particular book? Click here to get more.

