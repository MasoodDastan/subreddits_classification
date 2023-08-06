## Problem Statement:

Cornerstone Financial Services (CFS), an emerging and versatile financial institution providing a comprehensive range of services, including banking, loans, and financial investments, is striving to optimize its targeted advertising campaigns.  CFS aims to deliver personalized ads to customers based on their online posts. The main objective is to identify the most suitable advertisements for each individual, specifically focusing on distinguishing customers with an interest in investing and stock market from those who may benefit from information about other services such as credit and loan services. By collaborating with our team at **New Edge Solutions**, CFS aims to leverage advanced natural language processing techniques to develop an innovative text analysis model. The goal is to tailor CFS's advertising strategy to align with customers' preferences, thereby maximizing the effectiveness of their promotional efforts.


## Data and project limitations:

Due to the time constraints of the project, our team had to expedite the data acquisition process. To gather data quickly, we decided to utilize customized subreddits on Reddit that are specifically related to finance. Among these subreddits, two main ones stood out as perfect candidates for our task: __Investing__ and **Personal Finance**.

The [*Investing*](https://www.reddit.com/r/investing/) subreddit primarily focuses on discussions and insights related to various investment opportunities, stock markets, companies, and financial trends. On the other hand, the [*Personal Finance*](https://www.reddit.com/r/personalfinance/) subreddit centers around discussions regarding personal financial management, budgeting, debt management, and general financial advice for individuals. Although both subreddits are finance-oriented, their objectives and areas of emphasis differ.

Despite the promising choice of subreddits, our project encountered an unexpected challenge when Reddit changed its API access rules shortly after our project initiation. This change severely impacted our daily access to data from Reddit. We faced limitations such as a maximum of 1000 posts per query and the unavailability of searching across time. These restrictions significantly hindered our ability to access data and engineer accurate models.

After consulting with Cornerstone Financial Services (CFS), it was decided to gather as much data as possible from Reddit and create a benchmark model based on the available data. Our team diligently scraped data from Reddit on a daily basis, starting from June 13th until June 18th, in order to collect a substantial amount of data. To maximize sample size , we expanded our collection beyond just new posts and also included data from the top, hot, and controversial categories within each subreddit.

The final dataset contains four columns. These columns include:

|Feature|Type|Description|
|---|---|---|
|**Timestamp**|*float*|Captures the date and time when each post was written.| 
|**Title**|*object*|The title of each post. Provides a concise summary or topic of discussion for the post.|
|**Text**|*object*|Contains the full text or body of each post. Provides a deeper understanding of the information, opinions, or questions shared by the users.| 
|**Subreddit**|*object*|Identifies the specific subreddit to which each post belongs (*Investing* or *Personal Finance*).| 


## Executive Summary:

At the core of this project is the objective of Cornerstone Financial Services (CFS) to optimize their targeted advertising campaigns and deliver personalized ads to potential customers based on their online posts and activities. To accomplish this, CFS plans to collaborate with New Edge Solution and their at-home analytics team to develop an innovative text analysis model.This model will enable them to discern customers' interests and tailor advertising accordingly. By leveraging advanced natural language processing techniques, this model will allow CFS to identify customers' interests and customize advertising efforts accordingly.

Data was collected from two popular subreddits, Investing and Personal Finance, over a period of 6 days. To ensure comprehensive coverage, data was gathered from various categories such as New, Hot, Top, and Controversial. Posts were also collected from different time ranges within the Top and Controversial categories, including all-time, past year, past month, and past week. Duplicate posts were removed to create a dataset of approximately 11,825 unique posts.

During the data cleaning process, posts with missing content were identified and replaced with an empty string. To maximize the available information for analysis, the title and text columns were combined. Furthermore, any mentions of the subreddit names (investing and personal finance) were removed to prevent the models from receiving excessive information that could potentially aid their predictions. Finally, to ensure consistency, all text data was converted to lowercase.

The dataset was divided into three sets - training, validation, and test - using a 60-20-20 split. We conducted extensive exploratory data analysis to uncover patterns and gain insights from the sample. Furthermore, we implemented text normalization techniques, including stemming and lemmatization to prepare the text data for modeling.

For the purpose of vectorizing the text data, we made use of two methods from the `sklearn.feature_extraction` module: `CountVectorizer` and `TfidfVectorizer`. These vectorization techniques enabled us to convert the textual information into numerical features that are compatible with machine learning algorithms.

Five classification models - Naïve Bayes, Logistic Regression, Random Forest, Support Vector Machine, and Gradient Boosting - were evaluated using a cross-validated randomized search over the vectorized data. This search aimed to identify the model with the best hyperparameters that yielded the highest performance.

Finally, the selected model from the randomized search was applied to unseen data for a final assessment of its performance. This evaluation allowed us to make informed decisions about the model's suitability for classifying posts.

This project involved comprehensive processes of data collection, cleaning, exploration, and modeling. By leveraging advanced techniques, the objective was to develop an accurate classification model that assists Cornerstone Financial Services in delivering targeted advertisements and enhancing their promotional efforts.

The model's performance is exceptional, surpassing the benchmark by around 30 percentage points and achieving a remarkable 97% AUC score. With an overall accuracy of 92%, a recall rate of 94%, and a specificity of 89%, the model demonstrates strong predictive capabilities. These results indicate its ability to accurately classify and differentiate between different categories. The model's performance is highly promising, and there is potential for even greater achievements by incorporating a larger and more up-to-date dataset. 