import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_restaurants = 100
num_reviews = 500
# Generate synthetic data
data = {
    'RestaurantID': np.random.randint(1, num_restaurants + 1, size=num_reviews),
    'Rating': np.random.randint(1, 6, size=num_reviews),  # Rating from 1 to 5
    'ReviewText': [' '.join(np.random.choice(['good', 'bad', 'excellent', 'terrible', 'average', 'delicious', 'awful'], size=np.random.randint(5, 20))) for _ in range(num_reviews)],
    'LongevityYears': np.random.randint(1, 11, size=num_reviews) # Restaurant longevity in years
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['Sentiment'] = df['ReviewText'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
# --- 3. Data Cleaning and Feature Engineering ---
# Group data by RestaurantID and calculate average metrics
restaurant_data = df.groupby('RestaurantID').agg({
    'Rating': 'mean',
    'Sentiment': 'mean',
    'LongevityYears': 'max' # Assume longevity is the maximum observed
}).reset_index()
# --- 4. Analysis and Visualization ---
# Correlation analysis
correlation_matrix = restaurant_data[['Rating', 'Sentiment', 'LongevityYears']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
# Scatter plot: Rating vs. Longevity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Rating', y='LongevityYears', data=restaurant_data)
plt.title('Restaurant Rating vs. Longevity')
plt.xlabel('Average Rating')
plt.ylabel('Longevity (Years)')
plt.savefig('rating_longevity.png')
print("Plot saved to rating_longevity.png")
# --- 5.  (Optional) Predictive Modeling (Simple Linear Regression Example)---
# This section demonstrates a basic predictive model;  more sophisticated models could be used.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = restaurant_data[['Rating', 'Sentiment']]
y = restaurant_data['LongevityYears']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# (Note: Model evaluation is omitted for brevity, but should be included in a real-world application)
print("Linear Regression Model trained. (Evaluation omitted for brevity)")