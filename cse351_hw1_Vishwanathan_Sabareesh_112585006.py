# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from wordcloud import WordCloud 

# %%
df = pd.read_csv("AB_NYC_2019.csv")
df.head()

# %%
# Task 1: Examine the data, there may be some anomalies in the data, and you will have to clean the data
# before you move forward to other tasks. Explain what you did to clean the data. 

# %%
# Check for zeroes in host_id, latitude, longitude, price, minimum nights, # of reviews, availability_365 and evaluate
# Zeroes in host_id, there are none
((df['host_id'] == 0)).any()

# %%
# Zeroes in latitude, there are none
((df['latitude'] == 0)).any()

# %%
# Zeroes in longitude, there are none
((df['longitude'] == 0)).any()

# %%
# Zeroes in minimum_nights, there are none
((df['minimum_nights'] == 0)).any()

# %%
# Zeroes in price, there are some
((df['price'] == 0)).any()

# %%
# (df['price'] == 0).sum()
len(df.loc[df['price'] == 0])

# %%
# Check number of listings before
len(df)

# %%
# Remove zero rows because removing only 11 out of all 48895 values wouldn't change much. Values can be removed with below code
# df.drop(df[df['price'] <= 0].index, inplace = True)

# %%
# Check number of listings after if you choose to remove 0 rows due to price
len(df)

# %%
# Zeroes in number_of_reviews, there are some, which is fine because some listings may not have been occupied
((df['number_of_reviews'] == 0)).any()

# %%
# Zeroes in availability_365, there are some, which is fine because some listings might be booked fully
((df['availability_365'] == 0)).any()

# %%
# Next step of clean is to look for outliers, can do so by using box plots

# %%
# Boxplot for prices, there are some outliers
df.boxplot(column=['price'], vert=False, figsize=(20,5)).set_title("Prices of Airbnb Listings")

# %%
# Checking listings with prices over 4000. They seem to be expensive because of the location, minimum nights, and 
# because some are luxury apartments as indicated by the name. Wouldn't make sense to remove these listings. 
df.loc[df['price'] > 4000].sort_values(by=['price'])

# %%
# Task 2: Examine how the prices of the Airbnb changes with the change in the neighborhood.
# a. Find Top 5 and Bottom 5 neighborhood based on the price of the Airbnb in that neighborhood
# (select only neighborhoods with more than 5 listings). (10 Points)
# b. Analyze, the price variation between different neighborhood group, and plot these trends. (5 Points)

# %%
# Top 5 neighborhood based on prices with neighborhoods with more than 5 listings are:
# Williamsburg, Bedford-Stuyvesant, Harlem, Bushwick and Upper West Side
top5df = df['neighbourhood'].value_counts()
top5df = top5df[top5df > 5]
top5df

# %%
# Bottom 5 neighborhood based on prices with neighborhoods with more than 5 listings are:
# Bull's Head, Midland Beach, Grant City, Mount Eden, Bay Terrace.
# Filtered == 6 to make sure there were only 5 neighborhoods that were above 5, as seen by above cell.
bot5df = df['neighbourhood'].value_counts()
bot5df = bot5df[bot5df == 6]
bot5df

# %%
# Part B, check price variation between different groups
ngdf = df.groupby('neighbourhood_group')['price'].mean().to_frame()
ngdf.plot.bar(y='price', rot=0, title='Mean of Neighborhood Groups in Airbnb Homes')


# %%
ngdf = df.groupby('neighbourhood_group')['price'].median().to_frame()
ngdf.plot.bar(y='price', rot=0, title='Median of Neighborhood Groups in Airbnb Homes')

# %%
ngdf = df.groupby('neighbourhood_group')['price'].std().to_frame()
ngdf.plot.bar(y='price', rot=0, title='Standard Deviation of Neighborhood Groups in Airbnb Homes')

# %%
# Task 3: Select a set of the most interesting features. Do a pairwise Pearson correlation analysis on all pairs
# of these variables. Show the result with a heat map and find out most positive and negative correlations. (5 points)

# %%
df.corr()

# %%
# Interesting Features:
# Most positive correlations:
# Reviews per month and number of reviews (0.55). This makes sense because the number of reviews per month is the total number of reviews 
# divided by the number of months a property was listed for.
# Most negative correlations:
# Minimum nights and reviews per month
corrdf = df.copy()
corrdf = corrdf.drop(columns=['id', 'host_id'])
dataplot = sb.heatmap(corrdf.corr(), cmap="YlGnBu", annot=True).set_title("Correlations between Interesting Listing Features")

sb.set(rc = {'figure.figsize':(8,8)})
plt.show()

# %%
# Task 4: The Latitude and Longitude of all the Airbnb listings are provided in the dataset.
# a. Plot a scatter plot based on these coordinates, where the points represent the location of an
# Airbnb, and the points are color coded based on the neighborhood group feature. (5 Points)
# b. Now again, plot a scatter plot based on these coordinates, where the points represent the
# location of an Airbnb, and the points are color coded based on the price of the particular Airbnb,
# where price of the listing is less than 1000. Looking at the graph can you tell which
# neighborhood group is the most expensive. (5 Points)

# %%
# Part A
sb.scatterplot(data=df, x="latitude", y="longitude", hue="neighbourhood_group").set_title("Airbnb Listings Based on Neighborhood Groups")
plt.ylim(-74.3, -73.7)
plt.xlim(40.4, 41)

# %%
# Part B
# After looking at the graph, it is clear that the most expensive location seems to be Manhattan as the area where the most expensive
# listings are correspond with where Manhattan is, as seen in the previous scatter plot. 
pricedf = df.copy()
pricedf = pricedf.loc[pricedf['price'] < 1000].sort_values(by="price")
sb.scatterplot(data=pricedf, x="latitude", y="longitude", hue="price").set_title("Airbnb Listings Based on Price")
plt.ylim(-74.3, -73.7)
plt.xlim(40.4, 41)

# %%
# Task 5:
# Word clouds are useful tool to explore the text data. Extract the words from the name of the Airbnb
# and generate a word cloud.

# %%
list = df['name'].to_frame().stack().str.split("[^\w+]").explode().tolist()
# list2 = []
# for i in list:
#     list2.append(i.replace('\'', ""))
# # print(list[0])
# wordcloud = WordCloud().generate(str(list2))
wordcloud = WordCloud(width = 1000, height = 1000).generate(str(list))

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# %%
# Task 6: Find out which areas has the busiest (hosts with high number of listings) host? Are there any
# reasons, why these hosts are the busiest, considers factors such as availability, price, review, etc.?
# Bolster you reasoning with different plots and correlations. (10 Points)

# %%
# This scatter plot is made to see the locations where hosts have a lot of listings. The locations can be eyeballed with the
# graph scatter plots that were made in Part 4 in order to see the neighborhood groups based on the latitude and longitude of 
# the locations. 
busydf = df.copy()
busydf = busydf.loc[busydf['calculated_host_listings_count'] > 30].sort_values(by="calculated_host_listings_count")
sb.scatterplot(data=busydf, x="latitude", y="longitude", hue="calculated_host_listings_count").set_title("Airbnb Listings Based on Neighborhood Groups")
plt.ylim(-74.3, -73.7)
plt.xlim(40.4, 41)

# %%
# In order to see the correlation between the busiest locations, the listings with owners that had few listings were removed as a 
# lot of those listings wouldn't be super representative of the busiest areas. 

corrdf = df.copy()
corrdf = corrdf.loc[corrdf['calculated_host_listings_count'] > 30].sort_values(by=['price'])
# corrdf = corrdf.drop_duplicates(subset=['host_id'], keep='first')
corrdf = corrdf.drop(columns=['id', 'host_id', 'latitude', 'longitude'])
corrdf['neighbourhood_group'] = corrdf['neighbourhood_group'].map({'Staten Island':0, 'Bronx':1, 'Queens':2, 'Brooklyn':3, 'Manhattan':4})
corrdf['room_type'] = corrdf['room_type'].map({'Shared room':0, 'Private room':1, 'Entire home/apt':2 })

dataplot = sb.heatmap(corrdf.corr(), cmap="YlGnBu", annot=True).set_title("Correlations between Interesting Listing Features for Busiest Locations")
sb.set(rc = {'figure.figsize':(8,8)})
plt.show()

# %%
corrdf = df.copy()
corrdf = corrdf.loc[corrdf['calculated_host_listings_count'] > 30].sort_values(by=['price'])
corrdf = corrdf.drop(columns=['id', 'host_id', 'latitude', 'longitude'])
corrdf['neighbourhood_group'].value_counts(sort=False).plot.bar(y='neighbourhood_group', rot=0, title='Number of Listings for Busiest Hosts Based on Location')


# %%
corrdf.boxplot(column=['availability_365'], vert=False, figsize=(20,5)).set_title("Availability for Busiest Hosts Based on Location")

# %%
# Part 6 Analysis:
# After looking at the scatter plot and the previous scatter plot in Part 4, it seems that most of the busiest listings seem to
# be in Manhattan. This fact is later confirmed by the bar chart for the Number of Listings for Busiest Hosts Based on Location. 
# When looking at the correlation analysis for the busiest hosts, there are some numbers that stand out:
# 1. Correlation Coefficient between Neighborhood Group and Room Type = 0.66
# 2. Calculated Host Listings Count and Reviews Per Month = 0.7
# 3. Neighborhood Group and Price = 0.4
# 4. Room Type and Price = 0.48
# The coefficient between Neighborhood Group and Room Type is interesting because when thinking about it, whether a listing is shared, 
# private or the entire home/apartment does relate to the group. In the 5 boroughs, Manhattan can have both entire apartments 
# or private or shared rooms, while in places like Queens, you're probably getting the entire home/apartment.
# Another value that was very intriguing was the coefficient between Calculated Host Listings Count and Reviews per Month because
# it is telling us that with the busiest hosts, the more listings they have, the more reviews they are getting. 
# With the last two values, it makes sense that the location and room type affects how costly or cheap a listing is. 

# %%
# Task 7: Create two plots (at least one unique plot not used above) of your own using the dataset that you
# think reveals something very interesting. Explain what it is, and anything else you learned. (10 Points)

# %%
# The first unique plot that is created is a pie plot, that is used as a proportional representation of data in a column. 
# Pie plots take in numeric data and the neighborhood column is of a different type, so the number of occurrences were 
# counted in order to quantify this data. Since the number of listings are well over over 40000 listings, I wanted to 
# find the most popular neighborhoods and see which places have the most listings within those areas. I selected locations
# that have more than 1000 listings and saw that Williamsburg and Bedford-Stuyvesant, which are both in Brooklyn, were 
# the most popular neighborhoods in the data set. 
top5df = df['neighbourhood'].value_counts()
top5df = top5df[top5df > 1000]
top5df.plot.pie(autopct='%1.0f%%').set_title("Most Popular Neighborhoods Based on Airbnb Listings")

# %%
# Another element I wanted to reveal about the data set was the types of rooms that were in the Airbnb listings, based on the
# the location of the listing. I used a scatter plot to do so After creating the chart (and using a previous chart from Part 4 for the location based on 
# the latitude and longitude), I was able to see that in Bronx and in Brooklyn there were more private rooms, in Queens, 
# there was a mix between private rooms and entire homes/apartments, and in Manhattan there were mostly listings that 
# included the entire home/apartment.
sb.scatterplot(data=df, x="latitude", y="longitude", hue="room_type").set_title("Airbnb Listings Based on Room Type")
plt.ylim(-74.3, -73.7)
plt.xlim(40.4, 41)


