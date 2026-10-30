# Nasdaq Signal Lab

A hackathon prototype exploring one question: **does the sentiment of financial headlines move with the Nasdaq-100's next-day return?**

## Open the demo

Open [index.html](./index.html) in a browser. It is a self-contained page with no installation or API key required.

## What is in this prototype

- An interactive-style research dashboard for the Nasdaq-100.
- A sample headline table with sentiment labels and next-day returns.
- A scatter plot and Pearson correlation calculation.
- A short explanation of the experiment and its limitations.

## Important: the current data is fictional

Every headline, sentiment score, and market return in the page is invented for demonstrating the interface. The displayed correlation is calculated from those invented rows and has no real-world meaning. The current version does not call an AI model, fetch live data, or test a trading strategy.

## Hackathon build plan

1. Replace the sample rows with a small, documented dataset of real headlines and Nasdaq-100 daily prices.
2. Use an AI model to label each headline as positive, neutral, or negative and return a score with a brief reason.
3. Match each headline to the next trading day's index return, using timestamps carefully to avoid look-ahead bias.
4. Compare the results with a simple baseline and explain that correlation does not prove prediction or causality.

Keep the first live version small: one index, a manageable date range, one consistent news source, and one clear baseline.

## Metric

For each observation, sentiment is a score from -1 to +1. Next-day return is the percent change from the prior close to the next trading day's close. The dashboard computes Pearson correlation between the two columns.

Do not present the prototype as investment advice or a reliable buy/sell signal.
