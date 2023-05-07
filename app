now = datetime.now()

# Set the title of the app
st.set_page_config(page_title="Stock Price App", page_icon=":chart_with_upwards_trend:")
st.title("Stock Price App :chart_with_upwards_trend:")

# Create a sidebar for user inputs
st.sidebar.header("User Inputs")

# Create a text input for the stock symbol
symbol = st.sidebar.text_input("Enter a stock symbol", "AAPL")

# Create a date input for the start and end date
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("2021-01-01"))

# Add the dropdown menu for selecting the trading strategy
strategy_options = ["Moving Average Crossover", "RSI Overbought/Oversold"]
selected_strategy = st.sidebar.selectbox("Select a trading strategy", strategy_options)

# Add a text input for setting the starting investment amount
starting_investment = st.sidebar.number_input("Enter the starting investment amount", min_value=0.0, value=10000.0, step=1000.0)

# Fetch the stock data from Yahoo Finance
try:
    data = yf.download(symbol, start_date, end_date)
except KeyError:
    st.error(f"Error downloading data for {symbol}. Please try again later.")
    data = pd.DataFrame()  # Return an empty DataFrame if an error occurs



# Set up the Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets", api_version="v2")

# Configure News API
newsapi = NewsApiClient(api_key=NEWS_API_KEY)


def backtest_strategy(data, buy_signal, sell_signal, initial_balance=10000):
    balance = initial_balance
    position = 0

    for i in range(len(data)):
        if buy_signal.iloc[i] and balance > 0:
            position = balance / data.iloc[i]  # Update this line
            balance = 0
        elif sell_signal.all() and position > 0:
            balance = position * data.iloc[i]
            position = 0
    final_balance = balance + position * data.iloc[-1]
    return final_balance


def moving_average_crossover_strategy(data, initial_balance=10000, short_window=50, long_window=200):
    # Calculate the short and long moving averages
    data['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    data['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Generate buy and sell signals
    data['signal'] = 0.0
    data.loc[data['short_mavg'] > data['long_mavg'], 'signal'] = 1.0
    data.loc[data['short_mavg'] < data['long_mavg'], 'signal'] = -1.0

    buy_signal = data['signal'] == 1.0
    sell_signal = data['signal'] == -1.0

    # Calculate positions and balance
    balance = initial_balance
    position = 0
    for i in range(len(data)):
        # Buy signal
        if data['signal'].iloc[i] == 1.0 and balance > 0:
            position = balance / data['Close'].iloc[i]
            balance = 0
        # Sell signal
        elif data['signal'].iloc[i] == -1.0 and position > 0:
            balance = position * data['Close'].iloc[i]
            position = 0

    # Calculate final balance
    final_balance = backtest_strategy(data, buy_signal, sell_signal, initial_balance=initial_balance)

    return final_balance


def rsi_overbought_oversold_strategy(data, initial_balance=10000, rsi_period=14, overbought_threshold=70,
                                     oversold_threshold=30):
    # Calculate the RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Generate buy and sell signals
    data['signal'] = 0.0
    data.loc[data['RSI'] < oversold_threshold, 'signal'] = 1.0
    data.loc[data['RSI'] > overbought_threshold, 'signal'] = -1.0

    # Calculate positions and balance
    balance = initial_balance
    position = 0
    for i in range(len(data)):
        # Buy signal
        if data['signal'].iloc[i] == 1.0 and balance > 0:
            position = balance / data['Close'].iloc[i]
            balance = 0
        # Sell signal
        elif data['signal'].iloc[i] == -1.0 and position > 0:
            balance = position * data['Close'].iloc[i]
            position = 0

    # Calculate final balance
    final_balance = balance + position * data['Close'].iloc[-1]

    return final_balance


def run_strategy(stock_symbol, start_date, end_date, strategy, initial_investment):
    if data.empty:
        return data, 0


    if strategy == "Moving Average Crossover":
        data['short_mavg'] = data['Close'].rolling(window=50, min_periods=1, center=False).mean()
        data['long_mavg'] = data['Close'].rolling(window=200, min_periods=1, center=False).mean()
        buy_signal = data['short_mavg'] > data['long_mavg']
        sell_signal = data['short_mavg'] < data['long_mavg']
        final_balance = backtest_strategy(data, buy_signal, sell_signal, initial_investment)
    elif strategy == "RSI Overbought/Oversold":
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()

        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        buy_signal = data['RSI'] < 30
        sell_signal = data['RSI'] > 70
        final_balance = backtest_strategy(data, buy_signal, sell_signal, initial_investment)

    return data, final_balance

def fetch_news_sentiment(stock_symbol, num_articles=100):
    # Fetch news articles related to the stock symbol
    articles = newsapi.get_everything(q=stock_symbol,
                                          sort_by='publishedAt',
                                          language='en',
                                          page_size=num_articles)

    # Fetch sentiment scores from news articles
    news_sentiment_scores = fetch_news_sentiment(stock_symbol)

    # Calculate sentiment scores for each article
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for article in articles['articles']:
        score = analyzer.polarity_scores(article['title'])
        scores.append(score['compound'])

    # Calculate average sentiment score
    if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
    else:
            avg_score = 0

            # Align the features and labels
            features = data.drop('Close', axis=1)
            labels = data['Close'].shift(-1)
            labels.fillna(method='ffill', inplace=True)

            # Calculate average sentiment score
            if len(scores) > 0:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0

            # Merge the sentiment scores
            sentiment_scores = pd.Series(avg_score, index=data.index)
            features['sentiment'] = sentiment_scores
            features.fillna(0, inplace=True)



    return avg_score


    # Preprocess the data
    data.fillna(0, inplace=True)
    features.dropna(inplace=True)
    labels.dropna(inplace=True)
    features.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)


def train_and_evaluate_model(features, labels):
    # Check if the lengths of the variables match
    if len(features) == len(labels):
        print("The lengths of the features and labels variables match.")
    else:
        print("The lengths of the features and labels variables do not match. Please check your preprocessing steps.")
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # Step 1: Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

    # Step 2: Train the model using a RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 3: Make predictions on the testing set using the trained model
    y_pred = model.predict(X_test)

    # Step 4: Evaluate the model's performance by calculating the Mean Absolute Error (MAE)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)

    return mae, y_pred, y_test

    print(f"Mean Absolute Error: {mae}")

    predicted_moving_average = pd.Series(y_pred).rolling(window=5).mean()
    buy_signal = (predicted_moving_average.reset_index(drop=True) > y_test.reset_index(drop=True)) & (
            predicted_moving_average.reset_index(drop=True).shift(1) <= y_test.reset_index(drop=True).shift(1))
    sell_signal = (predicted_moving_average.reset_index(drop=True) < y_test.reset_index(drop=True)) & (
            predicted_moving_average.reset_index(drop=True).shift(1) >= y_test.reset_index(drop=True).shift(1))

    # Display the model's performance
    st.write(f"Mean Absolute Error: {mae}")

    # Display the trading signals
    signal_data = pd.DataFrame({'Buy Signal': buy_signal, 'Sell Signal': sell_signal})
    st.dataframe(signal_data)

    # Display the percentage return
    st.write(f"Percentage Return: {percentage_return:.2f}%")

# Plot the predicted closing prices
def plot_closing_prices(data, stock_symbol):
     fig, ax = plt.subplots()
     ax.plot(data.index, data['Close'], label='Close')
     ax.set_xlabel('Date')
     ax.set_ylabel('Closing price ($)')
     ax.set_title(f'{stock_symbol} Closing Prices')
     ax.legend()
     st.pyplot(fig)

# Define the target variable (label) - Future stock price
labels = data['Close'].shift(-1)  # Shift the 'Close' column up by 1 to get the future stock price

# Define the input variables (features) - Historical data, technical indicators, sentiment scores, etc.
features = data.drop('Close', axis=1)  # Remove the 'Close' column from the dataset


if __name__ == "__main__":
    # Use the user input for the stock symbol
    stock_symbol = symbol
    start_date = start_date
    end_date = end_date
    data, percentage_return = run_strategy(stock_symbol, start_date, end_date, selected_strategy, starting_investment)

    columns_to_drop = ['RSI', 'signal']
    features = data.drop([col for col in columns_to_drop if col in data.columns], axis=1)

    # Display the data as a table
    st.dataframe(data)

    # Plot the closing prices
    plot_closing_prices(data, stock_symbol)

    # Define the target variable (label) - Future stock price
    labels = data['Close'].shift(-1)  # Shift the 'Close' column up by 1 to get the future stock price

    # Drop rows with NaN values in features and labels
    features.dropna(inplace=True)
    labels.dropna(inplace=True)

    # Reset the index of features and labels dataframes
    features.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    # Train and evaluate the model
    mae, y_pred, y_test = train_and_evaluate_model(features, labels)

    # Display the model's performance
    st.write(f"Mean Absolute Error: {mae}")

    # Display the trading signals
    predicted_moving_average = pd.Series(y_pred).rolling(window=5).mean()
    buy_signal = (predicted_moving_average.reset_index(drop=True) > y_test.reset_index(drop=True)) & (
            predicted_moving_average.reset_index(drop=True).shift(1) <= y_test.reset_index(drop=True).shift(1))
    sell_signal = (predicted_moving_average.reset_index(drop=True) < y_test.reset_index(drop=True)) & (
            predicted_moving_average.reset_index(drop=True).shift(1) >= y_test.reset_index(drop=True).shift(1))

    signal_data = pd.DataFrame({'Buy Signal': buy_signal, 'Sell Signal': sell_signal})
    st.dataframe(signal_data)

    # Display the percentage return
    st.write(f"Percentage Return: {percentage_return:.2f}%")
