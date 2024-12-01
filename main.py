import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
# Replace 'your_dataset.csv' with the actual path to your CSV file.
df = pd.read_csv('data.csv', header=None)

# Step 2: Convert the data to a list of transactions
transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()

# Step 3: Convert the transactions to a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Step 4: Run the Apriori algorithm with a minimum support of 0.5 (you can adjust this)
min_support = 0.5

frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)

# Step 5: Display frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Step 6: Generate association rules with minimum lift of 1.0 (you can adjust this)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


# Step 7: Display the association rules
print("\nAssociation Rules:")
print(rules)
