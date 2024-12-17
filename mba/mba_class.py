import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create dummy transaction data
np.random.seed(42)

def create_dummy_transactions(n_transactions=1000):
    """
    Create dummy transaction data with realistic shopping patterns
    """
    # Define products with categories
    products = {
        'Groceries': ['Milk', 'Bread', 'Eggs', 'Cheese', 'Yogurt'],
        'Beverages': ['Coffee', 'Tea', 'Juice', 'Soda'],
        'Snacks': ['Chips', 'Cookies', 'Chocolate', 'Nuts'],
        'Produce': ['Apples', 'Bananas', 'Tomatoes', 'Onions']
    }
    
    # Create empty list for transactions
    transactions = []
    
    for i in range(n_transactions):
        # Each transaction will have 2-8 items
        n_items = np.random.randint(2, 9)
        transaction = []
        
        # Add common pairs (to create realistic associations)
        if np.random.random() < 0.7:  # 70% chance
            if 'Bread' in transaction or len(transaction) == 0:
                transaction.extend(['Bread', 'Milk'])
        
        # Add random items
        all_products = [item for sublist in products.values() for item in sublist]
        while len(transaction) < n_items:
            item = np.random.choice(all_products)
            if item not in transaction:
                transaction.append(item)
        
        transactions.append(transaction)
    
    return transactions

# Create transactions and convert to DataFrame
transactions = create_dummy_transactions()
transaction_df = pd.DataFrame({
    'TransactionID': range(len(transactions)),
    'Items': transactions
})

# 2. Prepare data for analysis
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 3. Perform Market Basket Analysis
def perform_mba_analysis(df, min_support=0.01, min_confidence=0.3):
    """
    Perform market basket analysis and return frequent itemsets and rules
    """
    # Generate frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(df))
    
    print(rules)
    # Add lift ratio
    rules["lift_ratio"] = rules["lift"].apply(lambda x: "Strong" if x > 1 else "Weak")
    
    return frequent_itemsets, rules

# 4. Analysis and Visualization Functions
def plot_item_frequency(df):
    """
    Plot item frequency in transactions
    """
    item_freq = df.sum().sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    item_freq.plot(kind='barh')
    plt.title('Item Frequency in Transactions')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_rule_scatter(rules):
    """
    Plot scatter plot of rules by support and confidence
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rules, x="support", y="confidence", hue="lift_ratio", 
                    size="lift", sizes=(50, 400))
    plt.title('Rules Scatter Plot')
    plt.tight_layout()
    plt.show()

def analyze_top_rules(rules, metric="lift", top_n=10):
    """
    Display top rules based on specified metric
    """
    return rules.nlargest(top_n, metric)[['antecedents', 'consequents', 
                                        'support', 'confidence', 'lift']]

# 5. Perform Analysis
frequent_itemsets, rules = perform_mba_analysis(df)

# 6. Display Results
print("\n=== Top 10 Rules by Lift ===")
print(analyze_top_rules(rules))

# 7. Visualizations
plot_item_frequency(df)
plot_rule_scatter(rules)

# 8. Additional Analysis: Category-based patterns
def analyze_category_patterns(rules):
    """
    Analyze patterns between product categories
    """
    rules['rule_length'] = rules['antecedents'].apply(lambda x: len(x)) + \
                          rules['consequents'].apply(lambda x: len(x))
    
    print("\n=== Rule Length Distribution ===")
    print(rules['rule_length'].value_counts().sort_index())
    
    print("\n=== Average Lift by Rule Length ===")
    print(rules.groupby('rule_length')['lift'].mean().round(2))

analyze_category_patterns(rules)

# 9. Export results
rules.to_csv('market_basket_rules.csv', index=False)
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)