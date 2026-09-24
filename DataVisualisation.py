import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ladd the dataset containing company sales data
df = pd.read_csv("company_sales_data.csv")

# Preview the rows to ensure the dataset is loaded correctly
print(df.head())   # Quick overvieww of the first few rows

# Define consistent colors for each product for better readability across plots
product_colors = {
    "facecream": "purple",
    "facewash": "blue",
    "toothpaste": "green",
    "bathingsoap": "orange",
    "shampoo": "brown",
    "moisturizer": "pink"
}

# 1. Total profit per month (Line Plot)
plt.figure(figsize=(8,5))  # Adjust figure size for better clarity
plt.plot(
    df["month_number"],
    df["total_profit"],
    color="red",
    marker="o",
    linewidth=2
)
plt.xlabel("Month")
plt.ylabel("Total Profit")
plt.title("Total Profit per Month")
plt.grid(True)  # Add grid to improve readability of valuess
plt.show()

# Comment on your choice:
# A line plot is ideal to visualize trends over time, allowing a clear understanding of how total profit evolves month by month.
# Conclusion:
# The total profit demonstrates a steady increase throughout the year, with the highest values appearing towards the end, likely reflecting seasonal peaks in demand.

# 2. Sales of all products per month (Grouped Bar Chart)
plt.figure(figsize=(14,6))
months = df["month_number"].values
bar_width = 0.13
x = np.arange(len(months))  # Generate positions for each month group

# Plot each product as a separate bar for each month
for i, product in enumerate(product_colors):
    plt.bar(
        x + i*bar_width,  # Shift each bar horizontally to avoid overlapping
        df[product],
        width=bar_width,
        label=product.capitalize(),
        color=product_colors[product]
    )

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales of All Products per Month (Grouped Bar Chart)")
plt.xticks(x + bar_width*2.5, months)  # Center the month labels under each group
plt.legend()
plt.grid(axis='y')  # Only show horizontal grid lines for cleaner visualizaation
plt.show()

# Comment on your choice:
# Grouped bar charts allow comparison of multiple product sales side by side within each month.
# Using different colors and spacing improves clarity and makes smaller sales (like facewash) more visible.
# Conclusion:
# Toothpaste and bathing soap dominate sales, being daily essentials, while cosmetic products like facecream and moisturizer remain stable but lower.
# The clear separation of bars helps to identify trends for each product individually

# 3. Compare Face Cream vs. Toothpaste sales (Line Plot)
plt.figure(figsize=(8,5))
plt.plot(
    df["month_number"],
    df["facecream"],
    label="Face Cream",
    color=product_colors["facecream"],
    marker="o",
    linewidth=2
)
plt.plot(
    df["month_number"],
    df["toothpaste"],
    label="Toothpaste",
    color=product_colors["toothpaste"],
    marker="o",
    linewidth=2
)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Face Cream vs Toothpaste Sales")
plt.legend()
plt.grid(True)
plt.show()

# Comment on your choice:
# Line plots are perfect for comparing two series over time, highlighting trends and differences clearly.
# Conclusion:
# Toothpaste consistently outsells face cream throughout the year, emphasizing that essential hygiene products are prioritized over cosmetic items.

# 4. Distribution of total profits (Bar Plot)
plt.figure(figsize=(8,5))
plt.bar(df["month_number"], df["total_profit"], color='skyblue', edgecolor='black')
plt.xlabel("Month")
plt.ylabel("Profit")
plt.title("Profit per Month")
plt.grid(axis='y')  # Add horizontal grid lines for easier comparison of values
plt.show()

# Comment on your choice:
# A bar plot is appropriate to display individual monthly profit values, providing a straightforward visual comparison.
# Conclusion:
# Overall profits increase across the year, indicating stable business performance and growth.

# 5. Cumulative sales of all products (Stack Plot)
plt.figure(figsize=(10,6))
products = ["facecream","facewash","toothpaste","bathingsoap","shampoo","moisturizer"]
plt.stackplot(
    df["month_number"],
    df[products].T,  # Transpose to plot each product as a separate series
    labels=[p.capitalize() for p in products],
    colors=[product_colors[p] for p in products]
)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Cumulative Sales of All Products per Month")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# Comment on your choice:
# Stack plots provide a cumulative view while still allowing individual product contributions to be identified.
# Conclusion:
# The increasing total height of the stack illustrates overall sales growth.
# Toothpaste and bathing soap contribute the largest portions to cumulative sales, highlighting their market importance.
