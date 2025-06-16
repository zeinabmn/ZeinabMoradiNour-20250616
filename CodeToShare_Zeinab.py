# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:19:16 2025

@author: zeinabmn
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Utility Functions
# -----------------------------
def base36_to_decimal(value):
    try:
        return int(value, 36)
    except (ValueError, TypeError):
        return None

def separate_code_and_definition(df, column):
    df[column] = df[column].astype(str)
    df[[f"{column}_code", f"{column}_def"]] = df[column].str.split(":", n=1, expand=True)
    df[f"{column}_code"] = df[f"{column}_code"].str.strip()
    df[f"{column}_def"] = df[f"{column}_def"].str.strip()
    return df

# -----------------------------
# Data Loading & Cleaning
# -----------------------------
df_errands = pq.read_table('errands.parquet').to_pandas().drop_duplicates().dropna()
df_orders = pq.read_table('orders.parquet').to_pandas().drop_duplicates().dropna()

# Convert base-36 order_number to decimal
df_errands['order_number_dec'] = df_errands['order_number'].apply(base36_to_decimal)
df_errands['order_number_dec'] = df_errands['order_number_dec'].astype(int)
df_orders['order_id'] = df_orders['order_id'].astype(int)

# Join errands and orders on order_id
df_merged = df_orders.merge(df_errands, left_on='order_id', right_on='order_number_dec', how='inner')

# Filter valid entries using colons and remove malformed data
for col in ['errand_category', 'errand_type', 'errand_action', 'errand_channel']:
    df_merged = df_merged[
        df_merged[col].str.contains(":", na=False) & 
        ~df_merged['errand_action'].str.contains(r"\d+\.", na=False)
    ]

# Count contacts per order
df_merged['num_contacts'] = df_merged.groupby('order_id')['order_id'].transform('count')
df_merged = df_merged[df_merged['num_contacts'] < 100].dropna(subset=['num_contacts'])

# Split category/type/action/channel codes and definitions
for col in ['errand_category', 'errand_type', 'errand_action', 'errand_channel']:
    df_merged = separate_code_and_definition(df_merged, col)

# -----------------------------
# Contact Distribution & Pie Chart
# -----------------------------
distribution_df = df_merged['num_contacts'].value_counts().sort_index().reset_index()
distribution_df.columns = ['num_contacts', 'num_orders']
distribution_df['percentage'] = (distribution_df['num_orders'] / distribution_df['num_orders'].sum()) * 100
distribution_df.to_excel("contact_distribution.xlsx", index=False)

plt.figure(dpi=300)
plt.hist(df_merged['num_contacts'], bins=range(1, df_merged['num_contacts'].max() + 2))
plt.title("Distribution of Customer Contacts per Order")
plt.xlabel("Number of Contacts", fontsize=12)
plt.ylabel("Number of Orders", fontsize=12)
plt.tight_layout()
plt.savefig("orders_per_customer_distribution.png", dpi=300)
plt.show()

sizes = distribution_df['num_orders'].values
total = sizes.sum()
labels = [f"{i} contacts" if float(v) / total > 0.04 else "" for i, v in zip(distribution_df['num_contacts'], sizes)]

plt.figure(figsize=(6, 6), dpi=300)
plt.pie(
    sizes,
    labels=labels,
    autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
    startangle=140,
    wedgeprops={'edgecolor': 'white'},
    colors=plt.cm.Blues_r(sizes / sizes.max())
)
plt.title("Percentage Distribution of Customer Contacts per Order")
plt.tight_layout()
plt.show()

# -----------------------------
# Revenue Ratio Correlation
# -----------------------------
df_merged['revenue_ratio'] = df_merged['Revenue'] / df_merged['Order_Amount']
df_filtered = df_merged[(df_merged['revenue_ratio'] >= -1) & (df_merged['revenue_ratio'] < 2)]

corr = df_filtered['revenue_ratio'].corr(df_filtered['num_contacts'])
print(f"Correlation: {corr:.3f}")

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(df_filtered['revenue_ratio'], df_filtered['num_contacts'], '.', alpha=0.3)
plt.title("Number of Contacts vs Revenue Ratio", fontsize=18)
plt.xlabel("Revenue Ratio (Revenue / Ticket Price)", fontsize=18)
plt.ylabel("Number of Contacts", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# Focus Area Deep Dive
# -----------------------------
threshold = 3
df_merged['Contact Volume'] = df_merged['num_contacts'].apply(
    lambda x: f'>{threshold} Contacts' if x > threshold else f'≤{threshold} Contacts'
)

focus_categories = ["Rebooking", "Cancellation / refund", "Schedule change"]
df_focus = df_merged[df_merged['errand_category_def'].isin(focus_categories)]

# Channel usage in focus categories
channel_mix = df_focus.groupby(['errand_category_def', 'errand_channel_def']).size().reset_index(name='count')
pivot = channel_mix.pivot(index='errand_category_def', columns='errand_channel_def', values='count').fillna(0)
pivot_percent = pivot.div(pivot.sum(axis=1), axis=0)

plt.figure(dpi=300)
pivot_percent.plot(kind='barh', stacked=True, figsize=(10, 6), colormap='Set2')
plt.title("Channel Distribution by Category (Normalized %)", fontsize=12)
plt.xlabel("Percentage", fontsize=12)
plt.ylabel("Errand Category", fontsize=12)
plt.tight_layout()
plt.show()

# 1-contact vs Multi-contact actions
resolved_1_contact = df_focus[df_focus['num_contacts'] == 1].groupby(
    ['errand_category_def', 'errand_channel_def']
).size().reset_index(name='count')
total_per_category = resolved_1_contact.groupby('errand_category_def')['count'].transform('sum')
resolved_1_contact['percentage'] = resolved_1_contact['count'] / total_per_category

plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(
    data=resolved_1_contact,
    x='count',
    y='errand_category_def',
    hue='errand_channel_def',
    palette='Set2'
)
plt.title("1-Contact Resolved Cases by Category and Channel")
plt.tight_layout()
plt.show()

# Action frequency
top_actions = df_focus.groupby('errand_action_def').size().reset_index(name='count').sort_values('count', ascending=False).head(10)
plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(data=top_actions, y='errand_action_def', x='count', palette='Blues_d')
plt.title("Top 10 Most Frequent Errand Actions in Focus Area")
plt.tight_layout()
plt.show()

# Contact type analysis
df_focus['contact_type'] = df_focus['num_contacts'].apply(lambda x: '1-contact' if x == 1 else 'multi-contact')
action_summary = df_focus.groupby(['errand_action_def', 'contact_type']).size().reset_index(name='count')
pivot = action_summary.pivot(index='errand_action_def', columns='contact_type', values='count').fillna(0)

pivot = pivot.sort_values(by='multi-contact', ascending=False).head(10)
plt.figure(figsize=(12, 6), dpi=300)
pivot.plot(kind='barh', stacked=True, figsize=(10, 6), colormap='Paired')
plt.title("Top Errand Actions: 1-Contact vs Multi-Contact")
plt.xlabel("Number of Cases")
plt.ylabel("Errand Action")
plt.tight_layout()
plt.show()


# -----------------------------
# Errand category
# -----------------------------
type_summary = df_merged.groupby('errand_category_def')['num_contacts'].agg(['count', 'mean', 'median']).reset_index()
type_summary = type_summary.sort_values('mean', ascending=False)

plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x='mean', y='errand_category_def', data=type_summary, palette='Blues_d')
plt.title("Average Number of Contacts Errand Category", fontsize=18)
plt.tight_layout()
plt.savefig("avg_contacts_by_errand_Category_bigger.png", dpi=300)
plt.show()

# Sort by volume
type_summary = type_summary.sort_values('count', ascending=False)
plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x='count', y='errand_category_def', data=type_summary, palette='Blues_d')
plt.title("Volume of Customer Errands by Category", fontsize=18)
plt.tight_layout()
plt.savefig("errand_volume_by_Category_bigger.png", dpi=300)
plt.show()


# -----------------------------
# Additional Channel/Action Stats
# -----------------------------
type_summary = df_merged.groupby('errand_action_def')['num_contacts'].agg(['count', 'mean', 'median']).reset_index().head(10)
type_summary = type_summary.sort_values('mean', ascending=False)

plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x='mean', y='errand_action_def', data=type_summary, palette='Blues_d')
plt.title("Average Number of Contacts by Top 10 Errand Actions", fontsize=18)
plt.tight_layout()
plt.savefig("avg_contacts_by_errand_Action_bigger.png", dpi=300)
plt.show()

# Sort by volume
type_summary = type_summary.sort_values('count', ascending=False)
plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x='count', y='errand_action_def', data=type_summary, palette='Blues_d')
plt.title("Volume of Customer Errands by Action", fontsize=18)
plt.tight_layout()
plt.savefig("errand_volume_by_action_bigger.png", dpi=300)
plt.show()

# -----------------------------
# Channel Stats (High Contact Only)
# -----------------------------
type_summary = df_merged[df_merged['num_contacts'] > threshold].groupby('errand_channel_def')['num_contacts'].agg(['count', 'mean', 'median']).reset_index()
type_summary = type_summary.sort_values('mean', ascending=False)

plt.figure(figsize=(12, 18), dpi=300)
sns.barplot(x='mean', y='errand_channel_def', data=type_summary, palette='Blues_d')
plt.title("Avg Contacts per Order by Errand Channel", fontsize=18)
plt.tight_layout()
plt.savefig("avg_contacts_by_errand_channel_bigger.png", dpi=300)
plt.show()

# Volume
plt.figure(figsize=(12, 18), dpi=300)
type_summary = type_summary.sort_values('count', ascending=False)
sns.barplot(x='count', y='errand_channel_def', data=type_summary, palette='Blues_d')
plt.title("Volume of Customer Errands by Channel", fontsize=18)
plt.tight_layout()
plt.savefig("errand_volume_by_channel_bigger.png", dpi=300)
plt.show()

# -----------------------------
# Category vs Channel Heatmap
# -----------------------------
pivot_table = df_merged.pivot_table(
    index='errand_category_def', 
    columns='errand_channel', 
    aggfunc='size', 
    fill_value=0
)
category_channel_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0)

plt.figure(figsize=(16, 10), dpi=300)
sns.heatmap(category_channel_percent, annot=True, fmt=".1%", cmap="Blues")
plt.title("Channel Usage per Errand Category", fontsize=18)
plt.tight_layout()
plt.savefig("category_vs_channel_heatmap.png", dpi=300)
plt.show()
