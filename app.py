import streamlit as st
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define your existing variables and functions here
media_labels = ['Affiliate', 'Apple Search', 'Facebook', 'Google Brand', 'Google Search', 'Influencer', 'Pinterest', 'Snapchat', 'TikTok']
media_coef = [179.453, 77.9556, 359.589, 172.221, 204.032, 153.933, 24.0287, 21.4839, 45.6791]
media_const = [-1102.19, -458.212, -3460.35, -1044.24, -1809.58, -1018.58, -179.358, -135.677, -302.443]
max_budget = [30000*1.2, 7800*1.5, 335000*1.5, 24000*1.5, 146000*1.5, 121000*1.5, 4500*1.5, 11000*1.5, 26000*1.5]

# Add the plot_cost_curves_log function
def plot_cost_curves_log(media_labels, media_coef, media_const, media_budget, max_budget):
    fig = make_subplots(rows=3, cols=3, subplot_titles=media_labels)
    
    for i, (label, coef, const, max_bud) in enumerate(zip(media_labels, media_coef, media_const, max_budget)):
        row = i // 3 + 1
        col = i % 3 + 1
        
        spend = np.linspace(0, min(max_bud, media_budget), 1000)
        conversions = np.maximum(0, coef * np.log(spend + 1) + const)  # Add 1 to avoid log(0)
        
        fig.add_trace(
            go.Scatter(x=spend, y=conversions, name=label, mode='lines'),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Spend ($)", row=row, col=col, rangemode="tozero")
        fig.update_yaxes(title_text="Conversions", row=row, col=col, rangemode="tozero")

    fig.update_layout(height=900, width=1000, title_text="Cost Curves for Each Channel (Logarithmic Model)")
    return fig

# Add the run_optimizer function
def run_optimizer(total_budget, media_labels, media_coef, media_const, max_budget):
    def objective(x):
        return -np.sum([coef * np.log(spend + 1) + const for coef, const, spend in zip(media_coef, media_const, x)])

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget},
    ]

    bounds = [(0, max_bud) for max_bud in max_budget]

    initial_guess = [total_budget / len(media_labels)] * len(media_labels)

    result = sco.minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result

# Add the plot_cost_curves_with_optimal function
def plot_cost_curves_with_optimal(media_labels, media_coef, media_const, total_budget, max_budget, optimal_allocation):
    fig = make_subplots(rows=3, cols=3, subplot_titles=media_labels)
    
    for i, (label, coef, const, max_bud, opt_alloc) in enumerate(zip(media_labels, media_coef, media_const, max_budget, optimal_allocation)):
        row = i // 3 + 1
        col = i % 3 + 1
        
        spend = np.linspace(0, min(max_bud, total_budget), 1000)
        conversions = np.maximum(0, coef * np.log(spend + 1) + const)
        
        fig.add_trace(
            go.Scatter(x=spend, y=conversions, name=label, mode='lines'),
            row=row, col=col
        )
        
        opt_conversions = coef * np.log(opt_alloc + 1) + const
        fig.add_trace(
            go.Scatter(x=[opt_alloc], y=[opt_conversions], name=f'{label} Optimal',
                       mode='markers', marker=dict(size=10, color='red'),
                       text=f'Spend: ${int(opt_alloc):,}<br>Conversions: {int(opt_conversions):,}',
                       hoverinfo='text', showlegend=False),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Spend ($)", row=row, col=col, rangemode="tozero")
        fig.update_yaxes(title_text="Conversions", row=row, col=col, rangemode="tozero")

    fig.update_layout(height=900, width=1000, title_text="Cost Curves for Each Channel with Optimal Allocation")
    return fig

# Add the plot_combined_cost_curves function
def plot_combined_cost_curves(media_labels, media_coef, media_const, total_budget, max_budget, optimal_allocation):
    fig = go.Figure()
    
    for label, coef, const, max_bud, opt_alloc in zip(media_labels, media_coef, media_const, max_budget, optimal_allocation):
        spend = np.linspace(0, min(max_bud, total_budget), 1000)
        conversions = np.maximum(0, coef * np.log(spend + 1) + const)
        
        fig.add_trace(go.Scatter(x=spend, y=conversions, name=label, mode='lines'))
        
        opt_conversions = coef * np.log(opt_alloc + 1) + const
        fig.add_trace(
            go.Scatter(x=[opt_alloc], y=[opt_conversions], name=f'{label} Optimal',
                       mode='markers', marker=dict(size=10, color='red'),
                       text=f'{label}<br>Spend: ${int(opt_alloc):,}<br>Conversions: {int(opt_conversions):,}',
                       hoverinfo='text', showlegend=False)
        )

    fig.update_layout(
        title='Combined Cost Curves with Optimal Allocation',
        xaxis_title='Spend ($)',
        yaxis_title='Conversions',
        height=600, width=1000,
        xaxis=dict(rangemode="tozero"),
        yaxis=dict(rangemode="tozero")
    )
    return fig

# Add the plot_combined_cost_curves_log function
def plot_combined_cost_curves_log(media_labels, media_coef, media_const, media_budget, max_budget):
    fig = go.Figure()
    
    for label, coef, const, max_bud in zip(media_labels, media_coef, media_const, max_budget):
        spend = np.linspace(0, min(max_bud, media_budget), 1000)
        conversions = np.maximum(0, coef * np.log(spend + 1) + const)  # Add 1 to avoid log(0)
        
        fig.add_trace(go.Scatter(x=spend, y=conversions, name=label, mode='lines'))

    fig.update_layout(
        title='Combined Cost Curves for All Channels (Logarithmic Model)',
        xaxis_title='Spend ($)',
        yaxis_title='Conversions',
        height=600, width=1000,
        xaxis=dict(rangemode="tozero"),
        yaxis=dict(rangemode="tozero")
    )
    return fig

# Add this new function to create a bar chart
def plot_budget_allocation(media_labels, optimal_allocation):
    fig = go.Figure(data=[
        go.Bar(name='Budget Allocation', x=media_labels, y=optimal_allocation)
    ])
    fig.update_layout(
        title='Budget Allocation by Channel',
        xaxis_title='Channels',
        yaxis_title='Budget ($)',
        height=500, width=1000,
        yaxis=dict(rangemode="tozero")
    )
    return fig

# Define your existing functions here (model_function, plot_cost_curves_with_optimal, plot_combined_cost_curves_log, etc.)

# Streamlit app
st.title("Budget Optimizer")

# Show initial cost curves
st.subheader("Initial Cost Curves")
fig_initial = plot_cost_curves_log(media_labels, media_coef, media_const, 100000, max_budget)
st.plotly_chart(fig_initial)

# Show combined cost curves
st.subheader("Combined Cost Curves")
fig_combined_initial = plot_combined_cost_curves_log(media_labels, media_coef, media_const, 100000, max_budget)
st.plotly_chart(fig_combined_initial)

# User input for total budget
total_budget = st.number_input("Enter total budget ($)", min_value=1000, max_value=1000000, value=100000, step=1000)

# Button to run optimizer
if st.button("Run Optimizer"):
    # Run optimization
    solution = run_optimizer(total_budget, media_labels, media_coef, media_const, max_budget)
    
    # Display results
    st.subheader("Optimization Results")
    st.write(f"Total Budget: ${total_budget:,}")
    
    # Display allocation
    st.write("Budget Allocation:")
    for label, allocation in zip(media_labels, solution.x):
        st.write(f"- {label}: ${int(allocation):,} ({allocation/total_budget*100:.1f}%)")
    
    # Calculate and display CPA and Conversions
    cpa = total_budget / (-1 * solution.fun)
    conversions = int(-1 * solution.fun)
    st.write(f"CPA: ${cpa:.2f}")
    st.write(f"Conversions: {conversions:,}")
    
    # Add bar chart for budget allocation
    st.subheader("Budget Allocation Chart")
    fig_budget_allocation = plot_budget_allocation(media_labels, solution.x)
    st.plotly_chart(fig_budget_allocation)
    
    # Plot optimized cost curves
    st.subheader("Optimized Cost Curves")
    fig_optimized = plot_cost_curves_with_optimal(media_labels, media_coef, media_const, total_budget, max_budget, solution.x)
    st.plotly_chart(fig_optimized)
    
    # Plot combined cost curves with optimal allocation
    st.subheader("Combined Cost Curves with Optimal Allocation")
    fig_combined_optimal = plot_combined_cost_curves(media_labels, media_coef, media_const, total_budget, max_budget, solution.x)
    st.plotly_chart(fig_combined_optimal)
