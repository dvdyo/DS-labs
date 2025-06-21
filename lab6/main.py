import marimo

__generated_with = "0.14.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    return go, make_subplots, np


@app.cell
def _(go, np):
    #1.1
    def create_dataset(n_points=100):
        b00 = np.random.randint(0, 10)
        b11 = np.random.randint(1, 10)
        xs = np.linspace(0, 10, n_points)
        eps = np.random.normal(0, 1.5, n_points)

        ys_obs = b00 + b11 * xs + eps
        ys_exact = b00 + b11 * xs
        return xs, ys_obs, ys_exact, b00, b11, eps

    x_data, y_noisy, y_true, b0, b1, noise_vector = create_dataset(100)

    print("True parameters:")
    print(">>Intercept (b₀):", b0)
    print(">>Slope (b₁):", b1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_noisy, mode='markers', 
                            name='Noisy data', marker=dict(color='grey')))
    fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='lines', 
                            name='True line', line=dict(color='red')))
    fig.update_layout(
        title="Data generation around a given line",
        xaxis_title="x",
        yaxis_title="y",
        showlegend=True,
        width=700,
        height=400
    )
    fig.show()

    return b0, b1, noise_vector, x_data, y_noisy


@app.cell
def _(b0, b1, np, x_data, y_noisy):
    #1.2
    def lsq_regression(x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        b1_o = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        b0_o = y_mean - b1_o * x_mean
        return b0_o, b1_o

    b0_o, b1_o = lsq_regression(x_data, y_noisy)
    print("\nLeast squares estimates:")
    print("Intercept coefficient (b₀)", b0)
    print("Slope coefficient (b₁)", b1)
    print("Optimal intercept value (b₀):", b0_o)
    print("Optimal slope coefficient value (b₁):", b1_o)
    return b0_o, b1_o


@app.cell
def _(b0, b0_o, b1, b1_o, np, x_data, y_noisy):
    #1.3
    poly_coefs = np.polyfit(x_data, y_noisy, 1)
    est_b1_poly = poly_coefs[0]
    est_b0_poly = poly_coefs[1]

    print("\nMethod comparison:")
    print(">>>True parameters:         b₀ =", b0, ", b₁ =", b1)
    print(">>>LSQ:                     b₀ =", b0_o, ", b₁ =", b1_o)
    print(">>>np.polyfit:              b₀ =", est_b0_poly, ", b₁ =", est_b1_poly)
    return est_b0_poly, est_b1_poly


@app.cell
def _(b0, b0_o, b1, b1_o, est_b0_poly, est_b1_poly, go, noise_vector, x_data):
    #1.4
    def plot_regression_models(x, eps, b0, b1, lsq_params, poly_params):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=b0 + b1 * x + eps, mode='markers',
                                name='Noisy data', marker=dict(color='grey', opacity=0.65)))

        fig.add_trace(go.Scatter(x=x, y=b0 + b1 * x, mode='lines',
                                name='True line', line=dict(color='red')))

        fig.add_trace(go.Scatter(x=x, y=lsq_params[0] + lsq_params[1] * x, mode='lines',
                                name='LSQ estimate', line=dict(color='yellow')))

        fig.add_trace(go.Scatter(x=x, y=poly_params[0] + poly_params[1] * x, mode='lines',
                                name='np.polyfit estimate', line=dict(color='cyan', dash='dash')))

        fig.update_layout(
            title="Comparison of regression line estimates",
            xaxis_title="x",
            yaxis_title="y",
            showlegend=True,
            width=700,
            height=400
        )
        fig.show()

    plot_regression_models(x_data, noise_vector, b0, b1, (b0_o, b1_o), (est_b0_poly, est_b1_poly))
    return


@app.cell
def _(np, x_data, y_noisy):
    #2.1
    def gradient_descent(x, y, lr=0.001, num_iter=100000):
        n = len(y)
        cur_b0 = 0.0
        cur_b1 = 0.0
        for i in range(num_iter):
            preds = cur_b0 + cur_b1 * x
            error = y - preds
            grad_int = (-2/n) * np.sum(error)
            grad_slp = (-2/n) * np.sum(x * error)
            cur_b0 -= lr * grad_int
            cur_b1 -= lr * grad_slp
        return cur_b0, cur_b1

    b0_gd, b1_gd = gradient_descent(x_data, y_noisy, lr=0.001, num_iter=100000)
    print("\nGradient descent:")
    print(">>>Optimal intercept value (b₀):", b0_gd)
    print(">>>Optimal slope value (b₁):", b1_gd)
    return b0_gd, b1_gd, gradient_descent


@app.cell
def _(
    b0,
    b0_gd,
    b0_o,
    b1,
    b1_gd,
    b1_o,
    est_b0_poly,
    est_b1_poly,
    go,
    noise_vector,
    x_data,
):
    #2.2
    def plot_all_estimations(x, eps, b0, b1, lsq_params, poly_params, gd_params):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=b0 + b1 * x + eps, mode='markers',
                                name='Noisy data', marker=dict(color='grey', opacity=0.7)))

        fig.add_trace(go.Scatter(x=x, y=b0 + b1 * x, mode='lines',
                                name='True line', line=dict(color='red')))

        fig.add_trace(go.Scatter(x=x, y=lsq_params[0] + lsq_params[1] * x, mode='lines',
                                name='LSQ estimate', line=dict(color='yellow')))

        fig.add_trace(go.Scatter(x=x, y=poly_params[0] + poly_params[1] * x, mode='lines',
                                name='np.polyfit estimate', line=dict(color='cyan', dash='dash')))

        fig.add_trace(go.Scatter(x=x, y=gd_params[0] + gd_params[1] * x, mode='lines',
                                name='Gradient descent', line=dict(color='green', dash='dot')))

        fig.update_layout(
            title="Combined plot of regression estimates",
            xaxis_title="x",
            yaxis_title="y",
            showlegend=True,
            width=700,
            height=400
        )
        fig.show()

    plot_all_estimations(x_data, noise_vector, b0, b1,
                         (b0_o, b1_o),
                         (est_b0_poly, est_b1_poly),
                         (b0_gd, b1_gd))
    return


@app.cell
def _(go, gradient_descent, make_subplots, np, x_data, y_noisy):
    #2.3
    def plot_error_vs_iters(x, y, lr=0.001):
        iter_range = range(100, 1000, 100)
        mse_list = []
        mae_list = []
        for num in iter_range:
            int_est, slp_est = gradient_descent(x, y, lr, num)
            pred_vals = int_est + slp_est * x
            mse_err = np.mean((y - pred_vals) ** 2)
            mae_err = np.mean(np.abs(y - pred_vals))
            mse_list.append(mse_err)
            mae_list.append(mae_err)   

        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Mean Squared Error vs Iterations", 
                                         "Mean Absolute Error vs Iterations"))

        fig.add_trace(go.Scatter(x=list(iter_range), y=mse_list, mode='lines+markers',
                                name='MSE', line=dict(color='blue')), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(iter_range), y=mae_list, mode='lines+markers',
                                name='MAE', line=dict(color='red')), row=1, col=2)

        fig.update_xaxes(title_text="Iterations", row=1, col=1)
        fig.update_xaxes(title_text="Iterations", row=1, col=2)
        fig.update_yaxes(title_text="Mean Squared Error", row=1, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=2)

        fig.update_layout(height=400, width=800, showlegend=True)
        fig.show()

        return mse_list, mae_list

    mse_values, mae_values = plot_error_vs_iters(x_data, y_noisy, lr=0.001)
    return


@app.cell
def _(b0, b0_gd, b0_o, b1, b1_gd, b1_o, est_b0_poly, est_b1_poly):
    #2.4
    print("\nComparative summary:")
    print(">>>True values:                   b₀ =", b0, " , b₁ =", b1)
    print(">>>Least squares method (LSQ):    b₀ =", b0_o, " , b₁ =", b1_o)
    print(">>>np.polyfit:                    b₀ =", est_b0_poly, " , b₁ =", est_b1_poly)
    print(">>>Gradient descent:              b₀ =", b0_gd, " , b₁ =", b1_gd)
    return


if __name__ == "__main__":
    app.run()
