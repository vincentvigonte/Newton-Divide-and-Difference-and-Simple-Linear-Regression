import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def proterm(i, value, x):
    pro = 1
    for j in range(i):
        pro *= (value - x[j])
    return pro

def dividedDiffTable(x, y, n):
    table = np.zeros((n, n))
    table[:, 0] = y
    for i in range(1, n):
        for j in range(n - i):
            table[j][i] = (table[j][i - 1] - table[j + 1][i - 1]) / (x[j] - x[i + j])
    return table

def applyFormula(value, x, table, n):
    sum_ = table[0][0]
    for i in range(1, n):
        sum_ += proterm(i, value, x) * table[0][i]
    return sum_

def linear_regression_forecast(x, y, k):
    known_indices = np.where(~np.isnan(y))[0]
    known_x = x[known_indices]
    known_y = y[known_indices]

    x_mean = np.mean(known_x)
    y_mean = np.mean(known_y)

    b1 = np.sum((known_x - x_mean) * (known_y - y_mean)) / np.sum((known_x - x_mean) ** 2)
    b0 = y_mean - b1 * x_mean

    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = b0 + b1 * x[i]

    max_x = max(x)
    future_x = np.array([max_x + i for i in range(1, k + 1)])

    future_y = b0 + b1 * future_x

    return future_x, future_y

def display_table(x, y):
    table = "|     x     |     y     |\n|-----------|-----------|\n"
    for xi, yi in zip(x, y):
        y_value = "None " if np.isnan(yi) else f"{yi:.2f}"
        table += f"|     {xi}     |   {y_value}   |\n"
    print(table)

def main():
    while True:
        print("Options:")
        print("[1] - Newton Divided-Difference Interpolation")
        print("[2] - Linear Regression Forecasting")
        print("[0] - Exit")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            break

        elif choice in [1, 2]:
            file_path = input("Location of your dataset: ")
            data = pd.read_csv(file_path)

            if choice == 1:
                x = data['x'].values
                y = data['y'].replace({"none": None}).astype(float).values

                print("Initial Data")
                display_table(x, y)

                known_indices = np.where(~np.isnan(y))[0]
                known_x = x[known_indices]
                known_y = y[known_indices]

                table = dividedDiffTable(known_x, known_y, len(known_x))

                for i in range(len(y)):
                    if np.isnan(y[i]):
                        y[i] = applyFormula(x[i], known_x, table, len(known_x))

                print("Interpolated Data")
                display_table(x, y)

                x_smooth = np.linspace(min(x), max(x), 500)
                y_smooth = make_interp_spline(x, y)(x_smooth)

                plt.plot(x, y, 'o-', label="Initial Data")
                plt.plot(x_smooth, y_smooth, '-', label="Interpolated Curve")
                plt.title("Newton Divided-Difference Interpolation")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid(True)
                plt.show()

            elif choice == 2:
                x = data['x'].values
                y = data['y'].replace({"none": None}).astype(float).values

                print("Initial Data")
                display_table(x, y)

                k = int(input("Enter the number of future points to forecast: "))
                future_x, future_y = linear_regression_forecast(x, y, k)

                print("Forecasted Data")
                all_x = np.concatenate([x, future_x])
                all_y = np.concatenate([y, future_y])
                display_table(all_x, all_y)

                x_smooth = np.linspace(min(x), max(x), 500)
                y_smooth = make_interp_spline(x, y)(x_smooth)

                plt.plot(x, y, 'o-', label="Initial Data")
                plt.plot(x_smooth, y_smooth, '-', label="Regression Line")
                plt.plot(future_x, future_y, 'o--', label="Forecasted Data")
                plt.title("Linear Regression Forecasting")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid(True)
                plt.show()

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
