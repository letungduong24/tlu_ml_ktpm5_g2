import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Xử lí dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def createModel():
    lasso = joblib.load('models/lasso_model.pkl')
    linear = joblib.load('models/linear_model.pkl')
    mlp = joblib.load('models/mlp_model.pkl')
    stacking_regressor = joblib.load('models/stacking_model.pkl')
    return lasso, linear, mlp, stacking_regressor

def predict(linear, mlp, lasso, stacking):
    pred_linear = linear.predict(X_test)
    pred_mlp = mlp.predict(X_test)
    pred_lasso = lasso.predict(X_test)
    pred_stacking = stacking.predict(X_test)
    return pred_lasso, pred_linear, pred_mlp, pred_stacking

def predictFromUser(linear, mlp, lasso, stacking, new_data):
    pred_linear = linear.predict(new_data)[0]
    pred_mlp = mlp.predict(new_data)[0]
    pred_lasso = lasso.predict(new_data)[0]
    pred_stacking = stacking.predict(new_data)[0]
    return pred_lasso, pred_linear, pred_mlp, pred_stacking

def mse(pred_linear, pred_mlp, pred_lasso, pred_stacking):
    mse_linear = mean_squared_error(y_test, pred_linear)
    mse_lasso = mean_squared_error(y_test, pred_lasso)
    mse_mlp = mean_squared_error(y_test, pred_mlp)
    mse_stacking = mean_squared_error(y_test, pred_stacking)
    return mse_linear, mse_lasso, mse_mlp, mse_stacking

def r2(pred_linear, pred_mlp, pred_lasso, pred_stacking):
    r2_linear = r2_score(y_test, pred_linear)
    r2_lasso = r2_score(y_test, pred_lasso)
    r2_mlp = r2_score(y_test, pred_mlp)
    r2_stacking = r2_score(y_test, pred_stacking)
    return r2_linear, r2_lasso, r2_mlp, r2_stacking

def mae(pred_linear, pred_mlp, pred_lasso, pred_stacking):
    mae_linear = mean_absolute_error(y_test, pred_linear)
    mae_lasso = mean_absolute_error(y_test, pred_lasso)
    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    mae_stacking = mean_absolute_error(y_test, pred_stacking)
    return mae_linear, mae_lasso, mae_mlp, mae_stacking

def graph(pred_linear, pred_mlp, pred_lasso, pred_stacking):
    errors_lasso = y_test - pred_lasso
    errors_linear = y_test - pred_linear
    errors_mlp = y_test - pred_mlp
    errors_stacking = y_test - pred_stacking
    plt.figure(figsize=(24, 8))
    plt.figure(figsize=(24, 8))

    plt.subplot(1, 4, 1)
    plt.hist(errors_lasso, bins=50, edgecolor='k')
    plt.title('Lasso - Phân phối sai số')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    plt.subplot(1, 4, 2)
    plt.hist(errors_linear, bins=50, edgecolor='k')
    plt.title('Linear Regression - Phân phối sai số')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    plt.subplot(1, 4, 3)
    plt.hist(errors_mlp, bins=50, edgecolor='k')
    plt.title('MLP - Phân phối sai số')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    plt.subplot(1, 4, 4)
    plt.hist(errors_stacking, bins=50, edgecolor='k')
    plt.title('Stacking - Phân phối sai số')
    plt.xlabel('Khoảng sai số')
    plt.ylabel('Số lượng sai số')

    plt.tight_layout()
    # Lưu đồ thị phân phối sai số
    plt.savefig('static/error_distribution.png')
    plt.close()

    # Đồ thị so sánh giá trị thực và dự đoán
    plt.figure(figsize=(24, 6))

    plt.subplot(1, 4, 1)
    plt.scatter(y_test, pred_lasso, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Lasso - Đồ thị')
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')

    plt.subplot(1, 4, 2)
    plt.scatter(y_test, pred_linear, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Linear Regression - Đồ thị')
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')

    plt.subplot(1, 4, 3)
    plt.scatter(y_test, pred_mlp, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('MLP - Đồ thị')
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')

    plt.subplot(1, 4, 4)
    plt.scatter(y_test, pred_stacking, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Stacking - Đồ thị')
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')

    plt.tight_layout()
    # Lưu đồ thị so sánh giá trị thực và dự đoán
    plt.savefig('static/predict_distributton.png')
    plt.close()
