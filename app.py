from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

app = Flask(__name__)

# Import data, khai báo features và target
ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Xử lí dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Tạo mô hình và tối ưu
param_grid_lasso = {
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
}
# Lasso
baseLasso = Lasso()
lasso = GridSearchCV(baseLasso, param_grid_lasso, cv=3, n_jobs=-1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
lassoMAE = mean_absolute_error(y_test, y_pred_lasso)
lassoMSE = mean_squared_error(y_test, y_pred_lasso)
lassoR2 = r2_score(y_test, y_pred_lasso)
print(f'Lasso - MAE: {lassoMAE}, MSE: {lassoMSE}, R²: {lassoR2}')

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)
linearMAE = mean_absolute_error(y_test, y_pred_linear)
linearMSE = mean_squared_error(y_test, y_pred_linear)
linearR2 = r2_score(y_test, y_pred_linear)
print(f'Linear Regression - MAE: {linearMAE}, MSE: {linearMSE}, R²: {linearR2}')

# MLP Regressor with Early Stopping
mlp = MLPRegressor(
    hidden_layer_sizes=(50,),         # Kích thước lớp ẩn
    max_iter=500,                      # Số vòng lặp tối đa
    early_stopping=True,               # Bật early stopping
    validation_fraction=0.1,           # Tỷ lệ dữ liệu dùng để validation
    n_iter_no_change=10,               # Số vòng lặp không thay đổi hiệu suất trước khi dừng
    random_state=1                     # Hạt giống để tái lập kết quả
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mlpMAE = mean_absolute_error(y_test, y_pred_mlp)
mlpMSE = mean_squared_error(y_test, y_pred_mlp)
mlpR2 = r2_score(y_test, y_pred_mlp)
print(f'MLP - MAE: {mlpMAE}, MSE: {mlpMSE}, R²: {mlpR2}')

# Stacking Regressor
base_models = [
    ('linear', LinearRegression()),
    ('lasso', lasso.best_estimator_),
    ('mlp', mlp)
]
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), n_jobs=-1)  # Sử dụng n_jobs để chạy song song
stacking_regressor.fit(X_train, y_train)
y_pred_stacking = stacking_regressor.predict(X_test)
stackingMAE = mean_absolute_error(y_test, y_pred_stacking)
stackingMSE = mean_squared_error(y_test, y_pred_stacking)
stackingR2 = r2_score(y_test, y_pred_stacking)
print(f'Stacking Regressor - MAE: {stackingMAE}, MSE: {stackingMSE}, R²: {stackingR2}')


# Vẽ đồ thị
errors_lasso = y_test - y_pred_lasso
errors_linear = y_test - y_pred_linear
errors_mlp = y_test - y_pred_mlp
errors_stacking = y_test - y_pred_stacking

plt.figure(figsize=(24, 8))

# Đồ thị phân phối sai số
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
plt.scatter(y_test, y_pred_lasso, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Lasso - Đồ thị')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')

plt.subplot(1, 4, 2)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Linear Regression - Đồ thị')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')

plt.subplot(1, 4, 3)
plt.scatter(y_test, y_pred_mlp, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('MLP - Đồ thị')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')

plt.subplot(1, 4, 4)
plt.scatter(y_test, y_pred_stacking, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Stacking - Đồ thị')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')

plt.tight_layout()
# Lưu đồ thị so sánh giá trị thực và dự đoán
plt.savefig('static/predict_distributton.png')
plt.close()


# Route để hiển thị form nhập liệu
@app.route('/')
def home():
    return render_template('index.html')

# Route để xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    medInc = float(request.form['med-inc'])
    houseAge = float(request.form['house-age'])
    aveRooms = float(request.form['ave-rooms'])
    aveBedrms = float(request.form['ave-bedrms'])
    population = float(request.form['population'])
    aveOccup = float(request.form['ave-occup'])
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])


    # Dữ liệu mới từ người dùng
    new_data = [[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]]
    
    # Dự đoán với các mô hình
    prediction_linear = linear.predict(new_data)[0]
    prediction_lasso = lasso.predict(new_data)[0]
    prediction_mlp = mlp.predict(new_data)[0]
    prediction_stacking = stacking_regressor.predict(new_data)[0]
    
    
    # Trả về dữ liệu dự đoán, MSE và trạng thái của mô hình dưới dạng JSON
    return jsonify({
        'prediction_linear': prediction_linear,
        'prediction_lasso': prediction_lasso,
        'prediction_mlp': prediction_mlp,
        'prediction_stacking': prediction_stacking,
        'mse_linear': linearMSE,
        'mse_lasso': lassoMSE,
        'mse_mlp': mlpMSE,
        'mse_stacking': stackingMSE,
        'mae_linear': linearMAE,
        'mae_lasso': lassoMAE,
        'mae_mlp': mlpMAE,
        'mae_stacking': stackingMAE,
        'r2_linear': linearR2,
        'r2_lasso': lassoR2,
        'r2_mlp': mlpR2,
        'r2_stacking': stackingR2,
    })

if __name__ == '__main__':
    app.run(debug=True)