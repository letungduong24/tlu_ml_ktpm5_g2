param_grid_lasso = {
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
}

mlp = MLPRegressor(
    hidden_layer_sizes=(50,),         # Kích thước lớp ẩn
    max_iter=500,                      # Số vòng lặp tối đa
    early_stopping=True,               # Bật early stopping
    validation_fraction=0.1,           # Tỷ lệ dữ liệu dùng để validation
    n_iter_no_change=10,               # Số vòng lặp không thay đổi hiệu suất trước khi dừng
    random_state=1                     # Hạt giống để tái lập kết quả
)

base_models = [
    ('linear', LinearRegression()),
    ('lasso', lasso.best_estimator_),
    ('mlp', mlp)
]
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), n_jobs=-1) 