from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import models

linear, lasso, mlp, stacking = models.createModel()
pred_linear, pred_lasso, pred_mlp, pred_stacking = models.predict(linear, mlp, lasso, stacking)
models.graph(pred_linear, pred_lasso, pred_mlp, pred_stacking)

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
    new_data = [[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]]
    prediction_linear, prediction_lasso, prediction_mlp, prediction_stacking = models.predictFromUser(linear, lasso, mlp, stacking, new_data)
    linearMSE, lassoMSE, mlpMSE, stackingMSE = models.mse(pred_linear, pred_mlp, pred_lasso, pred_stacking)
    linearMAE, lassoMAE, mlpMAE, stackingMAE = models.mae(pred_linear, pred_mlp, pred_lasso, pred_stacking)
    linearR2, lassoR2, mlpR2, stackingR2 = models.r2(pred_linear, pred_mlp, pred_lasso, pred_stacking)

    
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