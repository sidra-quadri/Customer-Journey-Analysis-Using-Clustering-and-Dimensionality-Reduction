from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

# Create an uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the app to allow file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to handle file upload and analysis
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # Handle POST request (file upload)
        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load and process the CSV file
            df = pd.read_csv(filepath)

            # Data processing logic
            funnel_data = df[['UserID', 'Yearly_avg_view_on_travel_page', 'Yearly_avg_comment_on_travel_page', 'Taken_product']].copy()
            funnel_data['Buy Product'] = (funnel_data['Taken_product'] == 'Yes').astype(int)
            funnel_data['View Travel Page'] = funnel_data['Yearly_avg_view_on_travel_page'].apply(lambda x: 1 if x > 0 else 0)
            funnel_data['Comment on Travel Page'] = funnel_data['Yearly_avg_comment_on_travel_page'].apply(lambda x: 1 if x > 0 else 0)
            
            funnel_percentages = funnel_data[['View Travel Page', 'Comment on Travel Page', 'Buy Product']].mean() * 100

            # Clean non-numeric values in the columns before clustering
            features_for_clustering = ['Yearly_avg_view_on_travel_page', 'yearly_avg_Outstation_checkins', 'Daily_Avg_mins_spend_on_traveling_page']

            # Convert columns to numeric, coerce errors to NaN, and drop rows with NaN
            df[features_for_clustering] = df[features_for_clustering].apply(pd.to_numeric, errors='coerce')

            # Drop rows with NaN values in the selected columns
            df_clustering = df[features_for_clustering].dropna()

            # Scaling the data
            X = StandardScaler().fit_transform(df_clustering)

            # Clustering
            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
            df_clustering['Cluster'] = kmeans.fit_predict(X)

            # Merging the cluster results back into the original DataFrame
            df['Cluster'] = df_clustering['Cluster']

            return render_template('index.html', funnel_percentages=funnel_percentages.to_dict(), clusters=df[['UserID', 'Cluster']].to_dict(orient='records'))

    # Default route behavior for GET requests
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
