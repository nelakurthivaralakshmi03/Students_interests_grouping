from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)

            interest_cols = ['Tech', 'Music', 'Reading', 'Art', 'Coding', 'Sports']
            X = df[interest_cols]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)

            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            cluster_centers_df = pd.DataFrame(cluster_centers, columns=interest_cols)
            cluster_centers_df['Cluster'] = range(3)

            cluster_labels = {}
            for i in range(3):
                top2 = cluster_centers_df.loc[i, interest_cols].sort_values(ascending=False).index[:2]
                cluster_labels[i] = f"{top2[0]} + {top2[1]} Lovers"

            df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
            cluster_label_sizes = df.groupby('Cluster_Label').size().sort_values(ascending=False)
            most_populated_label = cluster_label_sizes.idxmax()
            most_count = cluster_label_sizes.max()

            result['labels'] = cluster_labels
            result['sizes'] = cluster_label_sizes.to_dict()
            result['most'] = f"{most_populated_label} ({most_count} students)"

            # Plot
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            df['PCA1'] = components[:, 0]
            df['PCA2'] = components[:, 1]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_Label', palette='Set2', s=100)
            plt.title('Student Interest Clusters')
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.legend(title='Group')
            plt.tight_layout()
            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png')
            plt.savefig(plot_path)
            plt.close()
            result['plot'] = plot_path

    return render_template('group.html', result=result)

# ✅ This is necessary to run the app
if __name__ == '__main__':
    app.run(debug=True)
