import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

class AnomalyVisualizer:
    def __init__(self, anomaly_df, incidents_df=None, score_col='scores', anomaly_col='anomaly'):
        """
        Initializes the AnomalyVisualizer.

        Parameters:
        - anomaly_df (DataFrame): DataFrame containing anomaly data.
        - incidents_df (DataFrame): DataFrame containing incident data (optional).
        - score_col (str): Column name for anomaly scores (default: 'scores').
        - anomaly_col (str): Column name for anomaly labels (default: 'anomaly').
        """
        self.anomaly_df = anomaly_df
        self.incidents_df = incidents_df
        self.score_col = score_col
        self.anomaly_col = anomaly_col

    def plot_static(self, zoom=False, zoom_date=None, 
                    colors=None, title=None, xlabel=None, ylabel=None, legend_labels=None):
        """
        Plots a static graph using matplotlib.

        Parameters:
        - zoom (bool): Whether to zoom into a specific date range.
        - zoom_date (list): List of two dates [start_date, end_date] for zooming.
        - colors (dict): Custom colors for 'normal', 'anomaly', and 'incident'.
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - legend_labels (list): Custom labels for the legend.
        """
        if colors is None:
            colors = {'normal': 'green', 'anomaly': 'red', 'incident': 'blue'}
        if legend_labels is None:
            legend_labels = ['Normal', 'Anomalía', 'Incidente']

        plt.figure(figsize=(12, 6))
        plt.plot(self.anomaly_df.index, self.anomaly_df[self.score_col], 'gray', alpha=0.5)
        plt.scatter(
            self.anomaly_df.index, self.anomaly_df[self.score_col],
            c=[colors['anomaly'] if x == -1 else colors['normal'] for x in self.anomaly_df[self.anomaly_col]],
            alpha=0.5
        )

        if self.incidents_df is not None:
            for _, row in self.incidents_df.iterrows():
                plt.axvspan(row['start_time'], row['end_time'], color=colors['incident'], alpha=0.1)

        plt.xlabel(xlabel or 'Tiempo')
        plt.ylabel(ylabel or 'Anomalías detectadas')
        plt.title(title or 'Anomalías detectadas vs incidentes')
        plt.xticks(rotation=45)
        plt.legend(legend_labels, loc='upper right')

        if zoom and zoom_date:
            plt.xlim(pd.Timestamp(zoom_date[0]), pd.Timestamp(zoom_date[1]))

        plt.show()

    def plot_dynamic(self, colors=None, title=None, xaxis_title=None, yaxis_title=None):
        """
        Plots a dynamic graph using Plotly.

        Parameters:
        - colors (dict): Custom colors for 'scores', 'anomalies', and 'incident'.
        - title (str): Title of the plot.
        - xaxis_title (str): Label for the x-axis.
        - yaxis_title (str): Label for the y-axis.
        """
        if colors is None:
            colors = {'scores': 'blue', 'anomalies': 'red', 'incident': 'orange'}

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.anomaly_df.index, y=self.anomaly_df[self.score_col], mode='lines+markers',
            name='Anomaly Scores', line=dict(color=colors['scores']),
            marker=dict(size=5, opacity=0.4)
        ))

        fig.add_trace(go.Scatter(
            x=self.anomaly_df[self.anomaly_df[self.anomaly_col] == -1].index, 
            y=self.anomaly_df.loc[self.anomaly_df[self.anomaly_col] == -1, self.score_col],
            mode='markers', name='Anomalies',
            marker=dict(color=colors['anomalies'], size=8)
        ))

        if self.incidents_df is not None:
            for _, row in self.incidents_df.iterrows():
                fig.add_vrect(
                    x0=row['start_time'], x1=row['end_time'],
                    fillcolor='orange', opacity=0.1,
                    annotation_text=row['Servicio'],
                )

        fig.update_layout(
            title=title or 'Puntajes de anomalías',
            xaxis_title=xaxis_title or 'Time',
            yaxis_title=yaxis_title or 'Anomaly Score',
            legend=dict(x=0.01, y=0.99)
        )

        fig.show()




class SHAPVisualizer:
    """
    A visualizer for SHAP values to explain the predictions of the IsolationForestDetector.

    Parameters
    ----------
    model : IsolationForestDetector
        The trained IsolationForestDetector model.
    """
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(self.model.model)

    def plot_force(self, X, anomaly_index):
        """
        Plots a SHAP force plot for a specific anomaly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        anomaly_index : int
            The index of the anomaly in the dataset X to visualize.
        """
        shap_values = self.explainer.shap_values(X)
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[anomaly_index],
            X.iloc[anomaly_index],
            matplotlib=True,
            figsize=(20, 5),
            text_rotation=70
        )

    def plot_summary(self, X):
        """
        Plots a SHAP summary plot for the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """
        shap_values = self.explainer.shap_values(X)
        shap.summary_plot(shap_values, X)

