import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
import threading
import queue
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('QT5Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt  # Import pyplot for plotting

class StreamingDataManager:
    """
    A class to manage streaming of data in chunks from a pandas DataFrame.
    Attributes:

        - data (pd.DataFrame): The DataFrame containing the data to be streamed.

        - chunk_size (int): The number of rows in each chunk. Default is 100.

        - stream_interval (int): The time interval (in seconds) between streaming chunks. Default is 1.

        - data_queue (queue.Queue): A queue to hold the streamed data chunks.

        - current_index (int): The current index in the DataFrame for streaming.

        - streaming_active (bool): A flag indicating whether streaming is active.

        - thread (threading.Thread): The thread responsible for streaming data.

    """
    def __init__(self, data: pd.DataFrame, chunk_size=100, stream_interval=1):
        self.data = data
        self.chunk_size = chunk_size
        self.stream_interval = stream_interval
        self.data_queue = queue.Queue()
        self.current_index = 0
        self.streaming_active = False
        self.thread = None
    
    def start(self):
        """
        Starts the streaming process in a separate thread.

        This method sets the `streaming_active` flag to True and initializes
        a daemon thread to execute the `_stream_data` method. The thread is
        started immediately after creation.

        Note:

            The thread runs as a daemon, meaning it will not block the program
            from exiting if the main thread finishes execution.
        """
        self.streaming_active = True
        self.thread = threading.Thread(target=self._stream_data, daemon=True)
        self.thread.start()
    
    def _stream_data(self):
        """
        Streams data in chunks from the provided dataset.

        This method continuously streams chunks of data from the `data` attribute 
        into a queue (`data_queue`) while `streaming_active` is True and there is 
        remaining data to stream. The size of each chunk is determined by 
        `chunk_size`, and the interval between streaming each chunk is controlled 
        by `stream_interval`.

        Attributes:

            - streaming_active (bool): A flag indicating whether streaming is active.

            - current_index (int): The current position in the dataset from which 
                data is being streamed.

            - data (pandas.DataFrame): The dataset to be streamed.

            - chunk_size (int): The number of rows to include in each streamed chunk.

            - data_queue (queue.Queue): The queue where the streamed data chunks are 
                placed.

            - stream_interval (float): The time interval (in seconds) between 
                streaming each chunk.

        Yields:

            None
        """
        while self.streaming_active and self.current_index < len(self.data):
            chunk = self.data.iloc[self.current_index:self.current_index + self.chunk_size]
            self.data_queue.put(chunk)
            self.current_index += self.chunk_size
            time.sleep(self.stream_interval)
    
    def get_next_chunk(self):
        """
        Retrieves the next chunk of data from the data queue.
        This method attempts to fetch the next item from the `data_queue` within
        the specified `stream_interval`. If the queue is empty and the timeout
        is reached, it returns `None`.

        Returns:

            Any: The next chunk of data from the queue if available, or `None` 
            if the queue is empty and the timeout is reached.

        Raises:

            queue.Empty: If the queue is empty and no timeout is specified.
        """

        try:
            return self.data_queue.get(timeout=self.stream_interval)
        except queue.Empty:
            return None
    
    def stop(self):
        """
        Stops the streaming process by setting the `streaming_active` flag to False.
        If a thread is running, it waits for the thread to complete using `join()`.
        """
        self.streaming_active = False
        if self.thread:
            self.thread.join()


class AnomalyDetector:
    def __init__(self, window_size=120):
        self.window_size = window_size
        self.model = IsolationForest(n_estimators=250, contamination=0.05, random_state=42)
        self.trained = False  # Track whether the model is trained

    def fit(self, data: np.ndarray):
        self.model.fit(data)
        self.trained = True  # Mark the model as trained

    def detect(self, data: np.ndarray):
       
        # Perform predictions and calculate anomaly scores
        predictions = self.model.predict(data)
        scores = self.model.decision_function(data)

        # Check if all variables in the data are 0
        for i, element in enumerate(data):
            if np.all(element == 0):
                # Generate a random anomaly score between 0 and 0.05
                random_score = np.random.uniform(0, 0.05)
                predictions[i] = 1  # Mark as normal (1)
                scores[i] = -random_score
            # Store the last element of the previous chunk for comparison
            if hasattr(self, 'last_element'):
                if np.all(self.last_element == 0) and predictions[0] == -1:
                    # If the last element of the previous chunk is all 0 and the first of the current chunk is an anomaly
                    random_score = np.random.uniform(0, 0.05)
                    predictions[0] = 1  # Mark as normal (1)
                    scores[0] = -random_score

            # Check for anomalies within the current chunk
            for i, element in enumerate(data):
                if np.all(element == 0):
                    # Generate a random anomaly score between 0 and 0.05
                    random_score = np.random.uniform(0, 0.05)
                    predictions[i] = 1  # Mark as normal (1)
                    scores[i] = -random_score
                elif i > 0 and np.all(data[i - 1] == 0) and predictions[i] == -1:
                    # If the previous element is all 0 and the current is an anomaly
                    random_score = np.random.uniform(0, 0.05)
                    predictions[i] = 1  # Mark as normal (1)
                    scores[i] = -random_score
                elif i < len(data) - 1 and predictions[i] == -1 and np.all(data[i + 1] == 0):
                    # If the current element is an anomaly and the next element is all 0
                    random_score = np.random.uniform(0, 0.05)
                    predictions[i] = 1  # Mark as normal (1)
                    scores[i] = -random_score

            # Save the last element of the current chunk for the next comparison
            self.last_element = data[-1]
        return predictions, scores

class StreamingSimulation:
    """
        Simulates real-time data streaming, processes data for anomaly detection, and visualizes results.

        Attributes:

            - window_size (int): The size of the sliding window for anomaly detection.

            - threshold (float): The static threshold for anomaly detection.

            - dynamic_threshold (bool): Whether to use a dynamic threshold based on historical scores.

            - percentile (int): The percentile used for calculating the dynamic threshold.

            - manager (StreamingDataManager): Manages the streaming of data chunks.

            - detector (AnomalyDetector): Detects anomalies in the data.

            - historical_data (pd.DataFrame): Stores historical data for processing.

            - historical_scores (list): Stores historical anomaly scores.

            - data_source (pd.DataFrame): The source data for streaming.

            - chunk_size (int): The size of each data chunk for streaming.

            - stream_interval (int): The interval (in seconds) between streaming chunks.

            - queue (queue.Queue): Queue for storing data chunks to be processed.

            - plot_queue (queue.Queue): Queue for storing data to be plotted.

            - streaming_active (bool): Indicates whether the streaming simulation is active.

            - events (pd.DataFrame): DataFrame containing event start and end times, colors, and labels.
    """
    def __init__(self, data: pd.DataFrame, chunk_size=100, stream_interval=1, window_size=120, threshold=0.15, dynamic_threshold=False, percentile=95, events=None):
        self.window_size = window_size
        self.threshold = threshold
        self.dynamic_threshold = dynamic_threshold
        self.percentile = percentile
        self.manager = StreamingDataManager(data, chunk_size, stream_interval)
        self.detector = AnomalyDetector(self.window_size)
        self.historical_data = pd.DataFrame()
        self.historical_scores = []
        self.data_source = data
        self.chunk_size = chunk_size
        self.stream_interval = stream_interval
        self.queue = queue.Queue()
        self.plot_queue = queue.Queue()
        self.streaming_active = False
        self.events = events  # DataFrame with event start and end times

    def preprocess(self, chunk: pd.DataFrame):
        """
        Preprocesses a chunk of data by selecting numeric columns and filling NaN values with 0.

        Args:

            - chunk (pd.DataFrame): The input DataFrame containing the data to preprocess.

        Returns:

            np.ndarray: A NumPy array containing the preprocessed numeric data.
        """
        numeric_data = chunk.select_dtypes(include=[np.number]).fillna(0)
        return numeric_data.values

    def _calculate_dynamic_threshold(self):
        """
        Calculate the dynamic threshold based on historical anomaly scores.

        This method computes a dynamic threshold using the specified percentile
        of the historical anomaly scores. If no historical scores are available,
        it falls back to a predefined static threshold.

        Returns:

            float: The calculated dynamic threshold based on the percentile of
            historical scores, or the static threshold if no scores are available.
        """
        if len(self.historical_scores) > 0:
            return np.percentile(self.historical_scores, self.percentile)
        return self.threshold  # Fallback to static threshold if no scores are available

    def _stream_data(self):
        """
        Streams data from the data source in chunks and places each chunk into a queue.
        This method iterates over the data source in increments of `chunk_size` and 
        streams each chunk into a queue for further processing. The streaming process 
        can be controlled using the `streaming_active` flag, and a delay between 
        chunks is introduced using `stream_interval`.

        Yields:

            None: This method does not return any value but streams data chunks 
            into the queue.

        Attributes:

            - data_source (pd.DataFrame): The source of data to be streamed.

            - chunk_size (int): The size of each data chunk to be streamed.

            - queue (queue.Queue): The queue where data chunks are placed.

            - stream_interval (float): The time interval (in seconds) between streaming 
                consecutive chunks.

            - streaming_active (bool): A flag to control the streaming process. If set 
                to False, the streaming stops.

        """
       
        for i in range(0, len(self.data_source), self.chunk_size):
            if not self.streaming_active:
                break
            chunk = self.data_source.iloc[i:i + self.chunk_size]
            self.queue.put(chunk)
            time.sleep(self.stream_interval)

    def _plot_from_main_thread(self):
        """
        Continuously plots anomaly detection results and events in real-time from the main thread.
        This method is designed to run in a loop, updating the plot with data from a queue until
        the streaming process is deactivated and the queue is empty. It visualizes anomaly scores,
        thresholds, and events on a time series plot.

        Key Features:

            - Uses Matplotlib to create a real-time plot of anomaly scores.

            - Displays anomaly scores as a line plot.

            - Highlights anomalies exceeding the threshold with red scatter points.

            - Dynamically calculates the threshold if enabled, otherwise uses a static threshold.

            - Visualizes events as vertical lines and shaded regions with customizable labels and colors.

        Interactive Mode:

                - Uses Matplotlib's interactive mode (`plt.ion()`) to update the plot in real-time.

                - Ensures the plot is cleared and updated with new data during each iteration.

        Cleanup:
                - Disables interactive mode (`plt.ioff()`) and closes all figures to free memory after
                the loop ends.

        Raises:
            queue.Empty: If the plot queue is empty and no data is available within the timeout.

        Notes:
            - This method assumes `self.plot_queue` is a thread-safe queue containing tuples of
              filtered data and anomaly scores.

            - The `self.events` attribute should be a DataFrame with columns 'start', 'end',
              'color', and 'label' to define event visualization.
        """
        
        plt.ion()  # Enable interactive mode
        while self.streaming_active or not self.plot_queue.empty():
            try:
                filtered_data, filtered_scores = self.plot_queue.get(timeout=1)
                plt.clf()  # Clear the current figure
                plt.figure(figsize=(12, 6))
                plt.plot(filtered_data.index, filtered_scores * -1, label='Anomaly Scores', color='blue')

                # Determine the threshold (static or dynamic)
                current_threshold = self._calculate_dynamic_threshold() if self.dynamic_threshold else self.threshold
                plt.axhline(y=current_threshold, color='cyan', linestyle='--', label='Threshold')

                # Highlight anomalies
                for i in range(len(filtered_scores)):
                    if filtered_scores[i] * -1 > current_threshold:
                        plt.scatter(filtered_data.index[i], filtered_scores[i] * -1, color='red')

                # Plot events as vertical lines
                if self.events is not None:
                    for _, event in self.events.iterrows():
                        event_start = pd.to_datetime(event['start'])
                        event_end = pd.to_datetime(event['end'])
                        color = event['color']
                        label = event['label']
                        if event_start.date() == filtered_data.index[-1].date():
                            plt.axvline(x=event_start, color=color, linestyle='--', label=label)
                            plt.axvspan(event_start, event_end, color=color, alpha=0.3)
                            plt.axvline(x=event_end, color=color, linestyle='--')

                plt.xlabel('Time')
                plt.ylabel('Anomaly Score')
                plt.title('Anomaly Detection with Events')
                plt.xlim(filtered_data.index[0], filtered_data.index[-1])
                plt.legend()
                plt.pause(0.01)
            except queue.Empty:
                continue
        plt.ioff()  # Disable interactive mode
        plt.show()
        plt.close('all')  # Close all figures to free memory

    def process_stream(self):
        """
        Consume data from the queue and process it with the anomaly detector.

        This method continuously retrieves data chunks from a queue, processes them,
        and performs anomaly detection using a pre-trained detector. The results are
        then prepared for visualization and sent to a plotting queue.

        Workflow:

            1- Retrieve a chunk of data from the queue.

            2- Append the chunk to the historical data, maintaining a rolling window of the last 5000 records.

            3- Preprocess the data for anomaly detection.

            4- Train the anomaly detector if it is not already trained and sufficient data is available.

            5- Perform anomaly detection on the current chunk if the detector is trained.

            6- Update historical scores and count the number of anomalies detected in the current chunk.

            7- Filter historical data and scores to include only the last three hours of data.

            8- Send the filtered data and scores to the plotting queue for visualization.

        Exceptions:

            - Handles `queue.Empty` exceptions when the queue is empty and continues processing.

            - Catches and logs any other exceptions that occur during processing.

        Attributes:

            - self.streaming_active (bool): Flag to control the streaming process.

            - self.queue (queue.Queue): Queue from which data chunks are consumed.

            - self.historical_data (pd.DataFrame): DataFrame storing historical data for processing.

            - self.historical_scores (np.ndarray): Array storing historical anomaly scores.

            - self.window_size (int): Size of the rolling window for preprocessing.

            - self.detector (object): Anomaly detector instance with `fit` and `detect` methods.

            - self.plot_queue (queue.Queue): Queue to send data and scores for visualization.
        """
        
        while self.streaming_active:
            try:
                chunk = self.queue.get(timeout=1)
                if chunk is not None:
                    self.historical_data = pd.concat([self.historical_data, chunk]).iloc[-5000:]
                    processed_data = self.preprocess(self.historical_data.iloc[-self.window_size:])

                    if not self.detector.trained and len(processed_data) >= self.detector.window_size:
                        self.detector.fit(processed_data)

                    if self.detector.trained:
                        predictions, scores = self.detector.detect(self.preprocess(chunk))
                        self.historical_scores = np.append(self.historical_scores, scores)

                        anomaly_count = np.sum(predictions == -1)
                        print(f"Detected {anomaly_count} anomalies in the current chunk.")

                        self.historical_data.index = pd.to_datetime(self.historical_data.index)
                        three_hours_ago = self.historical_data.index[-1] - pd.Timedelta(hours=3)
                        filtered_data = self.historical_data[self.historical_data.index >= three_hours_ago]

                        filtered_scores = self.historical_scores[-len(filtered_data):] if len(self.historical_scores) >= len(filtered_data) else np.pad(self.historical_scores, (len(filtered_data) - len(self.historical_scores), 0), 'constant', constant_values=0)

                        self.plot_queue.put((filtered_data, filtered_scores))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing stream: {e}")

    def run(self):
        """
        Executes the streaming anomaly detection simulation.

        This method starts two separate threads:

            1. A streaming thread that simulates or handles incoming data.

            2. A processing thread that processes the streamed data.

        The simulation remains active until interrupted by a KeyboardInterrupt
        (e.g., pressing Ctrl+C). Upon interruption, the simulation stops gracefully
        by setting `self.streaming_active` to False and joining both threads.
        Additionally, this method handles plotting from the main thread.
        
        Raises:
        
            KeyboardInterrupt: If the simulation is manually stopped by the user.
        """
       
        self.streaming_active = True

        stream_thread = threading.Thread(target=self._stream_data, daemon=True)
        process_thread = threading.Thread(target=self.process_stream, daemon=True)

        stream_thread.start()
        process_thread.start()

        try:
            self._plot_from_main_thread()
        except KeyboardInterrupt:
            print("Stopping simulation...")
            self.streaming_active = False

        stream_thread.join()
        process_thread.join()


if __name__ == "__main__":
    data_source = pd.read_csv('../../docs/notebooks/data/PDTI_Feb_11_2025.csv', index_col=0).sort_index()
    
    # Example events DataFrame
    events = pd.DataFrame({
        'start': ['2025-02-11 22:00:00', '2025-02-11 23:59:00'],
        'end': ['2025-02-12 00:00:00', '2025-02-12 00:11:00'],
        'color':['orange','red'],
        'label':['Reinicio Caja 1 HSM', 'Incidente Mi Bancolombia, entre otros']
    })

    dashboard = StreamingSimulation(data_source, chunk_size=50, stream_interval=1, window_size=100, threshold=0.14, dynamic_threshold=False, events=events)
    dashboard.run()

